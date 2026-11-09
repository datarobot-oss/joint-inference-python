# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sign-symmetric prediction of the JointFM 1 backend.

With ``sign_symmetric_prediction`` on, the backend predicts with the
equal-weight mixture of the network's forecast for the history and the
negated forecast for the negated history. That mixture must be exactly
invariant under negating every column, which a randomly initialized network
is not, and every read-out (sampling, scoring, conditioning) must use it.
With the switch off, the backend must predict with the plain forward.
"""

from dataclasses import asdict
from typing import Any

import pytest
import torch

from jointfm_inference.model.inference.adapter import PredictorAdapter
from jointfm_inference.model.inference.jointfm_1_backend import (
    JointFM1InferenceBackend,
)
from jointfm_inference.model.jointfm_1.config import JointFMConfig
from jointfm_inference.model.jointfm_1.head import (
    AUTO_STANDARDIZATION_METADATA_KEYS,
)
from jointfm_inference.model.jointfm_1.loader import load_model_from_checkpoint
from jointfm_inference.model.jointfm_1.model import JointFM
from jointfm_inference.model.jointfm_1.sign_symmetry import symmetric_mixture
from jointfm_inference.model.service import DEFAULT_SCHEMA_VERSION
from tests.support.models import (
    TINY_CONFIG,
    checkpoint_mapping,
    tiny_config,
    tiny_model,
)
from tests.support.service import mount_service

BATCH, STEPS, COLUMNS, HORIZONS = 2, 10, 3, 4
K = TINY_CONFIG.n_mixtures
HEAD_KEYS = ("logits", "means", "df", "lowrank_diag", "lowrank_U")
SHARED_KEYS = ("df", "lowrank_diag", "lowrank_U")
# A non-zero location readout gives every column its own term, so the
# network's response to negation is far from odd.
READOUT_WEIGHT_STD = 0.5
READOUT_SEED = 7
SAMPLE_COUNT = 64
SAMPLE_SEED = 5
# The doubled mixture orders its halves differently for x and -x, so the two
# log-sum-exps agree only to float32 rounding.
SYMMETRY_TOLERANCE = 1e-4
# A gap the invariance test must see without the mixture, so it is not vacuous.
RAW_ASYMMETRY_FLOOR = 1e-2


def _model(*, symmetric: bool) -> JointFM:
    """Return the tiny network with a random location readout, in eval mode."""
    model = tiny_model(tiny_config(sign_symmetric_prediction=symmetric))
    with torch.no_grad():
        model.head.column_location.readout.output.weight.normal_(
            std=READOUT_WEIGHT_STD,
            generator=torch.Generator().manual_seed(READOUT_SEED),
        )
    return model


def _backend(model: JointFM) -> JointFM1InferenceBackend:
    """Mount ``model`` on CPU at full precision."""
    return JointFM1InferenceBackend(
        model,
        checkpoint_mapping(model),
        device=torch.device("cpu"),
        autocast_dtype=None,
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a trending history, its horizons, and one candidate future."""
    generator = torch.Generator().manual_seed(1)
    ramp = torch.linspace(0.0, 3.0, STEPS).view(1, STEPS, 1)
    past = 5.0 + ramp + 0.3 * torch.randn(BATCH, STEPS, COLUMNS, generator=generator)
    future = past[:, -1:, :] + 0.5 * torch.randn(
        BATCH, HORIZONS, COLUMNS, generator=generator
    )
    return past, torch.arange(HORIZONS), future


def _predict(model: JointFM, past: torch.Tensor) -> PredictorAdapter:
    """Return the backend's predictive distribution for ``past``."""
    return _backend(model).predict_distribution(past, torch.arange(HORIZONS))


class _CountingForward:
    """Wrap a network's forward, recording the history of every call."""

    def __init__(self, model: JointFM) -> None:
        """Bind the real forward of ``model``."""
        self._forward = model.forward
        self.histories: list[torch.Tensor] = []

    def __call__(
        self,
        past: torch.Tensor,
        horizons: torch.Tensor,
        past_time_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Record ``past``, then run the real forward."""
        self.histories.append(past.clone())
        return self._forward(past, horizons, past_time_indices=past_time_indices)


def test_the_log_density_is_invariant_under_negating_history_and_future() -> None:
    """Scoring y after x equals scoring -y after -x, which the plain forward fails."""
    past, _, future = _inputs()

    symmetric = _model(symmetric=True)
    raw = _model(symmetric=False)
    raw_gap = (
        _predict(raw, past).log_prob(future) - _predict(raw, -past).log_prob(-future)
    ).abs()

    torch.testing.assert_close(
        _predict(symmetric, past).log_prob(future),
        _predict(symmetric, -past).log_prob(-future),
        rtol=SYMMETRY_TOLERANCE,
        atol=SYMMETRY_TOLERANCE,
    )
    assert float(raw_gap.max()) > RAW_ASYMMETRY_FLOOR


def test_the_distribution_is_the_doubled_mixture_of_both_forwards() -> None:
    past, horizons, _ = _inputs()
    model = _model(symmetric=True)
    forward = _CountingForward(model)
    model.forward = forward  # ty: ignore[invalid-assignment]

    params = _predict(model, past).params
    with torch.no_grad():
        direct = forward(past, horizons)
        mirrored = forward(-past, horizons)

    assert len(forward.histories) == 4
    torch.testing.assert_close(forward.histories[1], -past)
    weights = torch.softmax(params["logits"], dim=-1)
    assert weights.shape == (BATCH, HORIZONS, 2 * K)
    torch.testing.assert_close(
        weights[..., :K].sum(dim=-1), torch.full((BATCH, HORIZONS), 0.5)
    )
    torch.testing.assert_close(
        weights[..., :K], 0.5 * torch.softmax(direct["logits"], dim=-1)
    )
    torch.testing.assert_close(params["means"][..., :K, :], direct["means"])
    torch.testing.assert_close(params["means"][..., K:, :], -mirrored["means"])
    for key in SHARED_KEYS:
        torch.testing.assert_close(params[key][:, :, :K], direct[key])
        torch.testing.assert_close(params[key][:, :, K:], mirrored[key])
    for key in AUTO_STANDARDIZATION_METADATA_KEYS:
        torch.testing.assert_close(params[key], direct[key])


def test_samples_are_drawn_from_the_doubled_mixture() -> None:
    past, horizons, _ = _inputs()
    model = _model(symmetric=True)
    with torch.no_grad():
        mixture = symmetric_mixture(model(past, horizons), model(-past, horizons))

    draws = _predict(model, past).sample(
        SAMPLE_COUNT, rng=torch.Generator().manual_seed(SAMPLE_SEED)
    )
    expected = model.head.sample(
        mixture,
        num_samples=SAMPLE_COUNT,
        rng=torch.Generator().manual_seed(SAMPLE_SEED),
    )

    assert draws.shape == (BATCH, SAMPLE_COUNT, HORIZONS, COLUMNS)
    torch.testing.assert_close(draws.reshape(expected.shape), expected)


def test_conditioning_on_a_known_column_keeps_the_symmetry() -> None:
    """The exact conditional of the doubled mixture is invariant under negation too."""
    past, _, future = _inputs()
    model = _model(symmetric=True)
    known = torch.tensor([True, False, False])

    def conditional_log_density(sign: float) -> torch.Tensor:
        """Score the unknown columns of ``sign * future`` given its known column."""
        conditional = _predict(model, sign * past).condition(sign * future, known)
        assert conditional.params["logits"].shape[-1] == 2 * K
        return conditional.log_prob(sign * future[..., ~known])

    torch.testing.assert_close(
        conditional_log_density(1.0),
        conditional_log_density(-1.0),
        rtol=SYMMETRY_TOLERANCE,
        atol=SYMMETRY_TOLERANCE,
    )


def test_switched_off_the_backend_predicts_with_the_plain_forward() -> None:
    past, horizons, _ = _inputs()
    model = _model(symmetric=False)
    forward = _CountingForward(model)
    model.forward = forward  # ty: ignore[invalid-assignment]

    adapter = _predict(model, past)
    with torch.no_grad():
        plain = forward(past, horizons)

    assert len(forward.histories) == 2
    assert adapter.params.keys() == plain.keys()
    for key, value in plain.items():
        assert torch.equal(adapter.params[key], value)


def _service_payload(sign: float, **overrides: Any) -> dict[str, Any]:
    """Build a request over two numeric columns whose every value is ``sign`` times the base."""
    payload: dict[str, Any] = {
        "schema_version": DEFAULT_SCHEMA_VERSION,
        "query_mode": "forecast",
        "return_mode": "log_prob",
        "time_index_mode": "absolute_datetime",
        "time_column": "timestamp",
        "columns": [
            {"name": "driver", "modality": "numeric"},
            {"name": "target", "modality": "numeric", "role": "target"},
        ],
        "history_rows": [
            {
                "timestamp": f"2026-01-0{day}T00:00:00Z",
                "driver": sign * (1.0 + day / 10 + (day % 2) / 20),
                "target": sign * (10.0 + day),
            }
            for day in range(1, 7)
        ],
        "query_times": ["2026-01-07T00:00:00Z", "2026-01-08T00:00:00Z"],
        "requested_columns": ["driver", "target"],
        "seed": 7,
    }
    payload.update(overrides)
    return payload


def test_a_served_score_is_invariant_under_negating_every_series() -> None:
    service = mount_service(_model(symmetric=True))

    def total_score(sign: float) -> float:
        """Return the served joint log-density of the negated or plain request."""
        query_rows = [
            {"timestamp": time, "driver": sign * 1.7, "target": sign * 17.5}
            for time in ("2026-01-07T00:00:00Z", "2026-01-08T00:00:00Z")
        ]
        response = service.predict(_service_payload(sign, query_rows=query_rows))
        return float(response["outputs"]["log_prob"]["total"])

    assert total_score(1.0) == pytest.approx(total_score(-1.0), abs=SYMMETRY_TOLERANCE)


def test_a_served_interval_condition_is_invariant_under_negating_every_series() -> None:
    """The single-column interval walk is exact, so both regions read one probability."""
    service = mount_service(_model(symmetric=True))
    lower, upper = 1.6, 1.9

    def region_log_probability(sign: float) -> float:
        """Return the served log-probability of the driver's interval at position 0."""
        bounds = sorted((sign * lower, sign * upper))
        condition = {
            "column": "driver",
            "kind": "interval",
            "lower": bounds[0],
            "upper": bounds[1],
            "query_time_indices": [0],
        }
        response = service.predict(
            _service_payload(
                sign,
                query_mode="condition",
                return_mode="mean",
                requested_columns=["target"],
                conditions=[condition],
            )
        )
        return float(response["plausibility"]["region_log_probability"])

    assert region_log_probability(1.0) == pytest.approx(
        region_log_probability(-1.0), abs=SYMMETRY_TOLERANCE
    )


def test_a_cfg_without_the_switch_loads_switched_off() -> None:
    raw_cfg = asdict(tiny_config(sign_symmetric_prediction=True))
    del raw_cfg["sign_symmetric_prediction"]

    assert not JointFMConfig.from_mapping(raw_cfg).sign_symmetric_prediction


def test_a_switched_on_checkpoint_loads_and_serves_the_doubled_mixture() -> None:
    past, _, _ = _inputs()
    model, cfg = load_model_from_checkpoint(checkpoint_mapping(_model(symmetric=True)))

    assert cfg.sign_symmetric_prediction
    assert _predict(model, past).params["logits"].shape[-1] == 2 * K


def test_a_non_bool_switch_is_refused() -> None:
    with pytest.raises(TypeError, match="sign_symmetric_prediction must be a bool"):
        tiny_config(sign_symmetric_prediction=1)


def test_the_mixture_refuses_keys_without_a_sign_symmetric_form() -> None:
    past, horizons, _ = _inputs()
    model = _model(symmetric=True)
    with torch.no_grad():
        direct = model(past, horizons)
    dense = {**direct, "chol": torch.zeros(1)}

    with pytest.raises(ValueError, match=r"\['chol'\] have no sign-symmetric form"):
        symmetric_mixture(dense, dense)
    with pytest.raises(ValueError, match="direct and mirrored forward outputs differ"):
        symmetric_mixture(direct, dense)
