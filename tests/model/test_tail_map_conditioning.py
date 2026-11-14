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

"""Equality and interval conditioning of a Student-t mixture with a per-column tail map."""

import math

import numpy as np
import pytest
import torch
from scipy import integrate

from jointfm_inference.model.inference import PredictorAdapter
from jointfm_inference.model.inference.conditional_query import (
    ConditionRequest,
    condition_at_position,
)
from jointfm_inference.model.jointfm_1.head import StudentTHead
from jointfm_inference.model.jointfm_1.tail_map import TAIL_MAP_KEYS

from tests.support.condition_mode import make_payload
from tests.support.models import tiny_config, tiny_model
from tests.support.service import mount_service
from tests.support.tail_map import mapped_mixture, oracle_column_cdf, oracle_log_prob

VALUES = [1.2, -3.5, 4.0]
# Conditioning stabilizes each Schur complement with a small jitter, which moves
# a conditional log-density by a few 1e-9; a missing Jacobian or a swapped side
# moves it by whole nats.
CONDITIONAL_TOLERANCE = 1e-7


def _adapter() -> PredictorAdapter:
    """Return an adapter over :func:`mapped_mixture` with a head that has a map."""
    head = StudentTHead(
        d_model=4,
        D=3,
        K=2,
        trunk_layers=1,
        trunk_hidden=4,
        covariance_rank=1,
        min_df=3.0,
        min_scale=1e-4,
        tail_parameter_bounds=(0.2, 1.4),
    )
    return PredictorAdapter(head, mapped_mixture())


def _condition(adapter: PredictorAdapter, known: list[int]) -> PredictorAdapter:
    """Condition ``adapter`` on :data:`VALUES` at the columns ``known``."""
    mask = torch.zeros(3, dtype=torch.bool)
    mask[known] = True
    return adapter.condition(torch.tensor([[VALUES]], dtype=torch.float64), mask)


@pytest.mark.parametrize("known", [[0], [2], [0, 2], [1, 2]])
def test_the_pinned_density_and_the_conditional_chain_to_the_joint(
    known: list[int],
) -> None:
    """``log p(y_P) + log p(y_Q | y_P) = log p(y)``, with ``p(y_P)`` the oracle's marginal."""
    adapter = _adapter()
    reduced = _condition(adapter, known)
    unknown = [column for column in range(3) if column not in known]

    pinned = float(torch.logsumexp(reduced.params["logits"], dim=-1))
    conditional = float(
        reduced.log_prob(
            torch.tensor([[[VALUES[c] for c in unknown]]], dtype=torch.float64)
        )
    )
    joint = float(adapter.log_prob(torch.tensor([[VALUES]], dtype=torch.float64)))

    assert pinned == pytest.approx(
        oracle_log_prob(mapped_mixture(), [VALUES[c] for c in known], known),
        abs=CONDITIONAL_TOLERANCE,
    )
    assert pinned + conditional == pytest.approx(joint, abs=CONDITIONAL_TOLERANCE)
    assert all(key in reduced.params for key in TAIL_MAP_KEYS)
    assert tuple(reduced.params["tail_upper"].shape) == (1, 1, len(unknown))


def test_successive_conditions_keep_the_joint_anchors() -> None:
    """Conditioning twice equals conditioning once on both columns."""
    adapter = _adapter()
    once = _condition(adapter, [0, 2])
    first = _condition(adapter, [0])
    second = first.condition(
        torch.tensor([[[VALUES[1], VALUES[2]]]], dtype=torch.float64),
        torch.tensor([False, True]),
    )

    for key in TAIL_MAP_KEYS:
        assert torch.allclose(second.params[key], once.params[key])
    probe = torch.tensor([[[0.4]]], dtype=torch.float64)
    assert torch.allclose(second.log_prob(probe), once.log_prob(probe), atol=1e-9)


def test_the_conditional_mean_is_the_mean_of_conditional_draws() -> None:
    """The exact conditional mean agrees with many conditional draws within their error."""
    reduced = _condition(_adapter(), [0])
    draws = reduced.sample(n_samples=200_000, rng=torch.Generator().manual_seed(11))

    mean = reduced.mean()[0, 0]
    drawn = draws[0, :, 0].mean(dim=0)
    standard_error = draws[0, :, 0].std(dim=0) / math.sqrt(draws.shape[1])

    assert bool(((mean - drawn).abs() <= 5.0 * standard_error).all())


def _interval_request(
    lower: float, upper: float, *, pin: bool
) -> tuple[ConditionRequest, PredictorAdapter]:
    """Return a request bounding column 1, optionally pinning column 0, and its adapter."""
    request = ConditionRequest(
        horizon=0,
        equality_columns=(0,) if pin else (),
        equality_values=(VALUES[0],) if pin else (),
        interval_columns=(1,),
        interval_lower=(lower,),
        interval_upper=(upper,),
    )
    return request, _adapter()


@pytest.mark.parametrize(
    ("lower", "upper"), [(-4.0, -1.0), (2.0, math.inf), (-math.inf, -6.0), (0.5, 9.0)]
)
def test_a_one_column_region_probability_is_the_oracle_distribution_difference(
    lower: float, upper: float
) -> None:
    """The latent box of one column holds exactly the data box's probability."""
    request, adapter = _interval_request(lower, upper, pin=False)

    read_out = condition_at_position(
        adapter,
        request,
        n_samples=4000,
        column_count=3,
        rng=torch.Generator().manual_seed(4),
    )

    params = mapped_mixture()
    probability = oracle_column_cdf(params, 1, upper) - oracle_column_cdf(
        params, 1, lower
    )
    assert read_out.region_log_probability is not None
    assert float(read_out.region_log_probability) == pytest.approx(
        math.log(probability), abs=CONDITIONAL_TOLERANCE
    )
    drawn = read_out.samples()[0, :, 0, 1]
    assert bool(((drawn >= lower) & (drawn <= upper)).all())


def test_interval_draws_follow_the_truncated_mapped_law() -> None:
    """Draws inside a bounded region fall below its midpoint at the oracle's rate."""
    lower, upper = -4.0, 3.0
    request, adapter = _interval_request(lower, upper, pin=False)
    draw_count = 20_000
    read_out = condition_at_position(
        adapter,
        request,
        n_samples=draw_count,
        column_count=3,
        rng=torch.Generator().manual_seed(6),
    )

    params = mapped_mixture()
    middle = 0.0
    expected = (
        oracle_column_cdf(params, 1, middle) - oracle_column_cdf(params, 1, lower)
    ) / (oracle_column_cdf(params, 1, upper) - oracle_column_cdf(params, 1, lower))
    share = float((read_out.samples()[0, :, 0, 1] <= middle).double().mean())
    assert share == pytest.approx(
        expected, abs=4.0 * math.sqrt(expected * (1.0 - expected) / draw_count)
    )


def test_a_region_after_a_pin_has_the_oracle_conditional_probability() -> None:
    """The box probability given the pins integrates the oracle's joint over the pinned density."""
    lower, upper = -2.0, 5.0
    request, adapter = _interval_request(lower, upper, pin=True)

    read_out = condition_at_position(
        adapter,
        request,
        n_samples=2000,
        column_count=3,
        rng=torch.Generator().manual_seed(8),
    )

    params = mapped_mixture()
    pinned = oracle_log_prob(params, [VALUES[0]], [0])
    probability, _ = integrate.quad(
        lambda y: math.exp(oracle_log_prob(params, [VALUES[0], y], [0, 1]) - pinned),
        lower,
        upper,
        epsabs=1e-13,
    )
    assert read_out.equality_log_density is not None
    assert float(read_out.equality_log_density) == pytest.approx(pinned, abs=1e-9)
    assert read_out.region_log_probability is not None
    assert float(read_out.region_log_probability) == pytest.approx(
        math.log(probability), abs=CONDITIONAL_TOLERANCE
    )
    assert np.isfinite(read_out.mean().numpy()).all()


def test_a_mapped_network_serves_every_return_mode_and_both_condition_kinds() -> None:
    """A real network with a non-identity map answers forecasts and conditions with finite numbers."""
    model = tiny_model(
        tiny_config(
            head_min_df=3.0, head_tail_parameter_min=0.2, head_tail_parameter_max=1.4
        )
    )
    assert model.head.column_tail is not None
    with torch.no_grad():
        model.head.column_tail.output.weight.normal_(0.0, 2.0)
    service = mount_service(model, max_sample_count=1000)
    forecast = make_payload(query_mode="forecast", n_samples=64)
    del forecast["conditions"]

    mean = service.predict(forecast)["outputs"]["mean"]
    samples = service.predict({**forecast, "return_mode": "samples"})["outputs"][
        "samples"
    ]
    [median] = service.predict(
        {**forecast, "return_mode": "quantiles", "quantiles": [0.5]}
    )["outputs"]["quantiles"]
    score = service.predict(
        {
            **forecast,
            "return_mode": "log_prob",
            "requested_columns": ["driver", "target"],
            "query_rows": [
                {"timestamp": time, "driver": 1.5, "target": 12.0}
                for time in forecast["query_times"]
            ],
        }
    )["outputs"]["log_prob"]
    condition = make_payload(
        return_mode="samples",
        n_samples=64,
        conditions=[
            {
                "column": "driver",
                "kind": "interval",
                "lower": 1.0,
                "upper": 2.0,
                "query_time_indices": [0],
            },
        ],
    )
    conditioned = service.predict(condition)
    pinned = service.predict(make_payload())

    assert np.isfinite(np.asarray(mean)).all()
    assert np.asarray(samples).shape == (64, 2, 1)
    assert np.isfinite(np.asarray(samples)).all()
    assert np.isfinite(np.asarray(median["values"])).all()
    assert np.isfinite(score["total"])
    assert np.isfinite(np.asarray(conditioned["outputs"]["samples"])).all()
    assert math.isfinite(conditioned["plausibility"]["region_log_probability"])
    assert math.isfinite(pinned["plausibility"]["equality_log_density"])
