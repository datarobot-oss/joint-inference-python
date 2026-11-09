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

"""Exact means and marginal quantiles of the JointFM 1 Student-t mixture."""

import math
from typing import Any

import pytest
import torch
from scipy import optimize, stats

from jointfm_inference.model.inference import PredictorAdapter
from jointfm_inference.model.jointfm_1.head import (
    AUTO_STANDARDIZATION_CENTER_KEY,
    AUTO_STANDARDIZATION_SCALE_KEY,
    StudentTHead,
)
from jointfm_inference.model.jointfm_1.marginals import (
    QUANTILE_TOLERANCE,
    mixture_mean,
    mixture_quantiles,
)

from tests.support.condition_mode import make_payload, random_weight_service
from tests.support.models import tiny_config, tiny_model
from tests.support.parity import assert_exact_summaries_match_draws

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)


def _three_component_mixture() -> dict[str, torch.Tensor]:
    """Three components over two columns that differ in location, scale, and df.

    The components overlap only partly, so the mixture's distribution function
    has a shoulder no single component's quantile describes.
    """
    return {
        "logits": torch.tensor([[[0.4, -0.3, 0.1]]], dtype=torch.float64),
        "means": torch.tensor(
            [[[[-2.0, 10.0], [0.5, 12.0], [3.0, 9.0]]]], dtype=torch.float64
        ),
        "df": torch.tensor([[[3.5, 8.0, 30.0]]], dtype=torch.float64),
        "lowrank_diag": torch.tensor(
            [[[[0.4, 1.0], [2.0, 0.2], [0.1, 3.0]]]], dtype=torch.float64
        ),
        "lowrank_U": torch.tensor(
            [[[[[0.5], [-0.3]], [[0.2], [0.4]], [[-0.6], [0.1]]]]],
            dtype=torch.float64,
        ),
    }


def _marginal_cdf(params: dict[str, torch.Tensor], column: int, value: float) -> float:
    """Evaluate one column's marginal mixture distribution function with SciPy."""
    weights = torch.softmax(params["logits"][0, 0], dim=-1)
    variance = params["lowrank_diag"][0, 0, :, column] + params["lowrank_U"][
        0, 0, :, column
    ].pow(2).sum(dim=-1)
    return float(
        sum(
            float(weights[k])
            * stats.t.cdf(
                value,
                float(params["df"][0, 0, k]),
                loc=float(params["means"][0, 0, k, column]),
                scale=math.sqrt(float(variance[k])),
            )
            for k in range(weights.shape[0])
        )
    )


def _dense(params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return the mixture with its scale matrices as dense Cholesky factors."""
    scale = torch.diag_embed(params["lowrank_diag"]) + params["lowrank_U"] @ params[
        "lowrank_U"
    ].transpose(-1, -2)
    return {
        "logits": params["logits"],
        "means": params["means"],
        "df": params["df"],
        "chol": torch.linalg.cholesky(scale),
    }


def test_mean_is_the_mixture_weighted_location_for_any_logit_shift() -> None:
    """Weights of one and three quarters put the mean 3/4 of the way to the second location."""
    params = {
        "logits": torch.tensor([[[0.0, math.log(3.0)]]], dtype=torch.float64),
        "means": torch.tensor([[[[-2.0, 10.0], [2.0, 6.0]]]], dtype=torch.float64),
        "df": torch.tensor([[[3.0, 9.0]]], dtype=torch.float64),
    }
    expected = torch.tensor([[[1.0, 7.0]]], dtype=torch.float64)

    torch.testing.assert_close(mixture_mean(params), expected)
    shifted = {**params, "logits": params["logits"] - 7.0}
    torch.testing.assert_close(mixture_mean(shifted), expected)


def test_mean_refuses_a_component_without_a_mean() -> None:
    params = _three_component_mixture()
    params["df"] = torch.tensor([[[3.5, 1.0, 30.0]]], dtype=torch.float64)

    with pytest.raises(ValueError, match="more than 1 degree of freedom"):
        mixture_mean(params)


def test_a_one_component_quantile_is_the_student_t_quantile() -> None:
    params = {
        "logits": torch.zeros(1, 1, 1, dtype=torch.float64),
        "means": torch.tensor([[[[4.0, -1.0]]]], dtype=torch.float64),
        "df": torch.tensor([[[4.5]]], dtype=torch.float64),
        "chol": torch.tensor([[[[[2.0, 0.0], [0.6, 0.8]]]]], dtype=torch.float64),
    }

    quantiles = mixture_quantiles(params, LEVELS)

    for column, (location, scale) in enumerate([(4.0, 2.0), (-1.0, 1.0)]):
        expected = stats.t.ppf(LEVELS, 4.5, loc=location, scale=scale)
        torch.testing.assert_close(
            quantiles[:, 0, 0, column],
            torch.as_tensor(expected, dtype=torch.float64),
            atol=1e-12,
            rtol=0.0,
        )


def test_a_mixture_quantile_is_the_root_of_the_marginal_distribution_function() -> None:
    params = _three_component_mixture()
    scales = (params["lowrank_diag"] + params["lowrank_U"].pow(2).sum(dim=-1)).sqrt()

    quantiles = mixture_quantiles(params, LEVELS)

    for column in range(2):
        tolerance = QUANTILE_TOLERANCE * float(scales[0, 0, :, column].min())
        for index, level in enumerate(LEVELS):
            root = optimize.brentq(
                lambda value, c=column, p=level: _marginal_cdf(params, c, value) - p,
                -100.0,
                100.0,
                xtol=1e-14,
            )
            assert abs(float(quantiles[index, 0, 0, column]) - root) <= tolerance


def test_both_scale_parameterizations_give_the_same_quantiles() -> None:
    params = _three_component_mixture()

    torch.testing.assert_close(
        mixture_quantiles(_dense(params), LEVELS), mixture_quantiles(params, LEVELS)
    )


def test_quantiles_do_not_depend_on_how_positions_are_split_over_threads() -> None:
    params = {
        name: value.expand(2, 5, *value.shape[2:]).contiguous()
        for name, value in _three_component_mixture().items()
    }
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        serial = mixture_quantiles(params, LEVELS)
        torch.set_num_threads(3)
        split = mixture_quantiles(params, LEVELS)
    finally:
        torch.set_num_threads(threads)

    assert split.shape == (len(LEVELS), 2, 5, 2)
    assert torch.equal(serial, split)


def test_quantiles_keep_the_parameters_device_and_dtype() -> None:
    params = {name: value.float() for name, value in _three_component_mixture().items()}

    quantiles = mixture_quantiles(params, [0.1, 0.9])

    assert quantiles.dtype == torch.float32
    assert quantiles.shape == (2, 1, 1, 2)


@pytest.mark.parametrize("levels", [[], [0.0], [1.0], [-0.2], [float("nan")]])
def test_quantiles_refuse_levels_outside_the_open_unit_interval(
    levels: list[float],
) -> None:
    with pytest.raises(ValueError, match="q must contain|must lie in"):
        mixture_quantiles(_three_component_mixture(), levels)


def test_quantiles_refuse_a_component_without_a_scale() -> None:
    params = _three_component_mixture()
    params["lowrank_diag"] = torch.zeros_like(params["lowrank_diag"])
    params["lowrank_U"] = torch.zeros_like(params["lowrank_U"])

    with pytest.raises(ValueError, match="finite and positive"):
        mixture_quantiles(params, [0.5])


def _data_unit_forward() -> tuple[StudentTHead, dict[str, torch.Tensor]]:
    """Run a seeded network on a history whose level and spread are far from one.

    Every component keeps more than two degrees of freedom, so the draws' mean
    has the standard error the parity check reads.
    """
    model = tiny_model(tiny_config(head_min_df=5.0))
    generator = torch.Generator().manual_seed(3)
    past = 5000.0 + 40.0 * torch.randn(1, 12, 3, generator=generator).cumsum(dim=1)
    with torch.no_grad():
        params = model(past, torch.arange(2))
    assert float(params[AUTO_STANDARDIZATION_SCALE_KEY].min()) > 10.0
    return model.head, params


def test_exact_summaries_match_the_sampler_in_data_units() -> None:
    head, params = _data_unit_forward()

    assert_exact_summaries_match_draws(
        head,
        params,
        q=LEVELS,
        draw_count=20_000,
        rng=torch.Generator().manual_seed(11),
    )


def test_exact_summaries_match_the_sampler_on_a_conditioned_mixture() -> None:
    """A conditioned mixture carries a dense Cholesky factor instead of a low rank."""
    head, params = _data_unit_forward()
    pinned = params["means"][:, :, 0, :].clone()
    mask = torch.tensor([True, False, False])
    conditioned = PredictorAdapter(head, params).condition(pinned, mask)

    assert "chol" in conditioned.params
    assert_exact_summaries_match_draws(
        head,
        conditioned.params,
        q=LEVELS,
        draw_count=20_000,
        rng=torch.Generator().manual_seed(12),
    )


def _rescaled(params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Apply the window's standardization to data-unit parameters a second time."""
    center = params[AUTO_STANDARDIZATION_CENTER_KEY].unsqueeze(2)
    scale = params[AUTO_STANDARDIZATION_SCALE_KEY].unsqueeze(2)
    return {
        **params,
        "means": center + scale * (params["means"] - center),
        "lowrank_diag": params["lowrank_diag"] * scale**2,
        "lowrank_U": params["lowrank_U"] * scale.unsqueeze(-1),
    }


class _MeanScaledTwiceHead(StudentTHead):
    """A head whose exact mean rescales the parameters it reads."""

    def mean(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        return super().mean(_rescaled(params))


class _QuantilesScaledTwiceHead(StudentTHead):
    """A head whose exact quantiles rescale the parameters they read."""

    def quantiles(self, params: dict[str, torch.Tensor], q: Any) -> torch.Tensor:
        return super().quantiles(_rescaled(params), q)


@pytest.mark.parametrize(
    ("faulty_head", "summary"),
    [(_MeanScaledTwiceHead, "mean"), (_QuantilesScaledTwiceHead, "quantile")],
)
def test_parity_fails_when_the_scale_is_applied_twice(
    faulty_head: type[StudentTHead], summary: str
) -> None:
    """Each summary's check catches the defect on its own."""
    head, params = _data_unit_forward()
    faulty = faulty_head(
        d_model=8,
        D=head.D,
        K=head.K,
        trunk_layers=1,
        trunk_hidden=8,
        covariance_rank=head.covariance_rank,
        min_df=head.min_df,
        min_scale=head.min_scale,
    )

    with pytest.raises(AssertionError, match=f"exact {summary} lies"):
        assert_exact_summaries_match_draws(
            faulty,
            params,
            q=LEVELS,
            draw_count=20_000,
            rng=torch.Generator().manual_seed(11),
        )


@pytest.mark.parametrize("return_mode", ["mean", "quantiles"])
@pytest.mark.parametrize("query_mode", ["forecast", "condition"])
def test_served_means_and_quantiles_are_the_same_for_every_seed(
    return_mode: str, query_mode: str
) -> None:
    """Unconditional and equality-conditioned answers draw nothing."""
    service = random_weight_service()
    payload = make_payload(return_mode=return_mode, quantiles=[0.1, 0.5, 0.9])
    if query_mode == "forecast":
        payload["query_mode"] = "forecast"
        del payload["conditions"]
    else:
        payload["requested_columns"] = ["driver", "target"]
    if return_mode == "mean":
        del payload["quantiles"]

    first = service.predict({**payload, "seed": 7, "n_samples": 16})
    second = service.predict({**payload, "seed": 8, "n_samples": 64})

    assert first["outputs"][return_mode] == second["outputs"][return_mode]


def test_interval_conditioned_quantiles_keep_their_draws() -> None:
    """An interval-conditioned answer rests on draws, so the seed moves it."""
    service = random_weight_service()
    payload = make_payload(
        return_mode="quantiles",
        quantiles=[0.1, 0.9],
        requested_columns=["driver", "target"],
        conditions=[
            {
                "column": "driver",
                "kind": "interval",
                "lower": 1.2,
                "upper": 1.6,
                "query_time_indices": [1],
            }
        ],
        n_samples=32,
    )

    first = service.predict({**payload, "seed": 7})
    second = service.predict({**payload, "seed": 8})

    assert first["outputs"]["quantiles"] != second["outputs"]["quantiles"]
