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

"""A Student-t head with a per-column tail map: its emitted law and every read-out."""

import math

import numpy as np
import pytest
import torch
from scipy import integrate, optimize

from jointfm_inference.model.jointfm_1.head import StudentTHead
from jointfm_inference.model.jointfm_1.loader import load_model_from_checkpoint
from jointfm_inference.model.jointfm_1.model import JointFM
from jointfm_inference.model.jointfm_1.marginals import (
    MAPPED_MEAN_TOLERANCE,
    QUANTILE_TOLERANCE,
    attach_column_anchors,
    column_anchors,
    component_marginal_scales,
)
from jointfm_inference.model.jointfm_1.tail_map import (
    ANCHOR_LOCATION_KEY,
    ANCHOR_SCALE_KEY,
    TAIL_LOWER_KEY,
    TAIL_MAP_KEYS,
    TAIL_UPPER_KEY,
)

from tests.support.models import checkpoint_mapping, tiny_config, tiny_model
from tests.support.service import mount_service
from tests.support.tail_map import (
    mapped_mixture,
    oracle_column_cdf,
    oracle_column_density,
    oracle_log_prob,
)

D_MODEL = 4
MIN_DF = 3.0
BOUNDS = (0.2, 1.4)
LEVELS = (0.01, 0.1, 0.5, 0.9, 0.99)
MAPPED_CONFIG = tiny_config(
    head_min_df=MIN_DF,
    head_tail_parameter_min=BOUNDS[0],
    head_tail_parameter_max=BOUNDS[1],
)


def _head(*, columns: int, components: int, mapped: bool) -> StudentTHead:
    """Return a head with fixed-seed weights, with or without a tail map."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return StudentTHead(
            d_model=D_MODEL,
            D=columns,
            K=components,
            trunk_layers=1,
            trunk_hidden=D_MODEL,
            covariance_rank=1,
            min_df=MIN_DF,
            min_scale=1e-4,
            tail_parameter_bounds=BOUNDS if mapped else None,
        )


def _mapped_head() -> StudentTHead:
    """Return a head with a map over the three columns of :func:`mapped_mixture`."""
    return _head(columns=3, components=2, mapped=True)


def test_a_zero_tail_readout_reproduces_the_head_without_a_map() -> None:
    """At the identity every read-out equals the unmapped head's, up to its tolerance."""
    plain = _head(columns=3, components=2, mapped=False)
    mapped = _head(columns=3, components=2, mapped=True)
    mapped.load_state_dict(plain.state_dict(), strict=False)
    assert mapped.column_tail is not None
    torch.nn.init.zeros_(mapped.column_tail.output.weight)
    generator = torch.Generator().manual_seed(1)
    context = torch.randn(1, 2, D_MODEL, generator=generator)
    tokens = torch.randn(1, 2, 3, D_MODEL, generator=generator)

    plain_params = plain(context, tokens)
    mapped_params = attach_column_anchors(mapped(context, tokens))
    values = torch.randn(1, 2, 3, generator=generator)

    assert torch.equal(mapped_params[TAIL_UPPER_KEY], torch.ones(1, 2, 3))
    assert torch.equal(mapped_params[TAIL_LOWER_KEY], torch.ones(1, 2, 3))
    assert torch.allclose(
        mapped.log_prob(mapped_params, values),
        plain.log_prob(plain_params, values),
        atol=1e-5,
    )
    assert torch.allclose(
        mapped.sample(mapped_params, 64, rng=torch.Generator().manual_seed(3)),
        plain.sample(plain_params, 64, rng=torch.Generator().manual_seed(3)),
        atol=1e-5,
    )
    assert torch.allclose(
        mapped.quantiles(mapped_params, LEVELS),
        plain.quantiles(plain_params, LEVELS),
        atol=1e-5,
    )
    anchor_scale = mapped_params[ANCHOR_SCALE_KEY]
    assert bool(
        (
            (mapped.mean(mapped_params) - plain.mean(plain_params)).abs()
            <= 10 * MAPPED_MEAN_TOLERANCE * anchor_scale + 1e-6
        ).all()
    )


@pytest.mark.parametrize(
    "value",
    [[0.0, 0.0, 0.0], [-7.5, 4.0, 0.3], [6.0, -9.0, -2.5], [0.9, 1.1, 12.0]],
)
def test_the_mapped_density_matches_the_oracle(value: list[float]) -> None:
    """The log-density divides the latent mixture's by the map's derivative."""
    params = mapped_mixture()
    values = torch.tensor([[value]], dtype=torch.float64)

    log_prob = float(_mapped_head().log_prob(params, values))

    assert log_prob == pytest.approx(oracle_log_prob(params, value), abs=1e-10)


def test_the_mapped_density_integrates_to_one_per_column() -> None:
    """A column's marginal density, the oracle's, integrates to one, so the oracle is a law."""
    params = mapped_mixture()
    for column in range(3):
        anchor = float(params[ANCHOR_LOCATION_KEY][0, 0, column])
        total = sum(
            integrate.quad(
                lambda y: oracle_column_density(params, column, y),
                low,
                high,
                limit=400,
            )[0]
            for low, high in ((-math.inf, anchor), (anchor, math.inf))
        )
        assert total == pytest.approx(1.0, abs=1e-8)


def test_the_mapped_quantiles_are_the_roots_of_the_oracle_distribution_function() -> (
    None
):
    """Each quantile is within the bisection tolerance of the oracle's root."""
    params = mapped_mixture()
    quantiles = _mapped_head().quantiles(params, LEVELS)
    tolerance = 2 * QUANTILE_TOLERANCE * float(component_marginal_scales(params).min())

    for level_index, level in enumerate(LEVELS):
        for column in range(3):
            root = optimize.brentq(
                lambda y: oracle_column_cdf(params, column, y) - level,
                -1e3,
                1e3,
                xtol=1e-13,
            )
            assert float(quantiles[level_index, 0, 0, column]) == pytest.approx(
                root, abs=tolerance
            )


def test_the_mapped_mean_matches_an_adaptive_integral() -> None:
    """The quadrature mean equals the integral of ``y`` against the oracle density."""
    params = mapped_mixture()
    mean = _mapped_head().mean(params)

    for column in range(3):
        anchor = float(params[ANCHOR_LOCATION_KEY][0, 0, column])
        reference = sum(
            integrate.quad(
                lambda y: y * oracle_column_density(params, column, y),
                low,
                high,
                limit=400,
                epsabs=1e-12,
            )[0]
            for low, high in ((-math.inf, anchor), (anchor, math.inf))
        )
        scale = float(params[ANCHOR_SCALE_KEY][0, 0, column])
        assert float(mean[0, 0, column]) == pytest.approx(
            reference, abs=MAPPED_MEAN_TOLERANCE * scale + 1e-9
        )


def test_mapped_draws_follow_the_exact_quantiles() -> None:
    """The empirical distribution of many draws sits at the exact quantiles' levels."""
    params = mapped_mixture()
    head = _mapped_head()
    draw_count = 40_000
    draws = head.sample(params, draw_count, rng=torch.Generator().manual_seed(5))
    quantiles = head.quantiles(params, LEVELS)

    for level_index, level in enumerate(LEVELS):
        share_below = (draws[:, 0, :] <= quantiles[level_index, 0, 0]).double().mean(0)
        tolerance = 4.0 * math.sqrt(level * (1.0 - level) / draw_count)
        assert torch.allclose(
            share_below, torch.full_like(share_below, level), atol=tolerance
        )


def test_a_head_refuses_parameters_whose_map_does_not_match() -> None:
    """A mapped head never scores the latent law, and an unmapped one never a mapped law."""
    params = mapped_mixture()
    without = {
        name: value for name, value in params.items() if name not in TAIL_MAP_KEYS
    }
    values = torch.zeros(1, 1, 3, dtype=torch.float64)

    with pytest.raises(ValueError, match="lack one"):
        _mapped_head().log_prob(without, values)
    with pytest.raises(ValueError, match="carry one"):
        _head(columns=3, components=2, mapped=False).sample(params, 4)


def _mapped_model() -> JointFM:
    """Return a tiny network with a map whose readout is not the identity."""
    model = tiny_model(MAPPED_CONFIG)
    assert model.head.column_tail is not None
    with torch.no_grad():
        model.head.column_tail.output.weight.normal_(0.0, 2.0)
    return model


def test_the_network_emits_the_map_over_its_active_columns() -> None:
    """The forward slices the tail parameters to the request and anchors the data-unit mixture."""
    model = _mapped_model()
    history = torch.randn(1, 12, 3, generator=torch.Generator().manual_seed(2)) * 5.0

    with torch.no_grad():
        output = model(history, torch.arange(2))

    location, scale = column_anchors(output)
    for key in TAIL_MAP_KEYS:
        assert tuple(output[key].shape) == (1, 2, 3)
    assert torch.allclose(output[ANCHOR_LOCATION_KEY], location)
    assert torch.allclose(output[ANCHOR_SCALE_KEY], scale)
    tails = torch.cat([output[TAIL_UPPER_KEY], output[TAIL_LOWER_KEY]])
    assert bool(((tails >= BOUNDS[0]) & (tails <= BOUNDS[1])).all())
    assert bool((tails != 1.0).any())


def test_a_checkpoint_with_a_map_round_trips_through_the_strict_loader() -> None:
    """The tail readout's weight loads strictly and reproduces the forward."""
    model = _mapped_model()
    history = torch.randn(1, 12, 3, generator=torch.Generator().manual_seed(2))

    reloaded, cfg = load_model_from_checkpoint(checkpoint_mapping(model))

    assert cfg.has_tail_map
    with torch.no_grad():
        expected = model(history, torch.arange(2))
        actual = reloaded(history, torch.arange(2))
    for key in (*TAIL_MAP_KEYS, "means", "lowrank_diag"):
        assert torch.equal(actual[key], expected[key])


def test_health_reports_the_tail_map_of_the_mounted_head() -> None:
    """``/healthz`` names the map, and a head without one reports none."""
    assert mount_service(_mapped_model()).health_payload()["tail_map"] == (
        "sinh_arcsinh_column"
    )
    assert mount_service(tiny_model()).health_payload()["tail_map"] == "none"


def test_a_mapped_column_with_a_heavy_upper_side_is_skewed_toward_it() -> None:
    """A heavy upper and light lower side put the upper tail farther from the median."""
    params = mapped_mixture()
    quantiles = _mapped_head().quantiles(params, (0.01, 0.5, 0.99)).numpy()[:, 0, 0]

    upper_reach = quantiles[2] - quantiles[1]
    lower_reach = quantiles[1] - quantiles[0]
    # Column 0 has delta+ 1.35 and delta- 0.6; column 1 the reverse.
    assert upper_reach[0] > lower_reach[0]
    assert lower_reach[1] > upper_reach[1]
    assert np.all(np.isfinite(quantiles))
