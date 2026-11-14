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

"""The per-column tail map's functions, its parameter mapping, and its configuration."""

import math
from dataclasses import asdict

import numpy as np
import pytest
import torch

from jointfm_inference.model.jointfm_1.config import JointFMConfig
from jointfm_inference.model.jointfm_1.tail_map import (
    TAIL_MAP_KEYS,
    TAIL_UPPER_KEY,
    ColumnTailMap,
    inverse_standardized,
    log_derivative_at_residual,
    map_standardized,
    tail_parameter,
)

from tests.support.models import tiny_config
from tests.support.tail_map import (
    mapped_mixture,
    oracle_data,
    oracle_latent,
    oracle_log_derivative,
)

TAILS = torch.tensor([0.2, 0.55, 1.0, 1.4, 1.9], dtype=torch.float64)
COORDINATES = torch.tensor([-1e6, -40.0, -1.3, -1e-9, 0.0, 0.7, 25.0, 3e5])


def test_the_map_is_the_identity_at_one_and_its_inverse_round_trips() -> None:
    """``g(1, u) = u``, and ``g^-1(g(u)) = u`` for every tail parameter."""
    latent = COORDINATES.double().unsqueeze(-1)

    assert torch.allclose(map_standardized(latent, torch.ones(1).double()), latent)
    mapped = map_standardized(latent, TAILS)
    assert torch.allclose(
        inverse_standardized(mapped, TAILS), latent.expand_as(mapped), rtol=1e-12
    )


def test_the_log_derivative_matches_the_derivative_of_the_map() -> None:
    """``log g'`` read from the data side equals the log of a central difference."""
    latent = torch.linspace(-30.0, 30.0, 121, dtype=torch.float64).unsqueeze(-1)
    step = 1e-6 * (1.0 + latent.abs())
    difference = (
        map_standardized(latent + step, TAILS) - map_standardized(latent - step, TAILS)
    ) / (2.0 * step)
    log_derivative = log_derivative_at_residual(map_standardized(latent, TAILS), TAILS)

    assert torch.allclose(log_derivative, difference.log(), atol=1e-7)


def test_a_zero_raw_value_is_the_identity_and_every_value_stays_in_bounds() -> None:
    """The squash maps zero to one and every finite value into the bounds."""
    raw = torch.tensor([-1e6, -3.0, -1e-3, 0.0, 1e-3, 3.0, 1e6])

    tail = tail_parameter(raw, minimum=0.3, maximum=1.4)

    assert float(tail[3]) == 1.0
    assert bool(((tail >= 0.3) & (tail <= 1.4)).all())
    assert bool((tail.diff() >= 0).all())


def test_a_non_finite_raw_value_is_refused() -> None:
    """A diverged readout raises instead of saturating silently."""
    with pytest.raises(RuntimeError, match="non-finite"):
        tail_parameter(torch.tensor([0.0, math.nan]), minimum=0.3, maximum=1.4)


def test_the_map_moves_data_and_latent_values_like_the_definition() -> None:
    """Data values map to the oracle's latent images and back, each side by its own parameter."""
    params = mapped_mixture()
    tail_map = ColumnTailMap.from_params(params)
    assert tail_map is not None
    values = torch.tensor([[[-6.0, 0.3, 9.0], [2.0, -4.0, -0.2]]], dtype=torch.float64)

    latent = tail_map.to_latent(values)

    assert np.allclose(latent.numpy()[0], oracle_latent(params, values.numpy()[0]))
    assert torch.allclose(tail_map.to_data(latent), values, rtol=1e-12)
    assert np.allclose(
        tail_map.log_derivative(values).numpy()[0],
        oracle_log_derivative(params, values.numpy()[0]),
    )
    assert np.allclose(
        tail_map.to_data(values).numpy()[0], oracle_data(params, values.numpy()[0])
    )


def test_a_mapping_without_the_map_carries_none_and_a_partial_one_is_refused() -> None:
    """All four map keys travel together."""
    params = mapped_mixture()
    without = {
        name: value for name, value in params.items() if name not in TAIL_MAP_KEYS
    }
    partial = {name: value for name, value in params.items() if name != TAIL_UPPER_KEY}

    assert ColumnTailMap.from_params(without) is None
    with pytest.raises(ValueError, match="missing"):
        ColumnTailMap.from_params(partial)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("upper", 2.5, "tail parameters must lie"),
        ("lower", 0.1, "tail parameters must lie"),
        ("anchor_scale", 0.0, "positive"),
        ("anchor_location", math.inf, "finite"),
    ],
)
def test_a_map_outside_its_envelope_is_refused(
    field: str, value: float, message: str
) -> None:
    """Out-of-envelope tail parameters and degenerate anchors raise."""
    parts = {
        "upper": torch.ones(2),
        "lower": torch.ones(2),
        "anchor_location": torch.zeros(2),
        "anchor_scale": torch.ones(2),
    }
    parts[field] = torch.tensor([1.0, value])
    with pytest.raises(ValueError, match=message):
        ColumnTailMap(**parts)


def test_a_residual_beyond_the_envelope_is_refused() -> None:
    """A standardized residual past the verified range raises instead of overflowing."""
    tail_map = ColumnTailMap(
        upper=torch.ones(1),
        lower=torch.ones(1),
        anchor_location=torch.zeros(1),
        anchor_scale=torch.ones(1),
    )
    with pytest.raises(ValueError, match="standardized residuals"):
        tail_map.to_latent(torch.tensor([1e31], dtype=torch.float64))


def test_a_config_without_bounds_has_no_map_and_one_with_both_has_one() -> None:
    """The two bounds switch the map on together; a cfg without them has none."""
    assert not tiny_config().has_tail_map
    assert tiny_config(
        head_min_df=3.0, head_tail_parameter_min=0.2, head_tail_parameter_max=1.4
    ).has_tail_map


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"head_tail_parameter_min": 0.3}, "set together"),
        (
            {"head_tail_parameter_min": 0.1, "head_tail_parameter_max": 1.4},
            "must satisfy",
        ),
        (
            {"head_tail_parameter_min": 0.3, "head_tail_parameter_max": 0.9},
            "must satisfy",
        ),
        (
            {"head_tail_parameter_min": 0.3, "head_tail_parameter_max": 1.6},
            "head_min_df / 2",
        ),
        (
            {
                "head_tail_parameter_min": 0.3,
                "head_tail_parameter_max": 1.4,
                "sign_symmetric_prediction": True,
            },
            "sign_symmetric_prediction",
        ),
    ],
)
def test_bounds_the_map_cannot_serve_are_refused(
    overrides: dict[str, object], message: str
) -> None:
    """Half-set, out-of-envelope, too-heavy, and sign-symmetric bounds fail at load."""
    with pytest.raises(ValueError, match=message):
        tiny_config(head_min_df=3.0, **overrides)


def test_a_cfg_mapping_without_the_bound_keys_loads_without_a_map() -> None:
    """A checkpoint written before the map existed keeps loading as today's head."""
    raw = asdict(tiny_config())
    raw.pop("head_tail_parameter_min")
    raw.pop("head_tail_parameter_max")

    assert not JointFMConfig.from_mapping(raw).has_tail_map
