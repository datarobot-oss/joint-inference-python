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

"""How a JointFM 1 network standardizes each column before reading it.

A column whose history defines no scale is refused instead of clamped,
because a clamped scale collapses its forecast onto the last value and places
any condition on it implausibly far from the forecast.
"""

import pytest
import torch

from jointfm_inference.model.jointfm_1.input_scale import (
    MIN_RELATIVE_SCALE_ULPS,
    history_scale_degeneracy,
    input_scale,
)
from tests.support.models import tiny_model

_ROWS, _HORIZONS, _COLUMNS = 12, 3, 3
_FLAT = 1


def _past() -> torch.Tensor:
    """Return a ``(1, T, D)`` random-walk history."""
    generator = torch.Generator().manual_seed(1)
    return torch.randn(1, _ROWS, _COLUMNS, generator=generator).cumsum(dim=1)


@pytest.mark.parametrize(
    ("values", "degenerate"),
    [
        (torch.zeros(_ROWS), True),
        (torch.full((_ROWS,), 7.0), True),
        (1000.0 + 1e-6 * (torch.arange(_ROWS) % 2), True),
        (1e-9 * torch.arange(_ROWS, dtype=torch.float32), False),
        (torch.cat([torch.zeros(_ROWS - 1), torch.ones(1)]), False),
    ],
    ids=["zero", "constant", "rounding", "tiny-unit", "one-move"],
)
def test_rule_refuses_flat_and_rounding_level_histories(
    values: torch.Tensor, degenerate: bool
) -> None:
    """Only a spread above the relative floor defines a scale, in any unit."""
    mask, spread, magnitude = history_scale_degeneracy(values.reshape(-1, 1))

    assert mask.tolist() == [degenerate]
    torch.testing.assert_close(spread, values.std(unbiased=False).reshape(1))
    torch.testing.assert_close(magnitude, values.abs().amax().reshape(1))


def test_rule_floor_is_relative_to_the_magnitude() -> None:
    floor = MIN_RELATIVE_SCALE_ULPS * torch.finfo(torch.float32).eps
    history = torch.tensor([[1.0 - floor], [1.0 + floor]])

    assert history_scale_degeneracy(history)[0].tolist() == [True]
    assert history_scale_degeneracy(history * 1e6)[0].tolist() == [True]


def test_rule_needs_two_rows() -> None:
    assert history_scale_degeneracy(torch.tensor([[1.0, 5.0]]))[0].tolist() == [
        True,
        True,
    ]


@pytest.mark.parametrize("flat_value", [0.0, 5.0])
def test_network_refuses_a_column_without_history_scale(flat_value: float) -> None:
    past = _past()
    past[:, :, _FLAT] = flat_value

    with pytest.raises(ValueError, match=rf"column {_FLAT} of batch element 0 "):
        tiny_model()(past, torch.arange(_HORIZONS))


def test_columns_take_the_last_value_and_history_spread() -> None:
    past = _past()

    center, scale = input_scale(past)

    torch.testing.assert_close(center, past[:, -1:, :])
    torch.testing.assert_close(scale, past.std(dim=1, keepdim=True, unbiased=False))
