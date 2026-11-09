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

The network reads every column as ``(x - center) / scale`` and its head
models the standardized values; the forward maps the head's parameters back
to data units.

- **Center** — the column's last history value, so a zero standardized
  forecast repeats it.
- **Scale** — the population standard deviation of the column's history.

A column whose history cannot define a scale is refused, never clamped: fewer than two history values, or a spread at or below
``MIN_RELATIVE_SCALE_ULPS`` float32 ulps of the column's largest magnitude.
Below that spread the standardized history is rounding noise, and a clamped
scale would turn any future move of the column into an extreme multiple of
it: a forecast collapsed onto the last value, and a condition placed
implausibly far from the forecast.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor

MIN_RELATIVE_SCALE_ULPS: Final = 64
"""Smallest admissible spread, in float32 ulps of a column's largest magnitude."""

_STATISTICS_DTYPE: Final = torch.float32


def history_scale_degeneracy(history: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return which columns of a history define no scale, and why.

    Args:
        history: ``(..., rows, columns)`` history values; every cell counts.

    Returns:
        ``(degenerate, spread, magnitude)``, each ``(..., columns)``: the bool
        mask of columns with fewer than two rows or a spread at or below
        ``MIN_RELATIVE_SCALE_ULPS`` float32 ulps of their largest magnitude,
        then each column's population standard deviation and largest
        absolute value.

    Raises:
        ValueError: If ``history`` has fewer than two dimensions.
    """
    if history.ndim < 2:
        raise ValueError("history must have shape (..., rows, columns)")
    cells = history.to(dtype=_STATISTICS_DTYPE)
    spread = cells.std(dim=-2, unbiased=False)
    magnitude = cells.abs().amax(dim=-2)
    floor = MIN_RELATIVE_SCALE_ULPS * torch.finfo(_STATISTICS_DTYPE).eps * magnitude
    degenerate = ~(spread > floor) | (history.shape[-2] < 2)
    return degenerate, spread, magnitude


def scale_refusal_detail(spread: float, magnitude: float) -> str:
    """Return the reason a column is refused, for an error message."""
    return (
        f"has history spread {spread:.6g} at magnitude {magnitude:.6g}. JointFM 1 "
        "standardizes every column by its history and needs at least 2 history "
        f"rows and a spread wider than {MIN_RELATIVE_SCALE_ULPS} float32 ulps of "
        "the column's magnitude. Leave out a column that does not move within "
        "its history."
    )


def input_scale(past: Tensor) -> tuple[Tensor, Tensor]:
    """Return the ``(B, 1, D)`` center and scale of a ``(B, T, D)`` history.

    Args:
        past: ``(B, T, D)`` history in data units.

    Returns:
        ``(center, scale)`` in ``past``'s dtype: the last history value and the
        history's population standard deviation.

    Raises:
        ValueError: If a column's history defines no scale; the message names
            the batch element and column of the first such column.
    """
    degenerate, spread, magnitude = history_scale_degeneracy(past)
    if bool(degenerate.any().item()):
        batch_index, column_index = (
            int(index) for index in degenerate.nonzero(as_tuple=False)[0].tolist()
        )
        detail = scale_refusal_detail(
            float(spread[batch_index, column_index]),
            float(magnitude[batch_index, column_index]),
        )
        raise ValueError(
            f"column {column_index} of batch element {batch_index} {detail}"
        )
    center = past[:, -1:, :]
    scale = past.std(dim=1, keepdim=True, unbiased=False)
    return center, scale


__all__ = [
    "MIN_RELATIVE_SCALE_ULPS",
    "history_scale_degeneracy",
    "input_scale",
    "scale_refusal_detail",
]
