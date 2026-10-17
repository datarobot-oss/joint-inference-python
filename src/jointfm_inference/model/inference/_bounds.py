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

"""Sample rejection bound helpers used by ``PredictorAdapter.sample``.

Normalizes user-supplied ``min_value``/``max_value`` arguments to the sample
device/dtype, computes per-sample violation masks, and produces compact error
fragments for rejection-sampling failures.
"""

from __future__ import annotations

import torch

_SampleBound = float | int | torch.Tensor


def _normalize_sample_bound(
    bound: _SampleBound | None,
    *,
    samples: torch.Tensor,
    field: str,
) -> torch.Tensor | None:
    """Normalize one sample rejection bound to the sample device and dtype."""
    if bound is None:
        return None
    if isinstance(bound, bool):
        raise TypeError(f"{field} must be numeric or a tensor")
    if isinstance(bound, torch.Tensor):
        if bound.numel() == 0:
            raise ValueError(f"{field} must not be empty")
        normalized = bound.to(device=samples.device, dtype=samples.dtype)
    elif isinstance(bound, (float, int)):
        normalized = torch.tensor(
            float(bound),
            device=samples.device,
            dtype=samples.dtype,
        )
    else:
        raise TypeError(f"{field} must be numeric or a tensor")

    if bool(torch.isnan(normalized).any().item()):
        raise ValueError(f"{field} must not contain NaNs")
    try:
        torch.broadcast_shapes(samples.shape, normalized.shape)
    except RuntimeError as error:
        raise ValueError(
            f"{field} must broadcast to sample shape {tuple(samples.shape)}"
        ) from error
    return normalized


def _sample_constraint_violations(
    samples: torch.Tensor,
    *,
    min_value: torch.Tensor | None,
    max_value: torch.Tensor | None,
) -> torch.Tensor:
    """Return the per-sample/horizon rejection mask for NaN and bounds failures."""
    is_bad = torch.isnan(samples).any(dim=-1)
    if min_value is not None:
        is_bad = is_bad | (samples < min_value).any(dim=-1)
    if max_value is not None:
        is_bad = is_bad | (samples > max_value).any(dim=-1)
    return is_bad


def _bound_error_fragment(field: str, bound: torch.Tensor) -> str:
    """Return a compact error fragment for one violated sample bound."""
    if bound.numel() == 1:
        return f" {field}={float(bound.item())} violation."
    return f" {field} violation."
