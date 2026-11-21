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
device/dtype, computes per-sample violation masks, and builds the error for
draws that redraws could not bring inside the bounds.
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


class UnmetSampleBoundError(RuntimeError):
    """Draws still fall outside a sample bound after every redraw.

    The distribution puts too little probability inside the bound for redraws
    to fill the requested draws, and drawing again does not change that, so
    the bound is what has to give. A caller that forwards the bound from a
    request reads which one from the attributes.

    Attributes:
        argument: ``"min_value"`` or ``"max_value"``, the argument the bound
            was passed as.
        column: The bound's column, an index on the last axis of the draws.
        bound: The bound's value in that column.
        failing: How many of the column's draws still miss the bound.
        total: How many draws the column has.
        redraws: The redraw rounds taken before giving up.
    """

    def __init__(
        self,
        message: str,
        *,
        argument: str,
        column: int,
        bound: float,
        failing: int,
        total: int,
        redraws: int,
    ) -> None:
        """Record the message and the bound the draws still miss."""
        super().__init__(message)
        self.argument = argument
        self.column = column
        self.bound = bound
        self.failing = failing
        self.total = total
        self.redraws = redraws


def _redraw_failure(
    message: str,
    samples: torch.Tensor,
    *,
    min_value: torch.Tensor | None,
    max_value: torch.Tensor | None,
    redraws: int,
) -> RuntimeError:
    """Return the error for draws that still break a constraint after the redraws.

    ``samples`` are the ``(B, S, H, D)`` draws left after the last redraw.
    Every cell still below ``min_value`` or above ``max_value`` counts against
    its column's bound on that side. A draw is replaced across all its columns
    at once, so a merely tight bound beside an unmeetable one also fails in
    some cells of the draws the unmeetable one keeps rejecting; the bound with
    the most failing cells is the one the distribution puts too little
    probability inside, and an :class:`UnmetSampleBoundError` names it. When
    no cell breaks a bound, only NaN draws remain, which no bound explains,
    and the error is a plain ``RuntimeError``.
    """
    sides: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    if min_value is not None:
        lower = torch.broadcast_to(min_value, samples.shape)
        sides.append(("min_value", lower, samples < lower))
    if max_value is not None:
        upper = torch.broadcast_to(max_value, samples.shape)
        sides.append(("max_value", upper, samples > upper))
    if not sides:
        return RuntimeError(message)
    counts = torch.stack(
        [violated.flatten(end_dim=-2).sum(dim=0) for _, _, violated in sides]
    )
    side, column = divmod(int(counts.argmax().item()), samples.shape[-1])
    failing = int(counts[side, column].item())
    if failing == 0:
        return RuntimeError(message)
    argument, bound, violated = sides[side]
    column_violated = violated[..., column]
    return UnmetSampleBoundError(
        message,
        argument=argument,
        column=column,
        bound=float(bound[..., column][column_violated][0].item()),
        failing=failing,
        total=column_violated.numel(),
        redraws=redraws,
    )
