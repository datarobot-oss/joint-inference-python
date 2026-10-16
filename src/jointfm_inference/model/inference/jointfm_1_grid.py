"""Request-time mapping onto the discrete time grid of JointFM 1 models.

A JointFM 1 model reads integer time positions bounded by the checkpoint's
``max_time_ctx`` and ``max_time_fut`` embedding capacities. These helpers map
the ordered, real-valued times of a canonical request onto that grid, keeping
gaps, and refuse requests whose times the grid cannot represent.
"""

from __future__ import annotations

import torch

from jointfm_inference.data.time_index import positions_within_relative_step_tolerance
from jointfm_inference.data.types import CanonicalSample

from .contract import InferenceRequestError
from .time_index import TimeIndexSupport


def grid_model_inputs(
    query_sample: CanonicalSample,
    *,
    time_index_support: TimeIndexSupport,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map one canonical request onto the JointFM 1 discrete time grid.

    Returns ``(past, past_time_indices, horizons)`` on ``device``: the observed
    rows' values as a ``(1, T, D)`` batch, their integer grid positions, and
    each query row's horizon index after the last history row. Gaps in
    the ordered request times survive the mapping losslessly.

    Raises:
        InferenceRequestError: when the sample has no history row, its times
            do not lie on one regular grid, or a grid position exceeds the
            checkpoint's time capacity.
    """
    query_rows = query_sample.query_row_mask
    grid_positions = _grid_time_positions(query_sample)
    past_time_indices = grid_positions[~query_rows]
    future_time_indices = grid_positions[query_rows]
    if past_time_indices.numel() <= 0:
        raise InferenceRequestError(
            "history_rows must provide at least one observed row",
            field="history_rows",
        )

    last_observed_index = int(past_time_indices[-1].item())
    # Query rows follow the history and times strictly increase, so every
    # horizon index is non-negative.
    horizons = future_time_indices - last_observed_index - 1

    _validate_grid_index_bound(
        indices=past_time_indices,
        max_size=time_index_support.max_time_ctx,
        field="history_rows",
        label="observed time indices",
    )
    _validate_grid_index_bound(
        indices=horizons,
        max_size=time_index_support.max_time_fut,
        field="query_times",
        label="forecast horizon indices",
    )

    past = query_sample.values[~query_rows].unsqueeze(0).to(device=device)
    return (
        past,
        past_time_indices.to(device=device, dtype=torch.long),
        horizons.to(device=device, dtype=torch.long),
    )


def _grid_time_positions(query_sample: CanonicalSample) -> torch.Tensor:
    """Convert ordered time positions into lossless grid indices."""
    time_positions = query_sample.time_positions.detach().to(dtype=torch.float64).cpu()
    if query_sample.time_index_mode == "ordinal":
        return _ordinal_positions_to_indices(time_positions)
    return _regular_grid_positions_to_indices(
        time_positions, mode_name=query_sample.time_index_mode
    )


def _ordinal_positions_to_indices(
    time_positions: torch.Tensor,
) -> torch.Tensor:
    """Convert ordinal time positions into zero-based grid indices without gap loss."""
    rounded = torch.round(time_positions)
    if not positions_within_relative_step_tolerance(
        time_positions,
        rounded,
        step_scale=1.0,
    ):
        raise InferenceRequestError(
            "ordinal history_rows and query_times must use integer step values",
            field="query_times",
        )
    return rounded.to(dtype=torch.long) - int(rounded[0].item())


def _regular_grid_positions_to_indices(
    time_positions: torch.Tensor,
    *,
    mode_name: str,
) -> torch.Tensor:
    """Convert ordered continuous or datetime positions onto one regular grid."""
    if time_positions.numel() <= 1:
        return torch.zeros(int(time_positions.numel()), dtype=torch.long)

    deltas = time_positions[1:] - time_positions[:-1]
    step = float(torch.min(deltas).item())
    if step <= 0.0:
        raise InferenceRequestError(
            f"{mode_name} time positions must be strictly increasing",
            field="query_times",
        )

    normalized = (time_positions - time_positions[0]) / step
    rounded = torch.round(normalized)
    reconstructed = time_positions[0] + rounded.to(dtype=time_positions.dtype) * step
    if not positions_within_relative_step_tolerance(
        time_positions,
        reconstructed,
        step_scale=abs(step),
    ):
        raise InferenceRequestError(
            (
                f"{mode_name} history_rows and query_times must lie on one regular "
                "step grid"
            ),
            field="query_times",
        )
    return rounded.to(dtype=torch.long)


def _validate_grid_index_bound(
    *,
    indices: torch.Tensor,
    max_size: int,
    field: str,
    label: str,
) -> None:
    """Fail fast when grid indices exceed the model's time capacity."""
    max_index = int(indices.max().item())
    if max_index >= max_size:
        raise InferenceRequestError(
            (
                f"The model cannot represent {label}: max index {max_index} "
                f"exceeds capacity {max_size - 1}"
            ),
            field=field,
        )
