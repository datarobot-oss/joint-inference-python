"""Helpers that resolve sampling configuration and RNG state per request."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from http import HTTPStatus
from typing import Any, TYPE_CHECKING

import torch

from jointfm_inference.data.adapters import ColumnSpec

from .errors import ServiceError

if TYPE_CHECKING:
    from .config import InferenceServiceConfig


def _resolve_sample_count(
    *,
    request_payload: Mapping[str, Any],
    config: "InferenceServiceConfig",
) -> int:
    """Resolve the effective sample count for the requested response mode.

    All return modes share one resolution rule: honor the client's
    ``n_samples`` when present, otherwise fall back to the operator-configured
    ``max_sample_count`` so the deployment uses the GPU budget it was sized
    for. Clients that explicitly want a faster, noisier response pass a
    smaller ``n_samples``; clients that want the deployment's best precision
    simply omit the field.
    """
    raw_n_samples = request_payload.get("n_samples", config.max_sample_count)
    if isinstance(raw_n_samples, bool) or not isinstance(raw_n_samples, int):
        raise ServiceError(
            "VALIDATION_ERROR",
            "n_samples must be an integer",
            status=HTTPStatus.BAD_REQUEST,
            field="n_samples",
        )
    if raw_n_samples <= 0:
        raise ServiceError(
            "VALIDATION_ERROR",
            "n_samples must be positive",
            status=HTTPStatus.BAD_REQUEST,
            field="n_samples",
        )
    if raw_n_samples > config.max_sample_count:
        raise ServiceError(
            "INPUT_SIZE_EXCEEDED",
            (
                "n_samples exceeds the configured container cap: "
                f"requested {raw_n_samples}, max {config.max_sample_count}"
            ),
            status=HTTPStatus.BAD_REQUEST,
            field="n_samples",
        )
    return raw_n_samples


def _sampling_bound_tensors(
    column_specs: Sequence[ColumnSpec],
    *,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Build per-column sample rejection bounds from request column metadata."""
    has_lower_bound = any(column.lower_bound is not None for column in column_specs)
    has_upper_bound = any(column.upper_bound is not None for column in column_specs)
    lower_bound_tensor = None
    upper_bound_tensor = None

    if has_lower_bound:
        lower_bound_tensor = torch.tensor(
            [
                float(column.lower_bound)
                if column.lower_bound is not None
                else float("-inf")
                for column in column_specs
            ],
            device=device,
            dtype=torch.float32,
        )
    if has_upper_bound:
        upper_bound_tensor = torch.tensor(
            [
                float(column.upper_bound)
                if column.upper_bound is not None
                else float("inf")
                for column in column_specs
            ],
            device=device,
            dtype=torch.float32,
        )

    return lower_bound_tensor, upper_bound_tensor


def _torch_generator(seed: int | None, device: torch.device) -> torch.Generator | None:
    """Create a seeded torch generator for the requested device when needed."""
    if seed is None:
        return None
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)
    return generator
