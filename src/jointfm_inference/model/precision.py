"""Model compute precision of inference.

A model precision names the dtype a model forward runs its matmuls and
attention in through ``torch.autocast`` (automatic mixed precision: eligible
operations run in the reduced dtype while numerically sensitive ones stay in
float32). The parameters themselves stay float32 either way. The session's
``precision`` argument selects it (``JOINT_PRECISION`` in the HTTP service).
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch
from torch import Tensor

MODEL_PRECISIONS = ("fp32", "bf16", "fp16")

_REDUCED_FLOAT_DTYPES = (torch.bfloat16, torch.float16)


def resolve_autocast_dtype(
    precision: str,
    device: torch.device,
    *,
    setting: str,
) -> torch.dtype | None:
    """Resolve a model precision into the autocast dtype for ``device``.

    Args:
        precision: One of :data:`MODEL_PRECISIONS`, already normalized.
        device: Device the model forward runs on.
        setting: Name of the setting that supplied ``precision``, quoted in
            error messages so the caller knows which knob to change.

    Returns:
        ``None`` for ``fp32`` (autocast disabled), otherwise the reduced dtype.

    Raises:
        ValueError: For an unknown precision, or a pair the device cannot
            serve — reduced precision outside cpu/cuda, bf16 on a CUDA device
            without bf16 support, and fp16 anywhere but CUDA (CPU fp16 kernel
            coverage is too sparse). Nothing degrades to fp32 silently.
    """
    if precision not in MODEL_PRECISIONS:
        raise ValueError(
            f"Expected {setting} to be one of {MODEL_PRECISIONS}, got {precision!r}."
        )
    if precision == "fp32":
        return None
    if device.type not in ("cpu", "cuda"):
        raise ValueError(
            f"{setting} supports {precision!r} only on cpu or cuda devices, got "
            f"device type {device.type!r}; select 'fp32' on this device."
        )
    if precision == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError(
                f"{setting} is 'bf16' but the CUDA device does not support bf16."
            )
        return torch.bfloat16
    if device.type != "cuda":
        raise ValueError(
            f"{setting} supports 'fp16' only on CUDA devices, got device type "
            f"{device.type!r}."
        )
    return torch.float16


def autocast_context(
    device_type: str,
    autocast_dtype: torch.dtype | None,
) -> AbstractContextManager[Any]:
    """Return the autocast context for one model forward (no-op for fp32)."""
    if autocast_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device_type, dtype=autocast_dtype)


def upcast_reduced_precision(tensors: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Return ``tensors`` with every bf16/fp16 tensor cast to float32.

    Autocast leaves a forward's outputs in the reduced dtype; distribution
    read-outs (sampling, quantiles, log-densities, conditioning) must run in
    float32 whatever precision produced their parameters. Tensors of any
    other dtype pass through unchanged.
    """
    return {
        name: (
            value.to(torch.float32) if value.dtype in _REDUCED_FLOAT_DTYPES else value
        )
        for name, value in tensors.items()
    }


__all__ = [
    "MODEL_PRECISIONS",
    "autocast_context",
    "resolve_autocast_dtype",
    "upcast_reduced_precision",
]
