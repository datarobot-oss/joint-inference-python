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

"""Tensor-to-JSON serialization helpers for inference responses.

Forecast arrays leave the device as contiguous float32 NumPy arrays and are
written to JSON by :func:`encode_response` with orjson, which formats a NumPy
array directly instead of first expanding it into nested Python lists. That
expansion is what dominates a large sample response: a Python float plus its
list slot costs about ten times the four bytes of the float32 it holds, and
the standard-library encoder then walks every one of those objects. The wire
format is unchanged — the same JSON document with the same nesting; orjson
writes each float32 as the shortest decimal that reads back to the same value.

orjson writes NaN and both infinities as ``null``, which a client would read as
a missing value. Values are therefore checked before encoding, so a
non-finite value fails here instead of reaching a client.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import orjson
import torch

RESPONSE_ENCODING_OPTIONS = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SORT_KEYS


def encode_response(payload: Mapping[str, Any]) -> bytes:
    """Encode one response payload, NumPy arrays included, as UTF-8 JSON bytes.

    Keys are sorted so equal payloads encode to equal bytes. Raises
    ``orjson.JSONEncodeError`` for values JSON cannot carry, such as a
    non-contiguous array or a non-string key.
    """
    return orjson.dumps(payload, option=RESPONSE_ENCODING_OPTIONS)


def _float32_array(tensor: torch.Tensor) -> np.ndarray:
    """Return a contiguous host float32 copy of one forecast tensor.

    The service hands float32 head outputs to the response; any other dtype
    means an upstream contract changed, so it is refused rather than cast.
    Raises ``RuntimeError`` for a non-finite value.
    """
    if tensor.dtype != torch.float32:
        raise RuntimeError(
            f"forecast tensors must be float32 before serialization; got {tensor.dtype}"
        )
    array = tensor.detach().contiguous().cpu().numpy()
    if not np.isfinite(array).all():
        raise RuntimeError("forecast values must be finite")
    return array


def _slice_forecast_tensor(
    tensor: torch.Tensor, requested_indices: Sequence[int]
) -> np.ndarray:
    """Extract requested forecast columns from a mean prediction tensor."""
    return _float32_array(tensor[0, :, requested_indices])


def _slice_sample_tensor(
    tensor: torch.Tensor, requested_indices: Sequence[int]
) -> np.ndarray:
    """Extract requested forecast columns from a sampled prediction tensor."""
    return _float32_array(tensor[0, :, :, requested_indices])


def _slice_quantile_tensor(
    tensor: torch.Tensor,
    *,
    quantiles: Sequence[float],
    requested_indices: Sequence[int],
) -> list[dict[str, Any]]:
    """Extract requested forecast columns from a quantile prediction tensor."""
    sliced = _float32_array(tensor[:, 0, :, requested_indices])
    return [
        {"quantile": float(quantile), "values": sliced[idx]}
        for idx, quantile in enumerate(quantiles)
    ]


def _serialize_log_prob_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    """Serialize per-horizon log density and NLL summaries for one request."""
    if tensor.ndim != 2 or int(tensor.shape[0]) != 1:
        raise RuntimeError("log_prob scoring expected one batch with horizon scores")
    values = tensor[0].detach().cpu()
    nll_values = -values
    return {
        "values": values.tolist(),
        "nll_values": nll_values.tolist(),
        "total": float(values.sum().item()),
        "mean": float(values.mean().item()),
        "nll_total": float(nll_values.sum().item()),
        "nll_mean": float(nll_values.mean().item()),
    }
