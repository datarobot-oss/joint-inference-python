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

"""Response encoding of the inference service.

Every served return mode passes through one encoder, which writes NumPy
arrays directly and refuses any value the wire would misread as missing.
"""

import json
import math
from typing import Any

import numpy as np
import pytest
import torch

from jointfm_inference.model.jointfm_1.model import JointFM
from jointfm_inference.model.service import (
    DEFAULT_SCHEMA_VERSION,
    JointFMHTTPService,
    encode_response,
)
from jointfm_inference.model.service.serialization import (
    _slice_forecast_tensor,
    _slice_quantile_tensor,
    _slice_sample_tensor,
)
from tests.support.models import TINY_CONFIG, tiny_model
from tests.support.service import mount_service

_SAMPLE_COUNT = 5
# Finite in float32, but a draw of this scale around this location overflows.
_OVERFLOWING_MAGNITUDE = 3e38


class _OverflowingJointFM(JointFM):
    """A JointFM 1 network whose finite emitted parameters draw infinite values."""

    def __init__(self) -> None:
        super().__init__(TINY_CONFIG)

    def forward(
        self,
        past: torch.Tensor,
        horizons: torch.Tensor,
        past_time_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del past_time_indices
        horizon_count, column_count = horizons.shape[0], past.shape[-1]
        return {
            "logits": torch.zeros(1, horizon_count, 1),
            "means": torch.full(
                (1, horizon_count, 1, column_count), _OVERFLOWING_MAGNITUDE
            ),
            "df": torch.full((1, horizon_count, 1), 5.0),
            "chol": (_OVERFLOWING_MAGNITUDE * torch.eye(column_count))
            .expand(1, horizon_count, 1, column_count, column_count)
            .clone(),
        }


def _service(model: JointFM | None = None) -> JointFMHTTPService:
    return mount_service(tiny_model() if model is None else model)


def _samples_payload() -> dict[str, Any]:
    return {
        "schema_version": DEFAULT_SCHEMA_VERSION,
        "return_mode": "samples",
        "time_index_mode": "ordinal",
        "time_column": "step",
        "columns": [
            {"name": "driver", "modality": "numeric"},
            {"name": "target", "modality": "numeric", "role": "target"},
        ],
        "history_rows": [
            {"step": step, "driver": 1.0 + step / 10, "target": 10.0 + step}
            for step in range(4)
        ],
        "query_times": [4, 5],
        "n_samples": _SAMPLE_COUNT,
        "seed": 7,
    }


def test_encode_response_round_trips_float32_arrays_bit_exactly() -> None:
    """A float32 array is written as shortest decimals that read back to the same bits."""
    values = np.array([[0.1, 101.23456573486328], [1e-8, -3.5]], dtype=np.float32)

    body = encode_response({"values": values, "count": 2})
    decoded = json.loads(body)

    assert body.startswith(b'{"count":2,')
    np.testing.assert_array_equal(
        np.asarray(decoded["values"], dtype=np.float32), values
    )


def test_forecast_slices_refuse_non_float32_tensors() -> None:
    """Another dtype means an upstream contract changed; casting would hide it."""
    with pytest.raises(RuntimeError, match="float32"):
        _slice_sample_tensor(torch.zeros(1, 2, 3, 4, dtype=torch.float64), [0])
    with pytest.raises(RuntimeError, match="float32"):
        _slice_quantile_tensor(
            torch.zeros(2, 1, 3, 4, dtype=torch.bfloat16),
            quantiles=[0.1, 0.9],
            requested_indices=[0],
        )


def test_forecast_slices_keep_the_requested_columns_in_request_order() -> None:
    mean = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    quantiles = torch.stack([mean, mean + 10.0])

    assert _slice_forecast_tensor(mean, [2, 0]).tolist() == [[3.0, 1.0], [6.0, 4.0]]
    assert _slice_sample_tensor(mean.unsqueeze(1), [1]).tolist() == [[[2.0], [5.0]]]
    sliced = _slice_quantile_tensor(
        quantiles, quantiles=[0.1, 0.9], requested_indices=[1]
    )
    assert [entry["quantile"] for entry in sliced] == [0.1, 0.9]
    assert [entry["values"].tolist() for entry in sliced] == [
        [[2.0], [5.0]],
        [[12.0], [15.0]],
    ]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_values_the_wire_would_misread_as_missing_are_refused(value: float) -> None:
    """orjson writes NaN and infinities as null, so neither may leave unchecked."""
    draws = torch.tensor([[[[1.5, value]]]])

    with pytest.raises(RuntimeError, match="must be finite"):
        _slice_sample_tensor(draws, [0, 1])
    with pytest.raises(RuntimeError, match="must be finite"):
        _slice_forecast_tensor(draws[:, 0], [0, 1])
    with pytest.raises(RuntimeError, match="must be finite"):
        _slice_quantile_tensor(
            draws.transpose(0, 1), quantiles=[0.5], requested_indices=[0, 1]
        )


def test_predict_returns_the_decoded_response_bytes() -> None:
    """In-process callers see exactly the JSON document an HTTP client decodes."""
    service = _service()
    request = _samples_payload()

    body = service.predict_response_bytes(request)
    decoded = json.loads(body)

    assert decoded == service.predict(request)
    samples = decoded["outputs"]["samples"]
    assert np.asarray(samples).shape == (_SAMPLE_COUNT, len(request["query_times"]), 2)
    assert list(decoded) == sorted(decoded)


@pytest.mark.parametrize("return_mode", ["mean", "samples"])
def test_a_forecast_that_overflows_fails_instead_of_answering_null(
    return_mode: str,
) -> None:
    """A response never carries ``null``: a non-finite forecast is a server error."""
    service = _service(_OverflowingJointFM())
    request = {**_samples_payload(), "return_mode": return_mode}

    with pytest.raises(RuntimeError, match="must be finite"):
        service.predict_response_bytes(request)
