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

"""Fixtures shared by the service-level tests of the ``condition`` query mode.

A tiny JointFM 1 network whose forward is replaced by one fixed Student-t
mixture parameter mapping, the services mounted over it, and the request
payloads the tests send. Everything here describes *what is mounted and sent*;
the in-process oracles the tests compare against stay next to the tests that
use them.
"""

import math
from typing import Any

import pytest
import torch

from jointfm_inference.model.inference import DEFAULT_INFERENCE_PRECISION
from jointfm_inference.model.jointfm_1.model import JointFM
from jointfm_inference.model.service import JointFMHTTPService
from jointfm_inference.model.service.constants import DEFAULT_SCHEMA_VERSION

from tests.support.models import TINY_CONFIG, tiny_model
from tests.support.service import mount_service

COLUMN_COUNT = 2
# Width of the fixture whose third column, ``extra``, correlates with the driver.
THREE_COLUMNS = ("driver", "target", "extra")
COMPONENT_COUNT = TINY_CONFIG.n_mixtures
HORIZON_COUNT = 2
# Low enough that the tail mass separates clearly from a normal one at the
# bounds these tests use.
STUDENT_T_DF = 5.0
# Large enough for the Monte Carlo oracles these tests compare against.
MAX_SAMPLE_COUNT = 20000


class FixedParameterForward:
    """A network forward that returns one fixed parameter mapping and counts calls."""

    def __init__(self, params: dict[str, torch.Tensor]) -> None:
        """Bind the parameters every call returns."""
        self._params = params
        self.calls = 0

    def __call__(
        self,
        past: torch.Tensor,
        horizons: torch.Tensor,
        past_time_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return the fixed parameters, recording that execution happened."""
        del past, horizons, past_time_indices
        self.calls += 1
        return dict(self._params)


def _mount(model: JointFM) -> JointFMHTTPService:
    """Serve ``model`` at the default precision a deployment runs at."""
    return mount_service(
        model, precision=DEFAULT_INFERENCE_PRECISION, max_sample_count=MAX_SAMPLE_COUNT
    )


def parameters(
    *, column_count: int = COLUMN_COUNT, lowrank: bool = False
) -> dict[str, torch.Tensor]:
    """Two components that disagree about location and about cross-column sign.

    The first two columns are the fixture every in-process oracle reads; a
    third column, kept for ``column_count=3``, correlates with the driver so
    that a path conditioning the wrong column changes a number. Both scale
    parameterizations describe one distribution: the dense form is a Cholesky
    factor, and the diagonal-plus-factor form reproduces its first two columns
    exactly.
    """
    if column_count not in (COLUMN_COUNT, len(THREE_COLUMNS)):
        raise ValueError(f"the fixture has two or three columns, got {column_count}")
    width = len(THREE_COLUMNS)
    means = torch.full((1, HORIZON_COUNT, COMPONENT_COUNT, width), 3.0)
    means[..., :COLUMN_COUNT] = torch.tensor(
        [[[[0.0, 1.0], [1.5, -0.5]], [[0.5, 2.0], [-1.0, 0.25]]]]
    )
    params = {
        "logits": torch.tensor([[[0.3, -0.2], [0.1, 0.4]]]),
        "means": means[..., :column_count].contiguous(),
        "df": torch.full((1, HORIZON_COUNT, COMPONENT_COUNT), STUDENT_T_DF),
    }
    if lowrank:
        # diag + u u^T reproduces each component's leading scale block:
        # 0.7 = a * b with a = b = sqrt(0.7), and -0.9 = 1.0 * -0.9, the
        # diagonal absorbing the rest of the scale.
        factor = torch.tensor(
            [[math.sqrt(0.7), math.sqrt(0.7), 0.5], [1.0, -0.9, 0.5]]
        ).reshape(1, 1, COMPONENT_COUNT, width, 1)
        diag = torch.tensor([[0.3, 0.3, 1.0], [1.0, 0.69, 1.0]]).reshape(
            1, 1, COMPONENT_COUNT, width
        )
        params["lowrank_diag"] = (
            diag[..., :column_count]
            .expand(1, HORIZON_COUNT, COMPONENT_COUNT, column_count)
            .contiguous()
        )
        params["lowrank_U"] = (
            factor[..., :column_count, :]
            .expand(1, HORIZON_COUNT, COMPONENT_COUNT, column_count, 1)
            .contiguous()
        )
        return params
    scale = torch.stack(
        [
            torch.tensor([[1.0, 0.7], [0.7, 1.0]]),
            torch.tensor([[2.0, -0.9], [-0.9, 1.5]]),
        ]
    )
    chol = torch.eye(width).repeat(1, HORIZON_COUNT, COMPONENT_COUNT, 1, 1)
    chol[..., :COLUMN_COUNT, :COLUMN_COUNT] = torch.linalg.cholesky(scale)
    chol[..., COLUMN_COUNT:, 0] = 0.5
    params["chol"] = chol[..., :column_count, :column_count].contiguous()
    return params


def service_and_model(
    *, params: dict[str, torch.Tensor] | None = None
) -> tuple[JointFMHTTPService, FixedParameterForward]:
    """Build one service over a network that always emits ``params``.

    The network is a real JointFM 1 instance so the backend accepts it; only
    its forward is replaced, which leaves the backend's precision handling and
    the head's algebra in the served path.
    """
    model = tiny_model()
    forward = FixedParameterForward(params or parameters())
    # An instance attribute shadows the class forward for this network only.
    model.forward = forward  # ty: ignore[invalid-assignment]
    return _mount(model), forward


def random_weight_service() -> JointFMHTTPService:
    """Mount a network with random weights whose forward is the real one.

    A real forward hands back more than the head's parameters: the history's
    standardization center and scale ride along. Weights are random because no
    assertion reads a number the network produced; what is under test is that
    the served request reaches an answer at all.
    """
    return _mount(tiny_model())


def make_payload(**overrides: Any) -> dict[str, Any]:
    """Build one condition request over two numeric columns and two future rows."""
    payload: dict[str, Any] = {
        "schema_version": DEFAULT_SCHEMA_VERSION,
        "query_mode": "condition",
        "return_mode": "mean",
        "time_index_mode": "absolute_datetime",
        "time_column": "timestamp",
        "columns": [
            {"name": "driver", "modality": "numeric"},
            {"name": "target", "modality": "numeric", "role": "target"},
        ],
        "history_rows": [
            {
                "timestamp": f"2026-01-0{index + 1}T00:00:00Z",
                "driver": 1.0 + index / 10,
                "target": 10.0 + index,
            }
            for index in range(4)
        ],
        "query_times": ["2026-01-05T00:00:00Z", "2026-01-06T00:00:00Z"],
        "requested_columns": ["target"],
        "conditions": [
            {
                "column": "driver",
                "kind": "equality",
                "value": 0.75,
                "query_time_indices": [1],
            }
        ],
        "seed": 7,
    }
    payload.update(overrides)
    return payload


def make_three_column_payload(**overrides: Any) -> dict[str, Any]:
    """Build one condition request over the three-column fixture's columns."""
    payload = make_payload(
        columns=[
            {"name": name, "modality": "numeric", "role": "target"}
            for name in THREE_COLUMNS
        ],
        history_rows=[
            {
                "timestamp": f"2026-01-0{index + 1}T00:00:00Z",
                **{name: 1.0 + index / 10 for name in THREE_COLUMNS},
            }
            for index in range(4)
        ],
    )
    payload.update(overrides)
    return payload


def three_column_service(
    *, lowrank: bool = False
) -> tuple[JointFMHTTPService, FixedParameterForward]:
    """Mount the three-column mixture, dense or diagonal-plus-factor."""
    return service_and_model(
        params=parameters(column_count=len(THREE_COLUMNS), lowrank=lowrank)
    )


def bounded_columns(*names: str) -> list[dict[str, Any]]:
    """Declare request columns carrying the sampling bounds a real request has.

    The bounds are wide enough to reject nothing, so an oracle stays exact;
    their presence is what makes the service build per-column bound tensors.
    """
    return [
        {"name": name, "modality": "numeric", "role": "target", "lower_bound": -50.0}
        for name in names
    ]


PARAMETERIZATIONS = pytest.mark.parametrize(
    "lowrank", [False, True], ids=["dense", "diagonal_plus_factor"]
)
