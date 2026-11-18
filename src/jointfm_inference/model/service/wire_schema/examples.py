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

"""Example requests: one per query mode and return mode, and a health request.

The examples share one small table of daily rows, so any served checkpoint
answers them, and a test sends each one to a service to check that it is
accepted and that the answer matches the document. The condition examples pin
one column at the first position and band it at the others, and the observed
rows of the ``log_prob`` examples keep to those conditions.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, Final

from ..constants import DEFAULT_SCHEMA_VERSION
from .vocabulary import (
    CONDITION_KIND_EQUALITY,
    CONDITION_KIND_INTERVAL,
    QUERY_MODE_CONDITION,
    QUERY_MODE_FORECAST,
    QUERY_MODES,
    REQUEST_TYPE_HEALTH,
    RETURN_MODES,
    TIME_INDEX_MODE_ABSOLUTE_DATETIME,
)

_TIME_COLUMN: Final = "timestamp"
_FEATURE: Final = "temperature"
_TARGET: Final = "demand"
_HISTORY: Final = (
    (3.1, 41.0),
    (2.4, 44.5),
    (1.8, 47.2),
    (2.9, 43.1),
    (3.6, 39.8),
    (4.0, 38.6),
    (3.2, 40.9),
    (2.7, 42.3),
)
_FUTURE: Final = ((1.5, 46.0), (2.2, 44.1), (3.0, 41.7))
_PINNED: Final = _FUTURE[0][0]


def _timestamp(day: int) -> str:
    """Return midnight UTC of ``day`` in January 2026 as an ISO 8601 string."""
    return f"2026-01-{day:02d}T00:00:00Z"


def _row(day: int, values: tuple[float, float]) -> dict[str, Any]:
    """Return one row of the example table."""
    return {_TIME_COLUMN: _timestamp(day), _FEATURE: values[0], _TARGET: values[1]}


_FIRST_FUTURE_DAY: Final = len(_HISTORY) + 1
_BASE: Final[Mapping[str, Any]] = {
    "schema_version": DEFAULT_SCHEMA_VERSION,
    "time_index_mode": TIME_INDEX_MODE_ABSOLUTE_DATETIME,
    "time_column": _TIME_COLUMN,
    "columns": [
        {"name": _FEATURE, "modality": "numeric", "role": "feature"},
        {"name": _TARGET, "modality": "numeric", "role": "target", "lower_bound": 0.0},
    ],
    "history_rows": [_row(day, values) for day, values in enumerate(_HISTORY, 1)],
    "query_times": [
        _timestamp(day)
        for day in range(_FIRST_FUTURE_DAY, _FIRST_FUTURE_DAY + len(_FUTURE))
    ],
    "seed": 7,
}
_RETURN_MODE_FIELDS: Final[Mapping[str, Mapping[str, Any]]] = {
    "mean": {"requested_columns": [_TARGET]},
    "samples": {"requested_columns": [_TARGET], "n_samples": 4},
    "quantiles": {"requested_columns": [_TARGET], "quantiles": [0.1, 0.5, 0.9]},
    "log_prob": {
        "query_rows": [
            _row(day, values) for day, values in enumerate(_FUTURE, _FIRST_FUTURE_DAY)
        ]
    },
}
_QUERY_MODE_FIELDS: Final[Mapping[str, Mapping[str, Any]]] = {
    QUERY_MODE_FORECAST: {},
    QUERY_MODE_CONDITION: {
        "conditions": [
            {
                "column": _FEATURE,
                "kind": CONDITION_KIND_EQUALITY,
                "value": _PINNED,
                "query_time_indices": [0],
            },
            {
                "column": _FEATURE,
                "kind": CONDITION_KIND_INTERVAL,
                "lower": 0.0,
                "upper": None,
                "query_time_indices": list(range(1, len(_FUTURE))),
            },
        ]
    },
}

HEALTH_REQUEST_EXAMPLE: Final = "health"


def example_name(query_mode: str, return_mode: str) -> str:
    """Return the document's name for the example of one mode pair."""
    return f"{query_mode}_{return_mode}"


def example_requests() -> dict[str, dict[str, Any]]:
    """Return a fresh copy of every example request, keyed by its name.

    One example per query mode and return mode, named by
    :func:`example_name`, and the health request under
    ``HEALTH_REQUEST_EXAMPLE``.
    """
    examples = {
        example_name(query_mode, return_mode): copy.deepcopy(
            {
                **_BASE,
                "query_mode": query_mode,
                "return_mode": return_mode,
                **_RETURN_MODE_FIELDS[return_mode],
                **_QUERY_MODE_FIELDS[query_mode],
            }
        )
        for query_mode in QUERY_MODES
        for return_mode in RETURN_MODES
    }
    examples[HEALTH_REQUEST_EXAMPLE] = {"request_type": REQUEST_TYPE_HEALTH}
    return examples
