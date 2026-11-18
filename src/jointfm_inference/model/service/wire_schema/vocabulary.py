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

"""The enumerated vocabularies of the service's wire contract.

The service owns its request types, query modes, condition kinds, and the
descriptor values it serves, so they are defined here and the request parsers
read them from here. The library owns the return modes
(``INFERENCE_RETURN_MODES``), time-index modes and roles (``TIME_INDEX_MODES``,
``ROLES``), precisions (``MODEL_PRECISIONS``), and the error code catalog
(``ServiceErrorCode``); for those this module only adds descriptions, and fails
at import when a value lacks one or a description names a value the code does
not serve.
"""

from __future__ import annotations

from typing import Final

from jointfm_inference.data.types import ROLES, TIME_INDEX_MODES
from jointfm_inference.model.inference.contract import INFERENCE_RETURN_MODES
from jointfm_inference.model.inference.time_index import (
    DISCRETE_GRID_TIME_INDEX_ENCODING,
)
from jointfm_inference.model.precision import MODEL_PRECISIONS

from ..errors import ERROR_VALUE_FIELDS, ServiceErrorCode
from .fields import WireValue, described_values, values_of

REQUEST_TYPE_PREDICT: Final = "predict"
REQUEST_TYPE_HEALTH: Final = "health"
REQUEST_TYPE_VALUES = (
    WireValue(REQUEST_TYPE_PREDICT, "Answer a forecast or condition query."),
    WireValue(
        REQUEST_TYPE_HEALTH,
        "Return the `/healthz` payload instead of a forecast, ignoring every "
        "other field, for clients behind a gateway that forwards only "
        "`POST /predict`.",
    ),
)
REQUEST_TYPES = values_of(REQUEST_TYPE_VALUES)
DEFAULT_REQUEST_TYPE: Final = REQUEST_TYPE_PREDICT

QUERY_MODE_FORECAST: Final = "forecast"
QUERY_MODE_CONDITION: Final = "condition"
QUERY_MODE_VALUES = (
    WireValue(
        QUERY_MODE_FORECAST,
        "The joint distribution of the columns at each future position, given "
        "the history.",
    ),
    WireValue(
        QUERY_MODE_CONDITION,
        "The same, additionally conditioned at the positions `conditions` "
        "covers on pinned values or bands of some columns; a position no "
        "condition covers answers as in `forecast`. Served only where "
        "`/healthz` lists it in `supported_query_modes`.",
    ),
)
QUERY_MODES = values_of(QUERY_MODE_VALUES)
DEFAULT_QUERY_MODE: Final = QUERY_MODE_FORECAST

RETURN_MODE_VALUES = described_values(
    {
        "mean": "The mean of each requested column at each future position.",
        "samples": "`n_samples` draws from the joint distribution, each "
        "covering every future position and requested column.",
        "quantiles": "The marginal quantiles at the levels `quantiles` lists, "
        "per future position and requested column.",
        "log_prob": "The joint log density of the observed `query_rows` at each "
        "future position.",
    },
    owned=INFERENCE_RETURN_MODES,
    vocabulary="return mode",
)
RETURN_MODES = values_of(RETURN_MODE_VALUES)
DEFAULT_RETURN_MODE: Final = "mean"

TIME_INDEX_MODE_ABSOLUTE_DATETIME: Final = "absolute_datetime"
TIME_INDEX_MODE_VALUES = described_values(
    {
        "ordinal": "Times are integer steps of a regular grid. Without "
        "`time_column`, history rows sit at steps `0` to `rows - 1`.",
        "continuous_float": "Times are real numbers in the column `time_column` names.",
        TIME_INDEX_MODE_ABSOLUTE_DATETIME: "Times are ISO 8601 timestamps in "
        "the column `time_column` names.",
    },
    owned=TIME_INDEX_MODES,
    vocabulary="time-index mode",
)

CONDITION_KIND_EQUALITY: Final = "equality"
CONDITION_KIND_INTERVAL: Final = "interval"
CONDITION_KIND_VALUES = (
    WireValue(
        CONDITION_KIND_EQUALITY,
        "Pins the column to one value at the covered positions.",
    ),
    WireValue(
        CONDITION_KIND_INTERVAL,
        "Confines the column to a band at the covered positions; either side "
        "may be open.",
    ),
)
CONDITION_KINDS = values_of(CONDITION_KIND_VALUES)

ROLE_VALUES = described_values(
    {
        "target": "A column the caller cares about forecasting.",
        "feature": "A column that informs the forecast. The model forecasts "
        "every column whatever its role.",
    },
    owned=ROLES,
    vocabulary="column role",
)
DEFAULT_ROLE: Final = "feature"

SERVED_MODALITY: Final = "numeric"
DEFAULT_CALENDAR_ID: Final = "pandas-default"

PRECISION_VALUES = described_values(
    {
        "fp32": "Every operation runs in float32.",
        "bf16": "Matrix products and attention run in bfloat16 under automatic "
        "mixed precision; numerically sensitive operations stay in float32.",
        "fp16": "As `bf16`, in float16.",
    },
    owned=MODEL_PRECISIONS,
    vocabulary="model precision",
)

TIME_INDEX_ENCODING_VALUES = (
    WireValue(
        DISCRETE_GRID_TIME_INDEX_ENCODING,
        "The model reads time as integer steps of a regular grid.",
    ),
)

ERROR_CODE_VALUES = described_values(
    {
        ServiceErrorCode.VALIDATION_ERROR: "The request is malformed or asks "
        "for something the service does not serve; `field` names the offending "
        "field.",
        ServiceErrorCode.UNSUPPORTED_HEAD_QUERY_COMBINATION: "The mounted head "
        "cannot answer the requested `query_mode`.",
        ServiceErrorCode.UNSUPPORTED_RETURN_MODE: "In the catalog but not sent by "
        "this service: an unserved `return_mode` answers `VALIDATION_ERROR`.",
        ServiceErrorCode.INPUT_SIZE_EXCEEDED: "The request exceeds a service "
        "cap: `n_samples` above `max_sample_count`, or a body above "
        "`max_request_bytes`. `requested` and `allowed` carry both sizes.",
        ServiceErrorCode.SCHEMA_VERSION_MISMATCH: "`schema_version` differs from "
        "the version the service speaks.",
        ServiceErrorCode.MODEL_VERSION_MISMATCH: "`model_version` differs from "
        "the version the service serves.",
        ServiceErrorCode.INTERNAL_ERROR: "The service failed on a request it accepted.",
    },
    owned=frozenset(code.value for code in ServiceErrorCode),
    vocabulary="error code",
)

ERROR_VALUE_VALUES = described_values(
    {
        "requested": "The size the request asked for, in the unit of `field`: "
        "samples for `n_samples`, bytes for `Content-Length`.",
        "allowed": "The largest size the service accepts, in the same unit.",
    },
    owned=frozenset(name for names in ERROR_VALUE_FIELDS.values() for name in names),
    vocabulary="error value",
)

# These defaults name values the library owns, so a library rename must fail
# here rather than leave a request default the service refuses.
if DEFAULT_RETURN_MODE not in RETURN_MODES or DEFAULT_ROLE not in ROLES:
    raise ValueError(
        f"default return mode {DEFAULT_RETURN_MODE!r} or role {DEFAULT_ROLE!r} "
        f"is outside {list(RETURN_MODES)} or {sorted(ROLES)}"
    )
