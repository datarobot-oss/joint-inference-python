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

"""The fields of ``POST /predict`` requests and of the objects they hold.

The request parsers read their key sets and served values from here, so the
document lists exactly the keys the service accepts: a key outside these
tuples is refused by the service and fails validation against the document.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any, Final

from jointfm_inference.data.time_index import DEFAULT_ABSOLUTE_DATETIME_SCALE_SECONDS
from jointfm_inference.data.types import TIME_INDEX_MODES

from ..constants import DEFAULT_SCHEMA_VERSION
from .fields import (
    WireField,
    enum_schema,
    field_names,
    nullable,
    object_schema,
    ref,
    vocabulary_text,
)
from .vocabulary import (
    CONDITION_KIND_EQUALITY,
    CONDITION_KIND_INTERVAL,
    DEFAULT_CALENDAR_ID,
    DEFAULT_QUERY_MODE,
    DEFAULT_REQUEST_TYPE,
    DEFAULT_RETURN_MODE,
    DEFAULT_ROLE,
    QUERY_MODE_CONDITION,
    QUERY_MODE_VALUES,
    REQUEST_TYPE_HEALTH,
    RETURN_MODE_VALUES,
    ROLE_VALUES,
    SERVED_MODALITY,
    TIME_INDEX_MODE_ABSOLUTE_DATETIME,
    TIME_INDEX_MODE_VALUES,
)

_NAME: Final[Mapping[str, Any]] = {"type": "string", "minLength": 1}
_NUMBER: Final[Mapping[str, Any]] = {"type": "number"}
QUERY_TIME_INDICES_FIELD: Final = "query_time_indices"

# Column descriptor fields the service accepts only absent, ``null``, or at
# the listed values.
DEFAULT_ONLY_DESCRIPTOR_VALUES: Final[Mapping[str, tuple[Any, ...]]] = {
    "nullable": (False,),
    "vocabulary_size": (),
    "level_count": (),
    "mapping": (),
    "time_value_kind": (),
    "time_value_scale_seconds": (),
    "time_value_use_local_normalized_time": (False,),
    "time_value_calendar_id": (DEFAULT_CALENDAR_ID,),
    "time_value_timezone": (),
    "traits": ({},),
}


def _default_only_field(name: str) -> WireField:
    """Return the descriptor field ``name``, served only at its default values."""
    served = DEFAULT_ONLY_DESCRIPTOR_VALUES[name]
    passes = " or ".join(f"`{json.dumps(value)}`" for value in (None, *served))
    return WireField(
        name,
        {"enum": [*served, None]},
        f"Not served; omit it or pass {passes}.",
    )


COLUMN_DESCRIPTOR_FIELDS: Final = (
    WireField(
        "name",
        _NAME,
        "The column's name, unique among `columns`: the key of its value in "
        "each row and the name the response reports it under.",
        required=True,
    ),
    WireField(
        "modality",
        {"type": "string", "const": SERVED_MODALITY},
        f"The kind of value the column holds; this service reads "
        f"`{SERVED_MODALITY}` columns only.",
        required=True,
    ),
    WireField(
        "role",
        nullable({**enum_schema(ROLE_VALUES), "default": DEFAULT_ROLE}),
        vocabulary_text(
            f"What the column is to the caller; defaults to `{DEFAULT_ROLE}`.",
            ROLE_VALUES,
        ),
    ),
    WireField(
        "lower_bound",
        nullable(_NUMBER),
        "A lower bound every sampled draw of the column respects; at most "
        "`upper_bound`.",
    ),
    WireField(
        "upper_bound",
        nullable(_NUMBER),
        "An upper bound every sampled draw of the column respects.",
    ),
    *(_default_only_field(name) for name in DEFAULT_ONLY_DESCRIPTOR_VALUES),
)
COLUMN_DESCRIPTOR_FIELD_NAMES: Final = field_names(COLUMN_DESCRIPTOR_FIELDS)

_CONDITION_COLUMN = WireField(
    "column",
    _NAME,
    "The name of the declared column the condition applies to.",
    required=True,
)
_QUERY_TIME_INDICES = WireField(
    QUERY_TIME_INDICES_FIELD,
    nullable(
        {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
            "minItems": 1,
            "uniqueItems": True,
        }
    ),
    "Zero-based indices into `query_times` of the future positions the "
    "condition covers; `null` covers every position.",
    required=True,
)
EQUALITY_CONDITION_FIELDS: Final = (
    _CONDITION_COLUMN,
    WireField(
        "kind",
        {"type": "string", "const": CONDITION_KIND_EQUALITY},
        "The condition kind.",
        required=True,
    ),
    WireField(
        "value",
        _NUMBER,
        "The finite value the column is pinned to.",
        required=True,
    ),
    _QUERY_TIME_INDICES,
)
INTERVAL_CONDITION_FIELDS: Final = (
    _CONDITION_COLUMN,
    WireField(
        "kind",
        {"type": "string", "const": CONDITION_KIND_INTERVAL},
        "The condition kind.",
        required=True,
    ),
    WireField(
        "lower",
        nullable(_NUMBER),
        "The band's finite lower end, below `upper`; `null` leaves it open.",
        required=True,
    ),
    WireField(
        "upper",
        nullable(_NUMBER),
        "The band's finite upper end; `null` leaves it open. A band open on "
        "both sides conditions nothing and is refused.",
        required=True,
    ),
    _QUERY_TIME_INDICES,
)
EQUALITY_CONDITION_FIELD_NAMES: Final = field_names(EQUALITY_CONDITION_FIELDS)
INTERVAL_CONDITION_FIELD_NAMES: Final = field_names(INTERVAL_CONDITION_FIELDS)

PREDICT_REQUEST_FIELDS: Final = (
    WireField(
        "request_type",
        {
            "type": "string",
            "enum": [DEFAULT_REQUEST_TYPE],
            "default": DEFAULT_REQUEST_TYPE,
        },
        f"`{DEFAULT_REQUEST_TYPE}`, the default. A request whose `request_type` "
        f"is `{REQUEST_TYPE_HEALTH}` is a `HealthRequest` instead.",
    ),
    WireField(
        "schema_version",
        {"type": "string", "const": DEFAULT_SCHEMA_VERSION},
        "The version of this contract the request speaks; any other is refused "
        "with `SCHEMA_VERSION_MISMATCH`.",
        required=True,
    ),
    WireField(
        "model_version",
        nullable(_NAME),
        "When set, the service refuses the request with "
        "`MODEL_VERSION_MISMATCH` unless it serves exactly this "
        "`model_version`, as `/healthz` reports it.",
    ),
    WireField(
        "query_mode",
        {**enum_schema(QUERY_MODE_VALUES), "default": DEFAULT_QUERY_MODE},
        vocabulary_text(
            f"What the request asks of the distribution; defaults to "
            f"`{DEFAULT_QUERY_MODE}`.",
            QUERY_MODE_VALUES,
        ),
    ),
    WireField(
        "return_mode",
        {**enum_schema(RETURN_MODE_VALUES), "default": DEFAULT_RETURN_MODE},
        vocabulary_text(
            f"What the response carries; defaults to `{DEFAULT_RETURN_MODE}`. "
            "`/healthz` lists the modes a service serves in "
            "`supported_return_modes`.",
            RETURN_MODE_VALUES,
        ),
    ),
    WireField(
        "time_index_mode",
        enum_schema(TIME_INDEX_MODE_VALUES),
        vocabulary_text(
            "How the request's times read. `/healthz` lists the modes a service "
            "serves in `supported_time_index_modes`.",
            TIME_INDEX_MODE_VALUES,
        ),
        required=True,
    ),
    WireField(
        "time_column",
        nullable(_NAME),
        "The row key holding each row's time; it must not name a declared "
        "column. Required unless `time_index_mode` is `ordinal`.",
    ),
    WireField(
        "time_scale_seconds",
        nullable({"type": "number", "exclusiveMinimum": 0}),
        "Seconds per time unit the model reads `absolute_datetime` times in; "
        f"defaults to {DEFAULT_ABSOLUTE_DATETIME_SCALE_SECONDS:.0f}, one Julian "
        "year.",
    ),
    WireField(
        "use_local_normalized_time",
        nullable({"type": "boolean", "const": False}),
        "Not served; omit it or pass `false`.",
    ),
    WireField(
        "calendar_id",
        nullable({"type": "string", "const": DEFAULT_CALENDAR_ID}),
        f"Not served beyond the default; omit it or pass `{DEFAULT_CALENDAR_ID}`.",
    ),
    WireField(
        "timezone",
        nullable(_NAME),
        "The IANA zone name, such as `Europe/Berlin`, of the "
        "`absolute_datetime` timestamps; it must match the zone the timestamps "
        "carry, if any.",
    ),
    WireField(
        "columns",
        {"type": "array", "items": ref("ColumnDescriptor"), "minItems": 1},
        "The columns of every row, in the order the response reports them by "
        "default. `/healthz` reports the column counts the model was built for "
        "in `data_generation`.",
        required=True,
    ),
    WireField(
        "requested_columns",
        nullable(
            {
                "type": "array",
                "items": {
                    "anyOf": [_NAME, {"type": "integer", "minimum": 0}],
                },
                "minItems": 1,
                "uniqueItems": True,
            }
        ),
        "The columns the response carries, in this order, each by name or by "
        "zero-based index into `columns`; omitted or `null`, every declared "
        "column in declared order. A `log_prob` request names every declared "
        "column in declared order or omits the field.",
    ),
    WireField(
        "history_rows",
        {"type": "array", "items": ref("Row"), "minItems": 1},
        "The observed rows, in strictly increasing time order. Each holds a "
        "finite number for every declared column and, when `time_column` is "
        "set, the row's time.",
        required=True,
    ),
    WireField(
        "query_times",
        {"type": "array", "items": {"type": ["string", "number"]}, "minItems": 1},
        "The future times to answer, strictly increasing and after the last "
        "history time: ISO 8601 timestamps for `absolute_datetime`, numbers "
        "otherwise. Each is one future position of the response.",
        required=True,
    ),
    WireField(
        "query_rows",
        {"type": "array", "items": ref("Row"), "minItems": 1},
        "Read only, and required, for `log_prob`: the observed row at each "
        "query time, in `query_times` order, with a finite value for every "
        "declared column and, when `time_column` is set, its query time.",
    ),
    WireField(
        "conditions",
        {"type": "array", "items": ref("Condition"), "minItems": 1},
        f"Required for, and allowed only with, `query_mode` "
        f"`{QUERY_MODE_CONDITION}`. A column carries at most one condition per "
        "position, and every position keeps at least one column unconditioned.",
    ),
    WireField(
        "n_samples",
        {"type": "integer", "minimum": 1},
        "The number of draws behind `samples` and behind any answer the service "
        "estimates by sampling. Defaults to the service's `max_sample_count`, "
        "which `/healthz` reports and no request may exceed "
        "(`INPUT_SIZE_EXCEEDED`).",
    ),
    WireField(
        "quantiles",
        {
            "type": "array",
            "items": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
            "minItems": 1,
        },
        "Read only, and required, for `quantiles`: the levels, each strictly "
        "between 0 and 1, in the order the response reports them.",
    ),
    WireField(
        "seed",
        nullable({"type": "integer"}),
        "Seeds the service's random draws, so the same request with the same "
        "seed returns the same answer; omitted or `null`, draws differ per "
        "request.",
    ),
)
REQUEST_FIELDS: Final = field_names(PREDICT_REQUEST_FIELDS)

# Per time-index mode: whether a request must name ``time_column``, and the
# JSON type of its ``query_times``.
_TIME_INDEX_MODE_RULES: Final[Mapping[str, tuple[bool, str]]] = {
    "ordinal": (False, "number"),
    "continuous_float": (True, "number"),
    TIME_INDEX_MODE_ABSOLUTE_DATETIME: (True, "string"),
}
if set(_TIME_INDEX_MODE_RULES) != TIME_INDEX_MODES:
    raise ValueError(
        f"time-index mode rules cover {sorted(_TIME_INDEX_MODE_RULES)}, the "
        f"library serves {sorted(TIME_INDEX_MODES)}"
    )


def _when(field: str, value: str, then: Mapping[str, Any]) -> dict[str, Any]:
    """Return the rule applying ``then`` to a request whose ``field`` is ``value``.

    The ``required`` keeps a request that omits ``field`` from matching, since
    an omitted mode takes its default rather than ``value``.
    """
    return {
        "if": {"required": [field], "properties": {field: {"const": value}}},
        "then": dict(then),
    }


def _time_index_rule(mode: str) -> dict[str, Any]:
    """Return the rule a request in time-index ``mode`` follows."""
    needs_time_column, query_time_type = _TIME_INDEX_MODE_RULES[mode]
    properties: dict[str, Any] = {"query_times": {"items": {"type": query_time_type}}}
    then: dict[str, Any] = {"properties": properties}
    if needs_time_column:
        then["required"] = ["time_column"]
        properties["time_column"] = {"type": "string"}
    return _when("time_index_mode", mode, then)


PREDICT_REQUEST_RULES: Final = (
    *(_time_index_rule(mode) for mode in _TIME_INDEX_MODE_RULES),
    _when("return_mode", "quantiles", {"required": ["quantiles"]}),
    _when("return_mode", "log_prob", {"required": ["query_rows"]}),
    {
        **_when("query_mode", QUERY_MODE_CONDITION, {"required": ["conditions"]}),
        "else": {"not": {"required": ["conditions"]}},
    },
)


def request_components() -> dict[str, dict[str, Any]]:
    """Return the request schemas of the document's ``components.schemas``."""
    predict_request = object_schema(
        PREDICT_REQUEST_FIELDS,
        description=(
            "A forecast or condition query over one table of rows. Rules that "
            "depend on the modes are listed under `allOf`."
        ),
    )
    predict_request["allOf"] = list(PREDICT_REQUEST_RULES)
    return {
        "Request": {
            "description": "The body of `POST /predict`.",
            "oneOf": [ref("PredictRequest"), ref("HealthRequest")],
        },
        "PredictRequest": predict_request,
        "HealthRequest": {
            "type": "object",
            "description": (
                "Asks for the `/healthz` payload through `POST /predict`. The "
                "service reads no other field."
            ),
            "properties": {
                "request_type": {"type": "string", "const": REQUEST_TYPE_HEALTH}
            },
            "required": ["request_type"],
        },
        "ColumnDescriptor": object_schema(
            COLUMN_DESCRIPTOR_FIELDS, description="One column of every row."
        ),
        "Row": {
            "type": "object",
            "description": (
                "One row: a value per declared column, keyed by the column's "
                "name, and the row's time under `time_column`. Other keys are "
                "ignored."
            ),
            "additionalProperties": {"type": ["number", "string"]},
        },
        "Condition": {
            "description": "One condition, told apart by `kind`.",
            "oneOf": [ref("EqualityCondition"), ref("IntervalCondition")],
            "discriminator": {
                "propertyName": "kind",
                "mapping": {
                    CONDITION_KIND_EQUALITY: ref("EqualityCondition")["$ref"],
                    CONDITION_KIND_INTERVAL: ref("IntervalCondition")["$ref"],
                },
            },
        },
        "EqualityCondition": object_schema(
            EQUALITY_CONDITION_FIELDS,
            description="Pins one column to one value at the covered positions.",
        ),
        "IntervalCondition": object_schema(
            INTERVAL_CONDITION_FIELDS,
            description="Confines one column to a band at the covered positions.",
        ),
    }
