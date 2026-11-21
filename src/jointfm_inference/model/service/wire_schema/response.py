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

"""The fields of the service's responses: forecasts, health, and errors.

A forecast response has one variant per return mode, told apart by
``return_mode``: each variant fills its own ``outputs`` entry and leaves the
other modes' entries ``null``. The service builds these payloads itself; the
acceptance test validates every one it sends against the schemas here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import Any, Final

from jointfm_inference.model.checkpoint import JOINTFM_1_GENERATION
from jointfm_inference.model.inference.capacity_envelope import CapacityEnvelope
from jointfm_inference.model.inference.contract import NO_TAIL_MAP

from ..constants import DEFAULT_SCHEMA_VERSION
from ..errors import ERROR_VALUE_FIELDS
from .fields import (
    WireField,
    described_values,
    enum_schema,
    nullable,
    object_schema,
    ref,
    vocabulary_text,
)
from .vocabulary import (
    CONDITION_KIND_VALUES,
    ERROR_CODE_VALUES,
    ERROR_VALUE_VALUES,
    PRECISION_VALUES,
    QUERY_MODE_CONDITION,
    QUERY_MODE_VALUES,
    RETURN_MODE_VALUES,
    RETURN_MODES,
    TIME_INDEX_ENCODING_VALUES,
    TIME_INDEX_MODE_VALUES,
)

_NUMBER: Final[Mapping[str, Any]] = {"type": "number"}
_STRING: Final[Mapping[str, Any]] = {"type": "string"}
_COUNT: Final[Mapping[str, Any]] = {"type": "integer", "minimum": 0}
_CAP: Final[Mapping[str, Any]] = {"type": "integer", "minimum": 1}
_SCHEMA_VERSION: Final[Mapping[str, Any]] = {
    "type": "string",
    "const": DEFAULT_SCHEMA_VERSION,
}
_MATRIX: Final[Mapping[str, Any]] = {
    "type": "array",
    "items": {"type": "array", "items": _NUMBER},
}


def _array_of(items: Mapping[str, Any]) -> dict[str, Any]:
    """Return the schema of a JSON array of ``items``."""
    return {"type": "array", "items": dict(items)}


def _all_required(*entries: WireField) -> tuple[WireField, ...]:
    """Return ``entries`` with every field required."""
    return tuple(
        WireField(entry.name, entry.schema, entry.description, required=True)
        for entry in entries
    )


_CAPACITY_ENVELOPE_VALUES = described_values(
    {
        "min_series": "The fewest columns the model was built for.",
        "max_series": "The most columns the model was built for.",
        "t_input": "The time span of the longest history the model was built for.",
        "t_output": "The time span of the furthest horizon the model was built for.",
        "n_input": "The longest history the model was built for, in steps.",
        "n_output": "The longest horizon the model was built for, in steps.",
    },
    owned=frozenset(entry.name for entry in fields(CapacityEnvelope)),
    vocabulary="capacity envelope field",
)
# The capacity envelope holds non-negative step and column counts and positive
# time spans; the dataclass annotations say which a field is.
_CAPACITY_ENVELOPE_TYPES: Final[Mapping[str, Mapping[str, Any]]] = {
    "int": _COUNT,
    "float": {"type": "number", "exclusiveMinimum": 0},
}
_CAPACITY_ENVELOPE_ANNOTATIONS = {
    entry.name: entry.type for entry in fields(CapacityEnvelope)
}
CAPACITY_ENVELOPE_FIELDS: Final = tuple(
    WireField(
        value.value,
        _CAPACITY_ENVELOPE_TYPES[str(_CAPACITY_ENVELOPE_ANNOTATIONS[value.value])],
        value.description,
        required=True,
    )
    for value in _CAPACITY_ENVELOPE_VALUES
)

HEALTH_FIELDS: Final = _all_required(
    WireField(
        "status",
        {"type": "string", "const": "ok"},
        "`ok` once the service has loaded its checkpoint and serves requests.",
    ),
    WireField(
        "schema_version",
        _SCHEMA_VERSION,
        "The version of this contract the service speaks, which every "
        "prediction request sends.",
    ),
    WireField(
        "image_version", _STRING, "The version of the `jointfm-inference` package."
    ),
    WireField(
        "model_version",
        _STRING,
        "Identifies the served model; a request may send it to be refused "
        "after the service changes models.",
    ),
    WireField(
        "checkpoint_version",
        _STRING,
        "A label for the served checkpoint, by default the start of its "
        "SHA-256 digest.",
    ),
    WireField(
        "checkpoint_path", _STRING, "Where the service reads the checkpoint file."
    ),
    WireField(
        "device",
        _STRING,
        "The device every model forward runs on, such as `cpu` or `cuda`.",
    ),
    WireField(
        "precision",
        enum_schema(PRECISION_VALUES),
        vocabulary_text("The precision every model forward runs at.", PRECISION_VALUES),
    ),
    WireField(
        "generation",
        _STRING,
        f"The generation of the served checkpoint, such as `{JOINTFM_1_GENERATION}`.",
    ),
    WireField("head", _STRING, "The kind of the served model's output head."),
    WireField(
        "tail_map",
        _STRING,
        f"The per-column tail map the head's forecast applies, or `{NO_TAIL_MAP}`.",
    ),
    WireField(
        "decoding_strategy",
        _STRING,
        "How the served model decodes future positions. A JointFM 1 model "
        "reports `parallel_dense`: one forward emits the joint "
        "distribution of every position, and each position is drawn on its "
        "own, so a draw is joint across the columns of a position and "
        "independent across positions.",
    ),
    WireField(
        "supported_query_modes",
        _array_of(enum_schema(QUERY_MODE_VALUES)),
        "The query modes the service answers.",
    ),
    WireField(
        "supported_condition_kinds",
        _array_of(enum_schema(CONDITION_KIND_VALUES)),
        "The condition kinds a `condition` query may use.",
    ),
    WireField(
        "supported_return_modes",
        _array_of(enum_schema(RETURN_MODE_VALUES)),
        "The return modes the service answers.",
    ),
    WireField(
        "supported_time_index_modes",
        _array_of(enum_schema(TIME_INDEX_MODE_VALUES)),
        "The time-index modes the service reads.",
    ),
    WireField(
        "time_index_encoding",
        enum_schema(TIME_INDEX_ENCODING_VALUES),
        vocabulary_text("How the served model reads time.", TIME_INDEX_ENCODING_VALUES),
    ),
    WireField(
        "max_sample_count",
        _CAP,
        "The largest `n_samples` a request may ask for, and its default.",
    ),
    WireField(
        "max_request_bytes",
        _CAP,
        "The largest request body, in bytes, the service reads; a larger one "
        "is refused unread with `INPUT_SIZE_EXCEEDED`.",
    ),
    WireField(
        "max_concurrent_requests",
        _CAP,
        "How many prediction requests the service runs at once; further ones "
        "wait for a free slot.",
    ),
    WireField(
        "data_generation",
        ref("CapacityEnvelope"),
        "The request sizes the served model was built for.",
    ),
)

# What each return mode's ``outputs`` entry holds; the entries of the other
# modes are ``null``.
_OUTPUT_FIELDS = described_values(
    {
        "mean": "Indexed `[position][column]`: the mean of each requested column "
        "at each future position.",
        "samples": "Indexed `[draw][position][column]`: `n_samples` joint draws.",
        "quantiles": "One entry per requested level, in request order.",
        "log_prob": "The log density of the observed `query_rows`.",
    },
    owned=frozenset(RETURN_MODES),
    vocabulary="return mode output",
)
_OUTPUT_SCHEMAS: Final[Mapping[str, Mapping[str, Any]]] = {
    "mean": _MATRIX,
    "samples": _array_of(_MATRIX),
    "quantiles": _array_of(ref("QuantileLevel")),
    "log_prob": ref("LogProbScores"),
}


def _component_name(return_mode: str) -> str:
    """Return the name ``return_mode``'s schemas share, such as ``LogProb``."""
    return "".join(part.capitalize() for part in return_mode.split("_"))


def _outputs_fields(return_mode: str) -> tuple[WireField, ...]:
    """Return the ``outputs`` fields of a ``return_mode`` response."""
    entries = [
        WireField(
            "query_times",
            _array_of({"type": ["string", "number"]}),
            "The request's `query_times`, one per future position: ISO 8601 "
            "strings for `absolute_datetime`, numbers otherwise.",
        ),
        WireField(
            "requested_columns",
            _array_of(_STRING),
            "The names of the columns the answer covers, in its column order.",
        ),
    ]
    for output in _OUTPUT_FIELDS:
        if output.value == return_mode:
            entries.append(
                WireField(
                    output.value, _OUTPUT_SCHEMAS[output.value], output.description
                )
            )
        else:
            entries.append(
                WireField(
                    output.value,
                    {"type": "null"},
                    f"`null`: filled only by `{output.value}` responses.",
                )
            )
    return _all_required(*entries)


QUANTILE_LEVEL_FIELDS: Final = _all_required(
    WireField("quantile", _NUMBER, "The requested level."),
    WireField(
        "values",
        _MATRIX,
        "Indexed `[position][column]`: the level's quantile of each requested "
        "column at each future position.",
    ),
)
LOG_PROB_FIELDS: Final = _all_required(
    WireField(
        "values",
        _array_of(_NUMBER),
        "The joint log density of the observed row at each future position.",
    ),
    WireField("nll_values", _array_of(_NUMBER), "The negated `values`."),
    WireField("total", _NUMBER, "The sum of `values`."),
    WireField("mean", _NUMBER, "The mean of `values`."),
    WireField("nll_total", _NUMBER, "The negated `total`."),
    WireField("nll_mean", _NUMBER, "The negated `mean`."),
)
PLAUSIBILITY_FIELDS: Final = _all_required(
    WireField(
        "equality_log_density",
        nullable(_NUMBER),
        "The joint log density of every pinned value under the forecast, "
        "summed over positions; `null` without equality conditions.",
    ),
    WireField(
        "region_log_probability",
        nullable(_NUMBER),
        "The log probability of every band under the forecast given the pinned "
        "values, summed over positions; `null` without interval conditions. Its "
        "sum with `equality_log_density` is the plausibility of the whole "
        "condition set; only differences between comparable requests carry "
        "meaning.",
    ),
)
_FORECAST_DIAGNOSTICS = (
    WireField("history_rows", _COUNT, "The number of history rows read."),
    WireField("horizon_count", _COUNT, "The number of future positions answered."),
    WireField(
        "seed", nullable({"type": "integer"}), "The request's `seed`, or `null`."
    ),
)
FORECAST_DIAGNOSTICS_FIELDS: Final = _all_required(*_FORECAST_DIAGNOSTICS)
CONDITION_DIAGNOSTICS_FIELDS: Final = _all_required(
    *_FORECAST_DIAGNOSTICS,
    WireField(
        "condition_draws",
        _COUNT,
        "The draws behind each conditioned position's answer.",
    ),
    WireField(
        "interval_estimator",
        nullable(ref("IntervalEstimator")),
        "The estimate behind the band probability at the position whose "
        "estimate is least supported; `null` unless some position bands more "
        "than one column.",
    ),
)
INTERVAL_ESTIMATOR_FIELDS: Final = _all_required(
    WireField(
        "points",
        _COUNT,
        "Quasi-random points per mixture component behind the estimate.",
    ),
    WireField(
        "effective_sample_size",
        _NUMBER,
        "Kish's effective sample size of the estimate's importance weights; "
        "far below `points`, the estimate is unreliable.",
    ),
)

# A ``condition`` response reports the plausibility of its conditions and
# condition diagnostics; a ``forecast`` response reports neither.
_QUERY_MODE_RULE: Final = {
    "if": {"properties": {"query_mode": {"const": QUERY_MODE_CONDITION}}},
    "then": {
        "properties": {
            "plausibility": ref("Plausibility"),
            "diagnostics": ref("ConditionDiagnostics"),
        }
    },
    "else": {
        "properties": {
            "plausibility": {"type": "null"},
            "diagnostics": ref("ForecastDiagnostics"),
        }
    },
}


def _predict_response_fields(return_mode: str) -> tuple[WireField, ...]:
    """Return the fields of a ``return_mode`` forecast response."""
    return _all_required(
        WireField("schema_version", _SCHEMA_VERSION, "As `/healthz` reports it."),
        WireField("image_version", _STRING, "As `/healthz` reports it."),
        WireField("model_version", _STRING, "As `/healthz` reports it."),
        WireField("checkpoint_version", _STRING, "As `/healthz` reports it."),
        WireField("head", _STRING, "As `/healthz` reports it."),
        WireField(
            "query_mode", enum_schema(QUERY_MODE_VALUES), "The request's query mode."
        ),
        WireField(
            "return_mode",
            {"type": "string", "const": return_mode},
            "The request's return mode, which decides the variant.",
        ),
        WireField(
            "outputs",
            ref(f"{_component_name(return_mode)}Outputs"),
            "The answer.",
        ),
        WireField(
            "plausibility",
            nullable(ref("Plausibility")),
            "How plausible a `condition` query's conditions are under the "
            "forecast; `null` for `forecast` queries.",
        ),
        WireField(
            "diagnostics",
            {
                "oneOf": [
                    ref("ForecastDiagnostics"),
                    ref("ConditionDiagnostics"),
                ]
            },
            "Counts behind the answer.",
        ),
        WireField(
            "errors",
            {"type": "array", "maxItems": 0},
            "Empty in every successful response.",
        ),
    )


ERROR_FIELDS: Final = (
    WireField(
        "code",
        enum_schema(ERROR_CODE_VALUES),
        vocabulary_text(
            "What went wrong, as a value a client can branch on.", ERROR_CODE_VALUES
        ),
        required=True,
    ),
    WireField("message", _STRING, "What went wrong, for people.", required=True),
    WireField(
        "field",
        nullable(_STRING),
        "The request field or header the error is about, such as "
        "`columns[0].modality` or `Content-Length`; `null` when it concerns "
        "none.",
        required=True,
    ),
    *(
        WireField(value.value, _COUNT, value.description)
        for value in ERROR_VALUE_VALUES
    ),
)


def _error_value_rules() -> list[dict[str, Any]]:
    """Return the rules that tie each error value field to the codes carrying it."""
    rules: list[dict[str, Any]] = []
    for value in ERROR_VALUE_VALUES:
        codes = [
            code.value
            for code, names in ERROR_VALUE_FIELDS.items()
            if value.value in names
        ]
        rules.append(
            {
                "if": {"properties": {"code": {"enum": codes}}},
                "then": {"required": [value.value]},
                "else": {"not": {"required": [value.value]}},
            }
        )
    return rules


def _variant(name: str, return_mode: str) -> dict[str, Any]:
    """Return the schema of the ``return_mode`` forecast response."""
    schema = object_schema(
        _predict_response_fields(return_mode),
        description=f"The answer to a `return_mode` `{return_mode}` request.",
        title=name,
    )
    schema["allOf"] = [_QUERY_MODE_RULE]
    return schema


def response_components() -> dict[str, dict[str, Any]]:
    """Return the response schemas of the document's ``components.schemas``."""
    variants = {f"{_component_name(mode)}Response": mode for mode in RETURN_MODES}
    error = object_schema(
        ERROR_FIELDS, description="One reason the service refused or failed a request."
    )
    error["allOf"] = _error_value_rules()
    components: dict[str, dict[str, Any]] = {
        "PredictResponse": {
            "description": "The answer to a prediction request, one variant per "
            "return mode.",
            "oneOf": [ref(name) for name in variants],
            "discriminator": {
                "propertyName": "return_mode",
                "mapping": {mode: ref(name)["$ref"] for name, mode in variants.items()},
            },
        },
    }
    for name, mode in variants.items():
        components[name] = _variant(name, mode)
        components[f"{_component_name(mode)}Outputs"] = object_schema(
            _outputs_fields(mode),
            description=f"The `outputs` of a `{mode}` response.",
        )
    components.update(
        {
            "QuantileLevel": object_schema(
                QUANTILE_LEVEL_FIELDS, description="The quantiles at one level."
            ),
            "LogProbScores": object_schema(
                LOG_PROB_FIELDS,
                description="The log density of the observed rows, per position "
                "and summed.",
            ),
            "Plausibility": object_schema(
                PLAUSIBILITY_FIELDS,
                description="How plausible the conditions are under the forecast.",
            ),
            "ForecastDiagnostics": object_schema(
                FORECAST_DIAGNOSTICS_FIELDS,
                description="The diagnostics of a `forecast` response.",
            ),
            "ConditionDiagnostics": object_schema(
                CONDITION_DIAGNOSTICS_FIELDS,
                description="The diagnostics of a `condition` response.",
            ),
            "IntervalEstimator": object_schema(
                INTERVAL_ESTIMATOR_FIELDS,
                description="The estimate behind a multi-column band probability.",
            ),
            "HealthResponse": object_schema(
                HEALTH_FIELDS,
                description="What the service serves and the limits a request "
                "must keep.",
            ),
            "CapacityEnvelope": object_schema(
                CAPACITY_ENVELOPE_FIELDS,
                description="The request sizes the served model was built for, "
                "so a client can size and split its requests.",
            ),
            "Error": error,
            "ErrorResponse": object_schema(
                _all_required(
                    WireField(
                        "schema_version",
                        _SCHEMA_VERSION,
                        "The version of this contract the service speaks.",
                    ),
                    WireField(
                        "errors",
                        {**_array_of(ref("Error")), "minItems": 1},
                        "Why the request was refused or failed.",
                    ),
                ),
                description="The body of every non-200 response.",
            ),
        }
    )
    return components
