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

"""Type validation helpers for JSON request payloads.

Each helper raises ``ServiceError`` with the offending field name when the
input does not match the contract documented in the public API schema.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from http import HTTPStatus
from typing import Any

from .errors import ServiceError


def _required_env(name: str) -> str:
    """Return one required environment variable or raise a startup error."""
    value = os.getenv(name)
    if value is None or value == "":
        raise ServiceError(
            "VALIDATION_ERROR",
            f"Required environment variable {name} is not set",
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
            field=name,
        )
    return value


def _optional_env(name: str) -> str | None:
    """Return one optional environment variable, or ``None`` when unset or empty."""
    value = os.getenv(name)
    if value is None or value == "":
        return None
    return value


def _positive_int_from_env(name: str, default: int) -> int:
    """Parse one positive integer environment variable with a default fallback."""
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ServiceError(
            "VALIDATION_ERROR",
            f"Environment variable {name} must be an integer, got {raw_value!r}",
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
            field=name,
        ) from error
    if value <= 0:
        raise ServiceError(
            "VALIDATION_ERROR",
            f"Environment variable {name} must be positive, got {value}",
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
            field=name,
        )
    return value


def _choice_from_env(name: str, default: str, choices: Sequence[str]) -> str:
    """Return one environment variable restricted to ``choices``, or its default."""
    value = os.getenv(name, default)
    if value not in choices:
        raise ServiceError(
            "VALIDATION_ERROR",
            f"Environment variable {name} must be one of {tuple(choices)}, "
            f"got {value!r}",
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
            field=name,
        )
    return value


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    """Validate that one request field is represented as a JSON object."""
    if not isinstance(value, Mapping):
        raise ServiceError(
            "VALIDATION_ERROR",
            f"{field} must be a JSON object",
            status=HTTPStatus.BAD_REQUEST,
            field=field,
        )
    return value


def _require_sequence(value: Any, *, field: str) -> Sequence[Any]:
    """Validate that one request field is a non-empty JSON array."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ServiceError(
            "VALIDATION_ERROR",
            f"{field} must be a JSON array",
            status=HTTPStatus.BAD_REQUEST,
            field=field,
        )
    if len(value) == 0:
        raise ServiceError(
            "VALIDATION_ERROR",
            f"{field} must not be empty",
            status=HTTPStatus.BAD_REQUEST,
            field=field,
        )
    return value


def _require_string(value: Any, *, field: str) -> str:
    """Validate that one request field is a non-empty string."""
    if not isinstance(value, str) or value == "":
        raise ServiceError(
            "VALIDATION_ERROR",
            f"{field} must be a non-empty string",
            status=HTTPStatus.BAD_REQUEST,
            field=field,
        )
    return value


def _optional_string(value: Any, field: str, default: str | None = None) -> str | None:
    """Return an optional string field or the provided default value."""
    if value is None:
        return default
    return _require_string(value, field=field)


def _optional_bool(value: Any, *, field: str, default: bool) -> bool:
    """Return an optional boolean field or the provided default value."""
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ServiceError(
            "VALIDATION_ERROR",
            f"{field} must be a boolean",
            status=HTTPStatus.BAD_REQUEST,
            field=field,
        )
    return value


def _optional_int(value: Any, *, field: str) -> int | None:
    """Return an optional integer field when one was supplied in the request."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ServiceError(
            "VALIDATION_ERROR",
            f"{field} must be an integer",
            status=HTTPStatus.BAD_REQUEST,
            field=field,
        )
    return value


def _optional_float(value: Any, *, field: str) -> float | None:
    """Return an optional numeric field normalized to a float."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ServiceError(
            "VALIDATION_ERROR",
            f"{field} must be numeric",
            status=HTTPStatus.BAD_REQUEST,
            field=field,
        )
    return float(value)


def _require_quantiles(value: Any) -> list[float]:
    """Parse and validate the requested quantile levels from one payload field."""
    quantile_values = _require_sequence(value, field="quantiles")
    parsed: list[float] = []
    for idx, quantile in enumerate(quantile_values):
        if isinstance(quantile, bool) or not isinstance(quantile, (int, float)):
            raise ServiceError(
                "VALIDATION_ERROR",
                f"quantiles[{idx}] must be numeric",
                status=HTTPStatus.BAD_REQUEST,
                field="quantiles",
            )
        parsed_quantile = float(quantile)
        if parsed_quantile <= 0.0 or parsed_quantile >= 1.0:
            raise ServiceError(
                "VALIDATION_ERROR",
                f"quantiles[{idx}] must lie strictly between 0 and 1",
                status=HTTPStatus.BAD_REQUEST,
                field="quantiles",
            )
        parsed.append(parsed_quantile)
    return parsed
