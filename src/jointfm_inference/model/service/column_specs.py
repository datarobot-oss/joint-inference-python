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

"""Column-spec parsing and selection helpers for inference requests."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from http import HTTPStatus
from typing import Any, Final, cast

from jointfm_inference.data.adapters import ColumnSpec
from jointfm_inference.data.types import ROLES, RoleLiteral

from .errors import ServiceError
from .validation import (
    _optional_float,
    _optional_string,
    _require_mapping,
    _require_sequence,
    _require_string,
)


# Every key a column descriptor may carry. A key outside this set is refused:
# reading only known keys would drop an unknown one, such as a misspelled
# option, without telling the caller.
COLUMN_DESCRIPTOR_FIELDS: frozenset[str] = frozenset(
    {
        "name",
        "modality",
        "role",
        "nullable",
        "vocabulary_size",
        "level_count",
        "mapping",
        "lower_bound",
        "upper_bound",
        "time_value_kind",
        "time_value_scale_seconds",
        "time_value_use_local_normalized_time",
        "time_value_calendar_id",
        "time_value_timezone",
        "traits",
    }
)

SERVED_MODALITY: Final = "numeric"
DEFAULT_CALENDAR_ID: Final = "pandas-default"

# Descriptor fields the service accepts only at their default value, with the
# defaults that pass. A missing key always passes.
_DEFAULT_ONLY_FIELDS: Final[Mapping[str, Collection[Any]]] = {
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


def _refuse(message: str, *, field: str) -> ServiceError:
    """Return the ``VALIDATION_ERROR`` for one refused descriptor field."""
    return ServiceError(
        "VALIDATION_ERROR", message, status=HTTPStatus.BAD_REQUEST, field=field
    )


def _parse_column_specs(raw_columns: Sequence[Any]) -> list[ColumnSpec]:
    """Parse request column descriptors into ``ColumnSpec`` objects.

    The service serves numeric columns: ``modality`` must be ``numeric``,
    ``role`` ``target`` or ``feature``, and every field of
    ``_DEFAULT_ONLY_FIELDS`` absent or at its default.

    Raises:
        ServiceError: ``VALIDATION_ERROR`` for an unknown key, a missing name
            or modality, and any refused value.
    """
    column_specs: list[ColumnSpec] = []
    for idx, raw_column in enumerate(raw_columns):
        field = f"columns[{idx}]"
        column = _require_mapping(raw_column, field=field)
        unknown_fields = sorted(set(map(str, column)) - COLUMN_DESCRIPTOR_FIELDS)
        if unknown_fields:
            raise _refuse(
                f"{field} has unknown fields {unknown_fields}; "
                f"expected a subset of {sorted(COLUMN_DESCRIPTOR_FIELDS)}",
                field=field,
            )
        name = _require_string(column.get("name"), field=f"{field}.name")
        modality = _require_string(column.get("modality"), field=f"{field}.modality")
        if modality != SERVED_MODALITY:
            raise _refuse(
                f"{field}.modality {modality!r} is not served; this service reads "
                f"{SERVED_MODALITY!r} columns only",
                field=f"{field}.modality",
            )
        role = _optional_string(column.get("role"), field=f"{field}.role") or "feature"
        if role not in ROLES:
            raise _refuse(
                f"{field}.role {role!r} is not served; expected one of {sorted(ROLES)}",
                field=f"{field}.role",
            )
        for key, defaults in _DEFAULT_ONLY_FIELDS.items():
            value = column.get(key)
            # A type match keeps 0 from passing as False.
            if value is not None and not any(
                type(value) is type(default) and value == default
                for default in defaults
            ):
                raise _refuse(
                    f"{field}.{key} is not served; omit it"
                    + (f" or pass {list(defaults)[0]!r}" if defaults else ""),
                    field=f"{field}.{key}",
                )
        try:
            column_specs.append(
                ColumnSpec(
                    name=name,
                    role=cast(RoleLiteral, role),
                    lower_bound=_optional_float(
                        column.get("lower_bound"), field=f"{field}.lower_bound"
                    ),
                    upper_bound=_optional_float(
                        column.get("upper_bound"), field=f"{field}.upper_bound"
                    ),
                )
            )
        except ValueError as error:
            raise _refuse(str(error), field=field) from error
    return column_specs


def _resolve_requested_columns(
    column_specs: Sequence[ColumnSpec],
    raw_requested_columns: Any,
) -> list[str]:
    """Resolve requested output columns by validated name or integer index.

    The projection is role-agnostic and condition-agnostic: any declared column
    may be named, in any order, whatever its role and whatever a ``condition``
    block says about it. Omitting the field returns every declared column in
    declared order, so a caller who states nothing gets the whole position and
    two requests that declare the same columns answer in the same layout
    however they differ otherwise.
    """
    column_names = [column.name for column in column_specs]
    name_to_index = {name: idx for idx, name in enumerate(column_names)}

    if raw_requested_columns is None:
        return column_names

    requested_columns = _require_sequence(
        raw_requested_columns, field="requested_columns"
    )
    resolved_names: list[str] = []
    for idx, raw_requested in enumerate(requested_columns):
        if isinstance(raw_requested, str):
            if raw_requested not in name_to_index:
                raise ServiceError(
                    "VALIDATION_ERROR",
                    f"requested_columns[{idx}] references unknown column {raw_requested!r}",
                    status=HTTPStatus.BAD_REQUEST,
                    field="requested_columns",
                )
            resolved_names.append(raw_requested)
            continue
        if isinstance(raw_requested, bool) or not isinstance(raw_requested, int):
            raise ServiceError(
                "VALIDATION_ERROR",
                f"requested_columns[{idx}] must be a string name or integer column index",
                status=HTTPStatus.BAD_REQUEST,
                field="requested_columns",
            )
        if raw_requested < 0 or raw_requested >= len(column_names):
            raise ServiceError(
                "VALIDATION_ERROR",
                f"requested_columns[{idx}] index {raw_requested} is out of bounds for {len(column_names)} columns",
                status=HTTPStatus.BAD_REQUEST,
                field="requested_columns",
            )
        resolved_names.append(column_names[raw_requested])

    if len(set(resolved_names)) != len(resolved_names):
        raise ServiceError(
            "VALIDATION_ERROR",
            "requested_columns must not contain duplicates",
            status=HTTPStatus.BAD_REQUEST,
            field="requested_columns",
        )
    return resolved_names
