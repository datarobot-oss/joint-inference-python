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

"""Pandas DataFrame builders for inference request rows."""

from __future__ import annotations

from collections.abc import Sequence
from http import HTTPStatus
from typing import Any

import numpy as np
import pandas as pd
import torch

from jointfm_inference.data.adapters import ColumnSpec, DataFrameSchema

from .errors import ServiceError
from .time_conversion import _serialize_query_times
from .validation import _require_mapping, _require_sequence


def _history_frame(
    history_rows: Sequence[Any],
    *,
    column_specs: Sequence[ColumnSpec],
    time_index_mode: str,
    time_column: str | None,
) -> pd.DataFrame:
    """Build the validated history frame expected by canonical sample conversion."""
    return _rows_frame(
        history_rows,
        field="history_rows",
        column_specs=column_specs,
        time_index_mode=time_index_mode,
        time_column=time_column,
    )


def _query_rows_frame(
    query_rows: Any,
    *,
    column_specs: Sequence[ColumnSpec],
    time_index_mode: str,
    time_column: str | None,
    query_times: Sequence[Any],
) -> pd.DataFrame:
    """Build and align the observed rows used for log-probability scoring."""
    raw_query_rows = _require_sequence(query_rows, field="query_rows")
    if len(raw_query_rows) != len(query_times):
        raise ServiceError(
            "VALIDATION_ERROR",
            "query_rows length must match query_times length",
            status=HTTPStatus.BAD_REQUEST,
            field="query_rows",
        )

    frame = _rows_frame(
        raw_query_rows,
        field="query_rows",
        column_specs=column_specs,
        time_index_mode=time_index_mode,
        time_column=time_column,
    )
    if time_column is not None:
        observed_times = _serialize_query_times(
            frame[time_column].tolist(),
            time_index_mode=time_index_mode,
        )
        expected_times = _serialize_query_times(
            query_times,
            time_index_mode=time_index_mode,
        )
        if observed_times != expected_times:
            raise ServiceError(
                "VALIDATION_ERROR",
                "query_rows time values must align with query_times in order",
                status=HTTPStatus.BAD_REQUEST,
                field="query_rows",
            )
    return frame


def _rows_frame(
    rows: Sequence[Any],
    *,
    field: str,
    column_specs: Sequence[ColumnSpec],
    time_index_mode: str,
    time_column: str | None,
) -> pd.DataFrame:
    """Build one validated row frame for request payload rows."""
    normalized_rows: list[dict[str, Any]] = []
    for idx, history_row in enumerate(rows):
        normalized_rows.append(
            dict(_require_mapping(history_row, field=f"{field}[{idx}]"))
        )
    frame = pd.DataFrame.from_records(normalized_rows)

    declared_columns = [column.name for column in column_specs]
    expected_columns = set(declared_columns)
    if time_column is not None and time_column in expected_columns:
        raise ServiceError(
            "VALIDATION_ERROR",
            f"time_column {time_column!r} must not also appear in the columns list",
            status=HTTPStatus.BAD_REQUEST,
            field="time_column",
        )
    missing_columns = sorted(expected_columns.difference(frame.columns))
    if missing_columns:
        raise ServiceError(
            "VALIDATION_ERROR",
            f"{field} are missing declared columns: {missing_columns}",
            status=HTTPStatus.BAD_REQUEST,
            field=field,
        )
    if time_column is not None and time_column not in frame.columns:
        raise ServiceError(
            "VALIDATION_ERROR",
            f"{field} are missing time_column {time_column!r}",
            status=HTTPStatus.BAD_REQUEST,
            field=field,
        )

    ordered_columns = declared_columns
    if time_column is not None:
        ordered_columns = [time_column, *ordered_columns]
    frame = frame.loc[:, ordered_columns].copy()

    if time_index_mode == "absolute_datetime":
        assert time_column is not None
        try:
            frame[time_column] = pd.to_datetime(
                frame[time_column], format="ISO8601", errors="raise"
            )
        except (TypeError, ValueError) as error:
            raise ServiceError(
                "VALIDATION_ERROR",
                f"{field} {time_column!r} must hold ISO 8601 timestamps: {error}",
                status=HTTPStatus.BAD_REQUEST,
                field=field,
            ) from error

    return frame


def _log_prob_values_tensor(
    *,
    query_frame: pd.DataFrame,
    schema: DataFrameSchema,
    requested_column_names: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    """Convert observed query rows into the full-width tensor a ``log_prob`` scores.

    The tensor carries every declared column of every query row: the forecast
    path scores the whole joint, and the conditional scorer needs the pinned
    and bounded columns' values too, to refuse a row that contradicts the
    request before scoring the rest.

    ``requested_columns`` stays honest with that score by listing every
    declared column in declared order, so a partial or reordered projection is
    refused. The rule is the same under a ``condition`` block, whose columns
    are declared columns like any other — and it is what omitting the field
    already resolves to, so a scoring request needs no projection at all.

    Raises:
        ServiceError: ``VALIDATION_ERROR`` on ``requested_columns`` for any
            projection other than every declared column in declared order, and
            on ``query_rows`` for a missing, non-finite, or unconvertible value.
    """
    column_names = [column.name for column in schema.columns]
    if tuple(requested_column_names) != tuple(column_names):
        raise ServiceError(
            "VALIDATION_ERROR",
            "log_prob return_mode requires requested_columns to include every "
            f"declared column in declared order; the declared columns are "
            f"{column_names}",
            status=HTTPStatus.BAD_REQUEST,
            field="requested_columns",
        )

    if bool(query_frame.loc[:, column_names].isna().any().any()):
        raise ServiceError(
            "VALIDATION_ERROR",
            "query_rows must provide values for every declared column "
            "when return_mode='log_prob'",
            status=HTTPStatus.BAD_REQUEST,
            field="query_rows",
        )
    try:
        values = torch.tensor(
            np.ascontiguousarray(
                query_frame.loc[:, column_names].to_numpy(dtype=np.float64)
            ),
            dtype=torch.float32,
        )
    except (TypeError, ValueError) as error:
        raise ServiceError(
            "VALIDATION_ERROR",
            f"query_rows values must be numeric: {error}",
            status=HTTPStatus.BAD_REQUEST,
            field="query_rows",
        ) from error
    if not bool(torch.isfinite(values).all().item()):
        raise ServiceError(
            "VALIDATION_ERROR",
            "query_rows must provide finite values for every declared column "
            "when return_mode='log_prob'",
            status=HTTPStatus.BAD_REQUEST,
            field="query_rows",
        )
    return values.unsqueeze(0).to(device=device)
