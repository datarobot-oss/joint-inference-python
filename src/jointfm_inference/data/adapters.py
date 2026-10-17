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

"""Conversion of a request's history rows into a canonical sample."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from jointfm_inference.data.time_index import (
    absolute_datetime_positions,
    positions_within_relative_step_tolerance,
)
from jointfm_inference.data.types import (
    METADATA_COLUMN_NAMES_KEY,
    METADATA_TIME_INDEX_KEY,
    ROLES,
    TIME_INDEX_MODES,
    CanonicalSample,
    RoleLiteral,
    TimeIndexModeLiteral,
)


@dataclass(frozen=True)
class ColumnSpec:
    """One numeric request column.

    Attributes:
        name: Column name in the request rows.
        role: ``target`` or ``feature``; every column is forecast either way.
        lower_bound: Optional lower bound every returned draw respects.
        upper_bound: Optional upper bound every returned draw respects.
    """

    name: str
    role: RoleLiteral = "feature"
    lower_bound: float | None = None
    upper_bound: float | None = None

    def __post_init__(self) -> None:
        """Refuse an unknown role and inconsistent bounds."""
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {sorted(ROLES)}, got {self.role!r}")
        for bound_name, bound in (
            ("lower_bound", self.lower_bound),
            ("upper_bound", self.upper_bound),
        ):
            if bound is not None and math.isnan(bound):
                raise ValueError(f"column '{self.name}' {bound_name} must not be NaN")
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise ValueError(
                f"column '{self.name}' lower_bound must not exceed upper_bound"
            )


@dataclass(frozen=True)
class DataFrameSchema:
    """How a history frame maps onto a canonical sample.

    Attributes:
        columns: The numeric columns, in canonical order.
        time_index_mode: ``ordinal``, ``continuous_float``, or ``absolute_datetime``.
        time_column: Column holding row times; optional only for ``ordinal``,
            whose rows are then ``0 .. rows-1``. It must not be in ``columns``.
        time_scale_seconds: Seconds per position unit for ``absolute_datetime``.
        timezone: Timezone of naive ``absolute_datetime`` timestamps.
    """

    columns: Sequence[ColumnSpec]
    time_index_mode: TimeIndexModeLiteral
    time_column: str | None = None
    time_scale_seconds: float | None = None
    timezone: str | None = None

    def __post_init__(self) -> None:
        """Refuse an unknown mode, a missing time column, and duplicate names."""
        if self.time_index_mode not in TIME_INDEX_MODES:
            raise ValueError(
                f"time_index_mode must be one of {sorted(TIME_INDEX_MODES)}, "
                f"got {self.time_index_mode!r}"
            )
        if self.time_index_mode != "ordinal" and self.time_column is None:
            raise ValueError(
                f"time_column must be provided for time_index_mode '{self.time_index_mode}'"
            )
        names = [column.name for column in self.columns]
        if not names:
            raise ValueError("DataFrameSchema.columns must not be empty")
        if len(set(names)) != len(names):
            raise ValueError("column names must be unique")
        if self.time_column is not None and self.time_column in names:
            raise ValueError("time_column must not also appear in the columns")


def _time_positions(
    df: pd.DataFrame, schema: DataFrameSchema
) -> tuple[torch.Tensor, dict[str, object] | None]:
    """Return the history positions and, for datetimes, their ``time_index`` block."""
    if schema.time_column is None:
        return torch.arange(len(df), dtype=torch.float64), None
    raw_time = df[schema.time_column]
    if schema.time_index_mode == "absolute_datetime":
        return absolute_datetime_positions(
            raw_time,
            time_scale_seconds=schema.time_scale_seconds,
            timezone=schema.timezone,
        )
    if pd.api.types.is_datetime64_any_dtype(raw_time):
        raise TypeError(
            f"time_column must be numeric when time_index_mode is '{schema.time_index_mode}'"
        )
    positions = torch.tensor(raw_time.to_numpy(dtype=np.float64), dtype=torch.float64)
    if schema.time_index_mode == "ordinal":
        rounded = torch.round(positions)
        if not positions_within_relative_step_tolerance(
            positions, rounded, step_scale=1.0
        ):
            raise TypeError(
                "time_column must use integer-like numeric values when time_index_mode='ordinal'"
            )
        positions = rounded
    return positions, None


def dataframe_to_canonical(
    df: pd.DataFrame, schema: DataFrameSchema
) -> CanonicalSample:
    """Convert a history frame into a canonical sample of history rows only.

    Raises:
        ValueError: when the frame is empty, misses a column, holds a missing
            or non-finite value, or its times are not strictly increasing.
        TypeError: when a time column has the wrong kind of values.
    """
    if len(df) == 0:
        raise ValueError("DataFrame must have at least one row")
    names = [column.name for column in schema.columns]
    expected = set(names) | (
        {schema.time_column} if schema.time_column is not None else set()
    )
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {sorted(missing)}")

    values = torch.tensor(
        df.loc[:, names].to_numpy(dtype=np.float64, na_value=np.nan),
        dtype=torch.float32,
    )
    for index, name in enumerate(names):
        if not bool(torch.isfinite(values[:, index]).all()):
            raise ValueError(
                f"history column '{name}' (column index {index}) holds a missing or "
                "non-finite value; JointFM 1 reads complete numeric histories"
            )
    positions, time_index_metadata = _time_positions(df, schema)
    metadata: dict[str, object] = {METADATA_COLUMN_NAMES_KEY: names}
    if time_index_metadata is not None:
        metadata[METADATA_TIME_INDEX_KEY] = time_index_metadata
    return CanonicalSample(
        values=values,
        time_positions=positions,
        time_index_mode=schema.time_index_mode,
        query_row_mask=torch.zeros(values.shape[0], dtype=torch.bool),
        metadata=metadata,
    )


__all__ = ["ColumnSpec", "DataFrameSchema", "dataframe_to_canonical"]
