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

"""The canonical sample one inference request is converted into."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal, TypeAlias

import torch

TimeIndexModeLiteral: TypeAlias = Literal[
    "ordinal", "continuous_float", "absolute_datetime"
]
TIME_INDEX_MODES: Final[frozenset[str]] = frozenset(
    {"ordinal", "continuous_float", "absolute_datetime"}
)
RoleLiteral: TypeAlias = Literal["target", "feature"]
ROLES: Final[frozenset[str]] = frozenset({"target", "feature"})

METADATA_COLUMN_NAMES_KEY: Final = "column_names"
METADATA_TIME_INDEX_KEY: Final = "time_index"

# Finite value written into the cells of query rows, which carry no data.
INERT_VALUE_FILLER: Final = 0.0


@dataclass(frozen=True)
class CanonicalSample:
    """One request's rows over its numeric columns: history rows, then query rows.

    Attributes:
        values: ``(rows, cols)`` float values; finite everywhere, and
            ``INERT_VALUE_FILLER`` in query rows.
        time_positions: ``(rows,)`` float64 positions, strictly increasing.
        time_index_mode: How the positions were derived from request times.
        query_row_mask: ``(rows,)`` bool mask of the rows to forecast; they
            follow every history row.
        metadata: ``column_names`` (one unique name per column) and, for
            ``absolute_datetime``, the ``time_index`` block that maps
            timestamps to positions.
    """

    values: torch.Tensor
    time_positions: torch.Tensor
    time_index_mode: TimeIndexModeLiteral
    query_row_mask: torch.Tensor
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """Fail fast on shapes, dtypes, positions, query rows, and column names."""
        if not torch.is_floating_point(self.values) or self.values.ndim != 2:
            raise ValueError("values must be a floating-point (rows, cols) tensor")
        n_rows, n_cols = self.values.shape
        if n_rows == 0:
            raise ValueError("a sample must carry at least one row")
        if not bool(torch.isfinite(self.values).all()):
            raise ValueError("values must be finite")
        if self.time_positions.shape != (n_rows,):
            raise ValueError("time_positions must have shape (rows,)")
        if not bool(torch.isfinite(self.time_positions).all()):
            raise ValueError("time_positions must be finite")
        if n_rows > 1 and not bool(
            torch.all(self.time_positions[1:] > self.time_positions[:-1])
        ):
            raise ValueError(
                "time_positions must be strictly monotonically increasing for "
                f"time_index_mode='{self.time_index_mode}'"
            )
        if self.time_index_mode not in TIME_INDEX_MODES:
            raise ValueError(
                f"time_index_mode must be one of {sorted(TIME_INDEX_MODES)}, "
                f"got {self.time_index_mode!r}"
            )
        if self.query_row_mask.dtype is not torch.bool or self.query_row_mask.shape != (
            n_rows,
        ):
            raise ValueError("query_row_mask must be a bool tensor of shape (rows,)")
        query_count = int(self.query_row_mask.sum().item())
        if bool(self.query_row_mask[: n_rows - query_count].any()):
            raise ValueError("query rows must follow every history row")
        column_names = self.metadata.get(METADATA_COLUMN_NAMES_KEY)
        if not isinstance(column_names, list) or len(column_names) != n_cols:
            raise ValueError("metadata['column_names'] must list one name per column")
        if len(set(column_names)) != n_cols:
            raise ValueError("metadata['column_names'] must be unique")


__all__ = [
    "INERT_VALUE_FILLER",
    "METADATA_COLUMN_NAMES_KEY",
    "METADATA_TIME_INDEX_KEY",
    "ROLES",
    "TIME_INDEX_MODES",
    "CanonicalSample",
    "RoleLiteral",
    "TimeIndexModeLiteral",
]
