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

"""Unit tests for request-row DataFrame builders and log-prob value shaping."""

import math
from typing import Any

import pandas as pd
import pytest
import torch

from jointfm_inference.data.adapters import ColumnSpec, DataFrameSchema
from jointfm_inference.model.service import ServiceError
from jointfm_inference.model.service.data_frames import (
    _history_frame,
    _log_prob_values_tensor,
    _query_rows_frame,
)

CPU = torch.device("cpu")


def _column_specs() -> list[ColumnSpec]:
    return [ColumnSpec(name="y", role="target"), ColumnSpec(name="z", role="target")]


def _schema() -> DataFrameSchema:
    return DataFrameSchema(columns=_column_specs(), time_index_mode="ordinal")


def _ordinal_query_frame(rows: Any, query_times: list[Any]) -> pd.DataFrame:
    return _query_rows_frame(
        rows,
        column_specs=_column_specs(),
        time_index_mode="ordinal",
        time_column="t",
        query_times=query_times,
    )


class TestHistoryFrame:
    def test_keeps_the_time_column_first_and_drops_undeclared_keys(self) -> None:
        frame = _history_frame(
            [
                {"z": 2.0, "extra": "x", "t": 0, "y": 1.0},
                {"z": 4.0, "extra": "x", "t": 1, "y": 3.0},
            ],
            column_specs=_column_specs(),
            time_index_mode="ordinal",
            time_column="t",
        )

        assert list(frame.columns) == ["t", "y", "z"]
        assert frame["y"].tolist() == [1.0, 3.0]

    def test_parses_mixed_iso_8601_spellings_as_one_time_column(self) -> None:
        frame = _history_frame(
            [
                {"t": "2026-01-01", "y": 1.0, "z": 2.0},
                {"t": "2026-01-02T06:00:00", "y": 3.0, "z": 4.0},
            ],
            column_specs=_column_specs(),
            time_index_mode="absolute_datetime",
            time_column="t",
        )

        assert pd.api.types.is_datetime64_any_dtype(frame["t"])
        assert frame["t"].tolist() == [
            pd.Timestamp("2026-01-01"),
            pd.Timestamp("2026-01-02T06:00:00"),
        ]

    def test_an_unparsable_timestamp_is_a_validation_error(self) -> None:
        with pytest.raises(ServiceError) as raised:
            _history_frame(
                [{"t": "not a time", "y": 1.0, "z": 2.0}],
                column_specs=_column_specs(),
                time_index_mode="absolute_datetime",
                time_column="t",
            )
        assert raised.value.field == "history_rows"

    def test_requires_the_declared_time_column_in_rows(self) -> None:
        with pytest.raises(ServiceError, match="missing time_column 't'") as exc_info:
            _history_frame(
                [{"y": 1.0, "z": 2.0}],
                column_specs=_column_specs(),
                time_index_mode="ordinal",
                time_column="t",
            )
        assert exc_info.value.field == "history_rows"

    def test_requires_every_declared_column_in_rows(self) -> None:
        with pytest.raises(ServiceError, match=r"missing declared columns: \['z'\]"):
            _history_frame(
                [{"y": 1.0}],
                column_specs=_column_specs(),
                time_index_mode="ordinal",
                time_column=None,
            )

    def test_rejects_a_time_column_listed_among_the_columns(self) -> None:
        with pytest.raises(ServiceError, match="must not also appear") as exc_info:
            _history_frame(
                [{"y": 1.0, "z": 2.0}],
                column_specs=_column_specs(),
                time_index_mode="ordinal",
                time_column="y",
            )
        assert exc_info.value.field == "time_column"

    def test_rejects_a_row_that_is_not_an_object(self) -> None:
        with pytest.raises(
            ServiceError, match=r"history_rows\[1\] must be a JSON object"
        ):
            _history_frame(
                [{"y": 1.0, "z": 2.0}, [3.0, 4.0]],
                column_specs=_column_specs(),
                time_index_mode="ordinal",
                time_column=None,
            )


class TestQueryRowsFrame:
    def test_orders_the_time_column_first(self) -> None:
        frame = _ordinal_query_frame(
            [{"z": 2.0, "y": 1.0, "t": 4}, {"z": 4.0, "y": 3.0, "t": 5}], [4, 5]
        )

        assert list(frame.columns) == ["t", "y", "z"]
        assert frame["y"].tolist() == [1.0, 3.0]

    @pytest.mark.parametrize(
        ("query_rows", "message"),
        [(None, "must be a JSON array"), ([], "must not be empty")],
    )
    def test_requires_a_non_empty_array(self, query_rows: Any, message: str) -> None:
        with pytest.raises(ServiceError, match=message) as exc_info:
            _ordinal_query_frame(query_rows, [4])
        assert exc_info.value.field == "query_rows"

    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(ServiceError, match="must match query_times length"):
            _ordinal_query_frame([{"t": 4, "y": 1.0, "z": 2.0}], [4, 5])

    def test_rejects_misaligned_times(self) -> None:
        with pytest.raises(ServiceError, match="align with query_times in order"):
            _ordinal_query_frame(
                [{"t": 5, "y": 1.0, "z": 2.0}, {"t": 4, "y": 3.0, "z": 4.0}], [4, 5]
            )

    def test_aligns_absolute_datetime_times_across_spellings(self) -> None:
        frame = _query_rows_frame(
            [{"t": "2026-01-03T00:00:00", "y": 1.0, "z": 2.0}],
            column_specs=_column_specs(),
            time_index_mode="absolute_datetime",
            time_column="t",
            query_times=["2026-01-03"],
        )

        assert frame["t"].tolist() == [pd.Timestamp("2026-01-03")]

    def test_skips_time_alignment_without_a_time_column(self) -> None:
        frame = _query_rows_frame(
            [{"y": 1.0, "z": 2.0}],
            column_specs=_column_specs(),
            time_index_mode="ordinal",
            time_column=None,
            query_times=[7],
        )

        assert list(frame.columns) == ["y", "z"]


class TestLogProbValuesTensor:
    @pytest.mark.parametrize(
        "requested", [["y"], ["z", "y"]], ids=["partial", "reordered"]
    )
    def test_requires_every_declared_column_in_declared_order(
        self, requested: list[str]
    ) -> None:
        with pytest.raises(
            ServiceError, match=r"every\s+declared column in declared order"
        ) as exc_info:
            _log_prob_values_tensor(
                query_frame=pd.DataFrame({"y": [3.0], "z": [4.0]}),
                schema=_schema(),
                requested_column_names=requested,
                device=CPU,
            )
        assert exc_info.value.field == "requested_columns"

    @pytest.mark.parametrize(
        ("bad_value", "message"),
        [
            (math.nan, "values for every declared column"),
            (math.inf, "finite values for every declared column"),
            ("abc", "values must be numeric"),
        ],
    )
    def test_rejects_unusable_query_values(self, bad_value: Any, message: str) -> None:
        with pytest.raises(ServiceError, match=message) as exc_info:
            _log_prob_values_tensor(
                query_frame=pd.DataFrame({"y": [bad_value], "z": [4.0]}),
                schema=_schema(),
                requested_column_names=["y", "z"],
                device=CPU,
            )
        assert exc_info.value.field == "query_rows"

    def test_returns_batched_query_rows_in_declared_order(self) -> None:
        query_frame = _ordinal_query_frame(
            [{"z": 30.0, "y": 3.0, "t": 3}, {"z": 40.0, "y": 4.0, "t": 4}], [3, 4]
        )

        values = _log_prob_values_tensor(
            query_frame=query_frame,
            schema=_schema(),
            requested_column_names=["y", "z"],
            device=CPU,
        )

        assert values.dtype == torch.float32
        torch.testing.assert_close(values, torch.tensor([[[3.0, 30.0], [4.0, 40.0]]]))

    def test_follows_declared_order_whatever_the_frame_order(self) -> None:
        values = _log_prob_values_tensor(
            query_frame=pd.DataFrame({"t": [3], "z": [30.0], "y": [3.0]}),
            schema=_schema(),
            requested_column_names=["y", "z"],
            device=CPU,
        )
        assert values.tolist() == [[[3.0, 30.0]]]
