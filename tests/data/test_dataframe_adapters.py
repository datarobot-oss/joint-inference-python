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

"""Tests for converting a history frame into a canonical sample."""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from jointfm_inference.data.adapters import (
    ColumnSpec,
    DataFrameSchema,
    dataframe_to_canonical,
)
from jointfm_inference.data.types import TimeIndexModeLiteral

SECONDS_PER_DAY = 86400.0


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [10.0, 20.0, 30.0, 40.0],
            "t": [0.0, 0.5, 1.0, 1.5],
        }
    )


def _schema(
    *,
    time_column: str | None = "t",
    time_index_mode: TimeIndexModeLiteral = "continuous_float",
) -> DataFrameSchema:
    return DataFrameSchema(
        columns=[ColumnSpec(name="a"), ColumnSpec(name="b", role="target")],
        time_index_mode=time_index_mode,
        time_column=time_column,
    )


class TestColumnSpec:
    def test_defaults_to_an_unbounded_feature(self) -> None:
        spec = ColumnSpec(name="a")

        assert (spec.role, spec.lower_bound, spec.upper_bound) == (
            "feature",
            None,
            None,
        )

    def test_rejects_an_unknown_role(self) -> None:
        with pytest.raises(ValueError, match="role must be one of"):
            ColumnSpec(name="a", role="static")  # ty: ignore[invalid-argument-type]

    def test_rejects_a_nan_bound(self) -> None:
        with pytest.raises(ValueError, match="lower_bound must not be NaN"):
            ColumnSpec(name="a", lower_bound=math.nan)
        with pytest.raises(ValueError, match="upper_bound must not be NaN"):
            ColumnSpec(name="a", upper_bound=math.nan)

    def test_rejects_a_lower_bound_above_the_upper_bound(self) -> None:
        with pytest.raises(ValueError, match="lower_bound must not exceed upper_bound"):
            ColumnSpec(name="a", lower_bound=2.0, upper_bound=1.0)

    def test_accepts_equal_and_one_sided_bounds(self) -> None:
        equal = ColumnSpec(name="a", lower_bound=1.0, upper_bound=1.0)
        lower_only = ColumnSpec(name="b", lower_bound=0.0)
        upper_only = ColumnSpec(name="c", upper_bound=math.inf)

        assert (equal.lower_bound, equal.upper_bound) == (1.0, 1.0)
        assert (lower_only.lower_bound, lower_only.upper_bound) == (0.0, None)
        assert (upper_only.lower_bound, upper_only.upper_bound) == (None, math.inf)


class TestDataFrameSchema:
    def test_rejects_an_unknown_time_index_mode(self) -> None:
        with pytest.raises(ValueError, match="time_index_mode must be one of"):
            DataFrameSchema(columns=[ColumnSpec(name="a")], time_index_mode="weekly")  # ty: ignore[invalid-argument-type]

    @pytest.mark.parametrize("mode", ["continuous_float", "absolute_datetime"])
    def test_requires_a_time_column_outside_ordinal_mode(
        self, mode: TimeIndexModeLiteral
    ) -> None:
        with pytest.raises(ValueError, match="time_column must be provided"):
            DataFrameSchema(columns=[ColumnSpec(name="a")], time_index_mode=mode)

    def test_rejects_empty_columns(self) -> None:
        with pytest.raises(ValueError, match="columns must not be empty"):
            DataFrameSchema(columns=[], time_index_mode="ordinal")

    def test_rejects_duplicate_column_names(self) -> None:
        with pytest.raises(ValueError, match="column names must be unique"):
            DataFrameSchema(
                columns=[ColumnSpec(name="a"), ColumnSpec(name="a")],
                time_index_mode="ordinal",
            )

    def test_rejects_a_time_column_listed_among_the_columns(self) -> None:
        with pytest.raises(ValueError, match="must not also appear"):
            DataFrameSchema(
                columns=[ColumnSpec(name="a"), ColumnSpec(name="t")],
                time_index_mode="continuous_float",
                time_column="t",
            )


class TestDataFrameToCanonical:
    def test_returns_history_rows_only(self) -> None:
        sample = dataframe_to_canonical(_frame(), _schema())

        assert tuple(sample.values.shape) == (4, 2)
        assert sample.values.dtype == torch.float32
        assert sample.time_positions.dtype == torch.float64
        assert sample.query_row_mask.tolist() == [False] * 4
        assert sample.time_index_mode == "continuous_float"
        assert sample.metadata == {"column_names": ["a", "b"]}

    def test_columns_follow_schema_order_and_ignore_extra_frame_columns(self) -> None:
        frame = _frame().assign(extra=[9.0, 9.0, 9.0, 9.0])[["extra", "b", "t", "a"]]
        schema = DataFrameSchema(
            columns=[ColumnSpec(name="b"), ColumnSpec(name="a")],
            time_index_mode="continuous_float",
            time_column="t",
        )

        sample = dataframe_to_canonical(frame, schema)

        assert sample.metadata["column_names"] == ["b", "a"]
        torch.testing.assert_close(
            sample.values[:, 0], torch.tensor([10.0, 20.0, 30.0, 40.0])
        )
        torch.testing.assert_close(
            sample.values[:, 1], torch.tensor([1.0, 2.0, 3.0, 4.0])
        )

    def test_continuous_float_positions_come_from_the_time_column(self) -> None:
        sample = dataframe_to_canonical(_frame(), _schema())

        torch.testing.assert_close(
            sample.time_positions,
            torch.tensor([0.0, 0.5, 1.0, 1.5], dtype=torch.float64),
        )

    def test_ordinal_without_time_column_numbers_rows_from_zero(self) -> None:
        sample = dataframe_to_canonical(
            _frame(), _schema(time_column=None, time_index_mode="ordinal")
        )

        torch.testing.assert_close(
            sample.time_positions, torch.arange(4, dtype=torch.float64)
        )

    def test_ordinal_time_column_keeps_declared_gaps(self) -> None:
        frame = _frame().assign(t=[1, 2, 4, 7])

        sample = dataframe_to_canonical(frame, _schema(time_index_mode="ordinal"))

        assert sample.time_positions.tolist() == [1.0, 2.0, 4.0, 7.0]

    def test_ordinal_time_column_rounds_noise_within_step_tolerance(self) -> None:
        frame = _frame().assign(t=[1.0, 2.0, 4.0 + 1.0e-12, 7.0])

        sample = dataframe_to_canonical(frame, _schema(time_index_mode="ordinal"))

        assert sample.time_positions.tolist() == [1.0, 2.0, 4.0, 7.0]

    def test_ordinal_time_column_rejects_fractional_steps(self) -> None:
        frame = _frame().assign(t=[1.0, 2.0, 4.0 + 5.0e-8, 7.0])

        with pytest.raises(TypeError, match="integer-like"):
            dataframe_to_canonical(frame, _schema(time_index_mode="ordinal"))

    @pytest.mark.parametrize("mode", ["ordinal", "continuous_float"])
    def test_numeric_modes_reject_a_datetime_time_column(
        self, mode: TimeIndexModeLiteral
    ) -> None:
        frame = _frame().assign(
            t=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"])
        )

        with pytest.raises(TypeError, match="must be numeric"):
            dataframe_to_canonical(frame, _schema(time_index_mode=mode))

    def test_absolute_datetime_positions_and_time_index_metadata(self) -> None:
        frame = _frame().assign(
            t=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-04", "2026-01-05"])
        )
        schema = DataFrameSchema(
            columns=[ColumnSpec(name="a"), ColumnSpec(name="b")],
            time_index_mode="absolute_datetime",
            time_column="t",
            time_scale_seconds=SECONDS_PER_DAY,
            timezone="UTC",
        )

        sample = dataframe_to_canonical(frame, schema)

        assert sample.time_positions.tolist() == pytest.approx([-4.0, -3.0, -1.0, 0.0])
        assert sample.metadata["time_index"] == {
            "time_origin_ns": int(pd.Timestamp("2026-01-05").value),
            "time_scale_seconds": SECONDS_PER_DAY,
            "timezone": "UTC",
            "data_timezone": None,
        }

    def test_absolute_datetime_rejects_a_numeric_time_column(self) -> None:
        schema = DataFrameSchema(
            columns=[ColumnSpec(name="a")],
            time_index_mode="absolute_datetime",
            time_column="t",
        )

        with pytest.raises(TypeError, match="datetime dtype"):
            dataframe_to_canonical(_frame(), schema)

    @pytest.mark.parametrize(
        "times",
        [[0.0, 1.0, 0.5, 2.0], [0.0, 1.0, 1.0, 2.0]],
        ids=["decreasing", "repeated"],
    )
    def test_rejects_times_that_are_not_strictly_increasing(
        self, times: list[float]
    ) -> None:
        with pytest.raises(ValueError, match="strictly monotonically increasing"):
            dataframe_to_canonical(_frame().assign(t=times), _schema())

    def test_rejects_repeated_timestamps(self) -> None:
        frame = _frame().assign(
            t=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-02", "2026-01-03"])
        )
        schema = DataFrameSchema(
            columns=[ColumnSpec(name="a")],
            time_index_mode="absolute_datetime",
            time_column="t",
        )

        with pytest.raises(ValueError, match="strictly monotonically increasing"):
            dataframe_to_canonical(frame, schema)

    @pytest.mark.parametrize("bad_value", [np.nan, None, math.inf, -math.inf])
    def test_rejects_a_missing_or_non_finite_value(
        self, bad_value: float | None
    ) -> None:
        frame = _frame().astype({"b": object})
        frame.loc[2, "b"] = bad_value

        with pytest.raises(ValueError, match=r"history column 'b' \(column index 1\)"):
            dataframe_to_canonical(frame, _schema())

    @pytest.mark.parametrize("missing", ["b", "t"])
    def test_rejects_a_frame_missing_a_declared_column(self, missing: str) -> None:
        with pytest.raises(ValueError, match=rf"missing columns: \['{missing}'\]"):
            dataframe_to_canonical(_frame().drop(columns=[missing]), _schema())

    def test_rejects_an_empty_frame(self) -> None:
        with pytest.raises(ValueError, match="at least one row"):
            dataframe_to_canonical(_frame().iloc[:0], _schema())

    def test_single_row_history_is_accepted(self) -> None:
        sample = dataframe_to_canonical(_frame().iloc[:1], _schema())

        assert tuple(sample.values.shape) == (1, 2)
        assert sample.time_positions.tolist() == [0.0]
