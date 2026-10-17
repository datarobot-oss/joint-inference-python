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

"""Tests for request-time positions and forecast query rows."""

from typing import Any

import pandas as pd
import pytest
import torch

from jointfm_inference.data.adapters import (
    ColumnSpec,
    DataFrameSchema,
    dataframe_to_canonical,
)
from jointfm_inference.data.time_index import (
    DEFAULT_ABSOLUTE_DATETIME_SCALE_SECONDS,
    absolute_datetime_positions,
    build_canonical_query_sample,
    positions_within_relative_step_tolerance,
)
from jointfm_inference.data.types import INERT_VALUE_FILLER, CanonicalSample

SECONDS_PER_DAY = 86400.0


def _continuous_float_sample() -> CanonicalSample:
    frame = pd.DataFrame(
        {"known": [1.0, 2.0, 3.0], "target": [10.0, 11.0, 12.0], "t": [0.0, 0.75, 2.25]}
    )
    schema = DataFrameSchema(
        columns=[ColumnSpec(name="known"), ColumnSpec(name="target", role="target")],
        time_index_mode="continuous_float",
        time_column="t",
    )
    return dataframe_to_canonical(frame, schema)


def _ordinal_sample() -> CanonicalSample:
    frame = pd.DataFrame({"known": [1.0, 2.0, 3.0], "target": [10.0, 11.0, 12.0]})
    schema = DataFrameSchema(
        columns=[ColumnSpec(name="known"), ColumnSpec(name="target", role="target")],
        time_index_mode="ordinal",
    )
    return dataframe_to_canonical(frame, schema)


def _absolute_datetime_sample(*, tz: str | None = None) -> CanonicalSample:
    timestamps = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
    if tz is not None:
        timestamps = timestamps.tz_localize(tz)
    frame = pd.DataFrame(
        {"known": [1.0, 2.0, 3.0], "target": [10.0, 11.0, 12.0], "ts": timestamps}
    )
    schema = DataFrameSchema(
        columns=[ColumnSpec(name="known"), ColumnSpec(name="target", role="target")],
        time_index_mode="absolute_datetime",
        time_column="ts",
        time_scale_seconds=SECONDS_PER_DAY,
    )
    return dataframe_to_canonical(frame, schema)


class TestPositionsWithinRelativeStepTolerance:
    def test_accepts_residuals_within_the_scaled_tolerance(self) -> None:
        expected = torch.tensor([0.0, 10.0, 20.0], dtype=torch.float64)
        actual = expected + torch.tensor([0.0, 5.0e-9, -5.0e-9], dtype=torch.float64)

        assert positions_within_relative_step_tolerance(
            actual, expected, step_scale=10.0
        )

    def test_rejects_a_residual_beyond_the_scaled_tolerance(self) -> None:
        expected = torch.tensor([0.0, 1.0], dtype=torch.float64)
        actual = torch.tensor([0.0, 1.0 + 1.0e-6], dtype=torch.float64)

        assert not positions_within_relative_step_tolerance(
            actual, expected, step_scale=1.0
        )

    def test_rejects_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError, match="one shape"):
            positions_within_relative_step_tolerance(
                torch.zeros(2), torch.zeros(3), step_scale=1.0
            )

    @pytest.mark.parametrize("step_scale", [0.0, -1.0])
    def test_rejects_non_positive_step_scale(self, step_scale: float) -> None:
        with pytest.raises(ValueError, match="step_scale must be positive"):
            positions_within_relative_step_tolerance(
                torch.zeros(2), torch.zeros(2), step_scale=step_scale
            )


class TestAbsoluteDatetimePositions:
    def test_positions_are_measured_from_the_last_history_timestamp(self) -> None:
        series = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-04"]))

        positions, time_index = absolute_datetime_positions(
            series, time_scale_seconds=SECONDS_PER_DAY, timezone=None
        )

        assert positions.dtype == torch.float64
        assert positions.tolist() == pytest.approx([-3.0, -2.0, 0.0])
        assert time_index == {
            "data_timezone": None,
            "time_origin_ns": int(pd.Timestamp("2026-01-04").value),
            "time_scale_seconds": SECONDS_PER_DAY,
            "timezone": None,
        }

    def test_defaults_to_a_one_year_scale(self) -> None:
        series = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-02"]))

        positions, time_index = absolute_datetime_positions(
            series, time_scale_seconds=None, timezone=None
        )

        assert (
            time_index["time_scale_seconds"] == DEFAULT_ABSOLUTE_DATETIME_SCALE_SECONDS
        )
        assert positions[0].item() == pytest.approx(
            -SECONDS_PER_DAY / DEFAULT_ABSOLUTE_DATETIME_SCALE_SECONDS
        )

    def test_coarse_resolution_timestamps_are_read_at_nanosecond_resolution(
        self,
    ) -> None:
        series = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-02"]).as_unit("s"))

        positions, time_index = absolute_datetime_positions(
            series, time_scale_seconds=SECONDS_PER_DAY, timezone=None
        )

        assert positions.tolist() == pytest.approx([-1.0, 0.0])
        assert time_index["time_origin_ns"] == int(pd.Timestamp("2026-01-02").value)

    def test_records_the_timezone_the_timestamps_carry(self) -> None:
        series = pd.Series(
            pd.to_datetime(["2026-01-01", "2026-01-02"]).tz_localize("Europe/Berlin")
        )

        _, time_index = absolute_datetime_positions(
            series, time_scale_seconds=None, timezone=None
        )

        assert time_index["timezone"] == "Europe/Berlin"

    def test_records_an_explicit_timezone_for_naive_timestamps(self) -> None:
        series = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-02"]))

        _, time_index = absolute_datetime_positions(
            series, time_scale_seconds=None, timezone="UTC"
        )

        assert time_index["timezone"] == "UTC"

    def test_rejects_a_timezone_that_contradicts_the_timestamps(self) -> None:
        series = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True))

        with pytest.raises(ValueError, match="timezone"):
            absolute_datetime_positions(
                series, time_scale_seconds=None, timezone="Europe/Berlin"
            )

    @pytest.mark.parametrize(
        ("series", "message"),
        [
            (pd.Series(pd.to_datetime([])), "at least one timestamp"),
            (pd.Series(pd.to_datetime(["2026-01-01", None])), "NaT"),
        ],
    )
    def test_rejects_empty_or_missing_timestamps(
        self, series: pd.Series, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            absolute_datetime_positions(series, time_scale_seconds=None, timezone=None)

    @pytest.mark.parametrize("scale", [0.0, -1.0])
    def test_rejects_non_positive_scale(self, scale: float) -> None:
        series = pd.Series(pd.to_datetime(["2026-01-01"]))

        with pytest.raises(ValueError, match="time_scale_seconds must be positive"):
            absolute_datetime_positions(series, time_scale_seconds=scale, timezone=None)

    def test_rejects_a_non_datetime_series(self) -> None:
        with pytest.raises(TypeError, match="datetime dtype"):
            absolute_datetime_positions(
                pd.Series([0.0, 1.0]), time_scale_seconds=None, timezone=None
            )


class TestBuildCanonicalQuerySample:
    def test_appends_one_query_row_per_query_time(self) -> None:
        sample = _continuous_float_sample()

        query_sample = build_canonical_query_sample(sample, query_times=[3.0, 4.5])

        assert tuple(query_sample.values.shape) == (5, 2)
        assert query_sample.time_positions.tolist() == pytest.approx(
            [0.0, 0.75, 2.25, 3.0, 4.5]
        )
        assert query_sample.time_positions.dtype == torch.float64
        torch.testing.assert_close(query_sample.values[:3], sample.values)
        assert bool(torch.all(query_sample.values[3:] == INERT_VALUE_FILLER))
        assert query_sample.query_row_mask.tolist() == [
            False,
            False,
            False,
            True,
            True,
        ]
        assert query_sample.time_index_mode == "continuous_float"

    def test_refuses_a_sample_that_already_carries_query_rows(self) -> None:
        query_sample = build_canonical_query_sample(
            _continuous_float_sample(), query_times=[3.0]
        )

        with pytest.raises(ValueError, match="already carries query rows"):
            build_canonical_query_sample(query_sample, query_times=[4.0])

    def test_leaves_the_history_sample_untouched(self) -> None:
        sample = _absolute_datetime_sample()
        values_before = sample.values.clone()

        query_sample = build_canonical_query_sample(
            sample,
            query_times=[pd.Timestamp("2026-01-04")],
        )
        query_sample.metadata["column_names"].append("mutated")
        query_sample.metadata["time_index"]["timezone"] = "mutated"

        assert tuple(sample.values.shape) == (3, 2)
        torch.testing.assert_close(sample.values, values_before)
        assert sample.metadata["column_names"] == ["known", "target"]
        assert sample.metadata["time_index"]["timezone"] is None

    def test_rejects_empty_query_times(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            build_canonical_query_sample(_continuous_float_sample(), query_times=[])

    @pytest.mark.parametrize(
        ("query_times", "message"),
        [
            ([4.0, 4.0], "Duplicate query times"),
            ([4.0, 3.0], "strictly monotonically increasing"),
            ([2.0, 3.0], "strictly after the last observed time"),
        ],
    )
    def test_rejects_invalid_forecast_sequences(
        self, query_times: list[float], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            build_canonical_query_sample(
                _continuous_float_sample(),
                query_times=query_times,
            )

    @pytest.mark.parametrize(
        ("sample_factory", "last_history_time"),
        [
            (_continuous_float_sample, 2.25),
            (_ordinal_sample, 2),
            (_absolute_datetime_sample, pd.Timestamp("2026-01-03")),
        ],
    )
    def test_rejects_a_query_time_equal_to_the_last_history_time(
        self, sample_factory: Any, last_history_time: Any
    ) -> None:
        # Accepting it would forecast the last history row itself.
        with pytest.raises(ValueError, match="strictly after the last observed time"):
            build_canonical_query_sample(
                sample_factory(),
                query_times=[last_history_time],
            )

    def test_ordinal_accepts_integer_valued_query_times(self) -> None:
        query_sample = build_canonical_query_sample(
            _ordinal_sample(), query_times=[3, 5.0]
        )

        assert query_sample.time_positions.tolist() == [0.0, 1.0, 2.0, 3.0, 5.0]

    @pytest.mark.parametrize("query_time", [3.5, True])
    def test_ordinal_rejects_non_integer_query_times(self, query_time: Any) -> None:
        with pytest.raises(ValueError, match="integer step values"):
            build_canonical_query_sample(
                _ordinal_sample(),
                query_times=[query_time],
            )

    @pytest.mark.parametrize(
        "query_time",
        [pd.Timestamp("2026-01-10"), pd.Timestamp("2026-01-10").to_datetime64()],
    )
    def test_continuous_float_rejects_datetime_query_times(
        self, query_time: Any
    ) -> None:
        with pytest.raises(ValueError, match="must be numeric"):
            build_canonical_query_sample(
                _continuous_float_sample(),
                query_times=[query_time],
            )

    def test_absolute_datetime_positions_use_the_history_origin_and_scale(self) -> None:
        sample = _absolute_datetime_sample()

        query_sample = build_canonical_query_sample(
            sample,
            query_times=[pd.Timestamp("2026-01-04"), "2026-01-06T12:00:00"],
        )

        assert query_sample.time_positions.tolist() == pytest.approx(
            [-2.0, -1.0, 0.0, 1.0, 3.5]
        )
        assert query_sample.metadata["time_index"] == sample.metadata["time_index"]

    def test_absolute_datetime_accepts_queries_in_the_history_timezone(self) -> None:
        sample = _absolute_datetime_sample(tz="UTC")

        query_sample = build_canonical_query_sample(
            sample,
            query_times=[pd.Timestamp("2026-01-04", tz="UTC")],
        )

        assert query_sample.time_positions[-1].item() == pytest.approx(1.0)

    def test_absolute_datetime_rejects_queries_in_another_timezone(self) -> None:
        with pytest.raises(ValueError, match="timezone semantics"):
            build_canonical_query_sample(
                _absolute_datetime_sample(),
                query_times=[pd.Timestamp("2026-01-04", tz="UTC")],
            )

    def test_absolute_datetime_rejects_missing_query_timestamps(self) -> None:
        with pytest.raises(ValueError, match="NaT"):
            build_canonical_query_sample(
                _absolute_datetime_sample(),
                query_times=[pd.NaT],
            )


class TestTimezoneAwarenessMustMatch:
    def test_naive_query_against_an_aware_history_is_refused(self) -> None:
        sample = _absolute_datetime_sample(tz="Europe/Berlin")
        with pytest.raises(ValueError, match="timezone"):
            build_canonical_query_sample(
                sample,
                query_times=[pd.Timestamp("2026-01-04")],
            )

    def test_aware_query_against_a_naive_history_with_an_explicit_zone_is_refused(
        self,
    ) -> None:
        frame = pd.DataFrame(
            {"y": [1.0, 2.0], "ts": pd.to_datetime(["2026-01-01", "2026-01-02"])}
        )
        schema = DataFrameSchema(
            columns=[ColumnSpec(name="y")],
            time_index_mode="absolute_datetime",
            time_column="ts",
            time_scale_seconds=SECONDS_PER_DAY,
            timezone="Europe/Berlin",
        )
        sample = dataframe_to_canonical(frame, schema)
        with pytest.raises(ValueError, match="timezone"):
            build_canonical_query_sample(
                sample,
                query_times=[pd.Timestamp("2026-01-03", tz="Europe/Berlin")],
            )

    def test_aware_query_in_the_history_zone_lands_on_the_grid(self) -> None:
        sample = _absolute_datetime_sample(tz="Europe/Berlin")
        query = build_canonical_query_sample(
            sample,
            query_times=[pd.Timestamp("2026-01-04", tz="Europe/Berlin")],
        )
        assert float(query.time_positions[-1]) == pytest.approx(1.0)


@pytest.mark.parametrize("times", [[float("nan")], [0.0, float("inf")]])
def test_non_finite_history_times_are_refused(times: list[float]) -> None:
    frame = pd.DataFrame({"y": [1.0] * len(times), "t": times})
    schema = DataFrameSchema(
        columns=[ColumnSpec(name="y")],
        time_index_mode="continuous_float",
        time_column="t",
    )
    with pytest.raises(ValueError, match="finite"):
        dataframe_to_canonical(frame, schema)


def test_an_unknown_timezone_name_is_refused() -> None:
    with pytest.raises(ValueError, match="timezone"):
        absolute_datetime_positions(
            pd.Series(pd.to_datetime(["2026-01-01", "2026-01-02"])),
            time_scale_seconds=SECONDS_PER_DAY,
            timezone="Not/AZone",
        )
