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

"""Unit tests for mapping request times onto the JointFM 1 time grid."""

from typing import Any

import pandas as pd
import pytest
import torch

from jointfm_inference.data.adapters import (
    ColumnSpec,
    DataFrameSchema,
    dataframe_to_canonical,
)
from jointfm_inference.data.time_index import build_canonical_query_sample
from jointfm_inference.data.types import CanonicalSample, TimeIndexModeLiteral
from jointfm_inference.model.inference.contract import InferenceRequestError
from jointfm_inference.model.inference.jointfm_1_grid import (
    _ordinal_positions_to_indices,
    _regular_grid_positions_to_indices,
    _validate_grid_index_bound,
    grid_model_inputs,
)
from jointfm_inference.model.inference.time_index import (
    DISCRETE_GRID_TIME_INDEX_ENCODING,
    TimeIndexSupport,
)

CPU = torch.device("cpu")
SECONDS_PER_DAY = 86400.0


def _support(*, max_time_ctx: int = 16, max_time_fut: int = 16) -> TimeIndexSupport:
    return TimeIndexSupport(
        encoding=DISCRETE_GRID_TIME_INDEX_ENCODING,
        supported_request_time_index_modes=(
            "ordinal",
            "continuous_float",
            "absolute_datetime",
        ),
        max_time_ctx=max_time_ctx,
        max_time_fut=max_time_fut,
    )


def _query_sample(
    history_times: list[Any],
    query_times: list[Any],
    *,
    mode: TimeIndexModeLiteral,
) -> CanonicalSample:
    """Return a two-column history with one query row per query time appended."""
    frame = pd.DataFrame(
        {
            "y": [float(i) for i in range(len(history_times))],
            "x": [10.0 * i for i in range(len(history_times))],
            "t": history_times,
        }
    )
    schema = DataFrameSchema(
        columns=[ColumnSpec(name="y", role="target"), ColumnSpec(name="x")],
        time_index_mode=mode,
        time_column="t",
        time_scale_seconds=SECONDS_PER_DAY if mode == "absolute_datetime" else None,
    )
    return build_canonical_query_sample(
        dataframe_to_canonical(frame, schema), query_times=query_times
    )


def _ordinal_sample(n_rows: int, *, query_rows: list[int]) -> CanonicalSample:
    """Return a one-column ordinal sample whose ``query_rows`` are queried."""
    query_row_mask = torch.zeros(n_rows, dtype=torch.bool)
    query_row_mask[query_rows] = True
    return CanonicalSample(
        values=torch.arange(n_rows, dtype=torch.float32).unsqueeze(1),
        time_positions=torch.arange(n_rows, dtype=torch.float64),
        time_index_mode="ordinal",
        query_row_mask=query_row_mask,
        metadata={"column_names": ["y"]},
    )


def test_returns_history_values_grid_indices_and_horizons() -> None:
    sample = _query_sample([0, 1, 2], [3, 4], mode="ordinal")

    past, past_time_indices, horizons = grid_model_inputs(
        sample,
        time_index_support=_support(),
        device=CPU,
    )

    torch.testing.assert_close(
        past, torch.tensor([[[0.0, 0.0], [1.0, 10.0], [2.0, 20.0]]])
    )
    assert past_time_indices.tolist() == [0, 1, 2]
    assert horizons.tolist() == [0, 1]
    assert past_time_indices.dtype == torch.long
    assert horizons.dtype == torch.long


def test_ordinal_gaps_survive_and_indices_start_at_the_first_history_row() -> None:
    sample = _query_sample([5, 7, 8], [12, 14], mode="ordinal")

    _, past_time_indices, horizons = grid_model_inputs(
        sample,
        time_index_support=_support(),
        device=CPU,
    )

    assert past_time_indices.tolist() == [0, 2, 3]
    assert horizons.tolist() == [3, 5]


def test_continuous_float_gaps_map_onto_the_smallest_step() -> None:
    sample = _query_sample([0.0, 0.5, 1.5], [2.0, 3.0], mode="continuous_float")

    _, past_time_indices, horizons = grid_model_inputs(
        sample,
        time_index_support=_support(),
        device=CPU,
    )

    assert past_time_indices.tolist() == [0, 1, 3]
    assert horizons.tolist() == [0, 2]


def test_absolute_datetime_gaps_map_onto_the_smallest_step() -> None:
    history = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-04"])
    sample = _query_sample(
        list(history),
        [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-08")],
        mode="absolute_datetime",
    )

    _, past_time_indices, horizons = grid_model_inputs(
        sample,
        time_index_support=_support(),
        device=CPU,
    )

    assert past_time_indices.tolist() == [0, 1, 3]
    assert horizons.tolist() == [0, 3]


def test_rejects_times_off_one_regular_grid() -> None:
    sample = _query_sample([0.0, 1.0, 2.0], [3.4], mode="continuous_float")

    with pytest.raises(InferenceRequestError, match="regular step grid") as exc_info:
        grid_model_inputs(
            sample,
            time_index_support=_support(),
            device=CPU,
        )
    assert exc_info.value.field == "query_times"


def test_requires_an_observed_history_row() -> None:
    sample = _ordinal_sample(2, query_rows=[0, 1])

    with pytest.raises(
        InferenceRequestError, match="at least one observed row"
    ) as exc_info:
        grid_model_inputs(
            sample,
            time_index_support=_support(),
            device=CPU,
        )
    assert exc_info.value.field == "history_rows"


@pytest.mark.parametrize(
    ("support", "field", "label"),
    [
        (_support(max_time_ctx=3), "history_rows", "observed time indices"),
        (_support(max_time_fut=5), "query_times", "forecast horizon indices"),
    ],
)
def test_rejects_indices_beyond_the_checkpoint_capacity(
    support: TimeIndexSupport, field: str, label: str
) -> None:
    sample = _query_sample([0, 2, 3], [7, 9], mode="ordinal")

    with pytest.raises(
        InferenceRequestError, match=f"cannot represent {label}"
    ) as exc_info:
        grid_model_inputs(
            sample,
            time_index_support=support,
            device=CPU,
        )
    assert exc_info.value.field == field


def test_accepts_indices_at_the_capacity_limit() -> None:
    sample = _query_sample([0, 2, 3], [7, 9], mode="ordinal")

    _, past_time_indices, horizons = grid_model_inputs(
        sample,
        time_index_support=_support(max_time_ctx=4, max_time_fut=6),
        device=CPU,
    )

    assert past_time_indices.tolist() == [0, 2, 3]
    assert horizons.tolist() == [3, 5]


def test_ordinal_positions_must_use_integer_steps() -> None:
    with pytest.raises(InferenceRequestError, match="integer step values"):
        _ordinal_positions_to_indices(torch.tensor([0.0, 0.4], dtype=torch.float64))


def test_regular_grid_single_position_maps_to_zero_index() -> None:
    indices = _regular_grid_positions_to_indices(
        torch.tensor([42.5], dtype=torch.float64), mode_name="continuous_float"
    )

    assert indices.tolist() == [0]
    assert indices.dtype == torch.long


def test_regular_grid_rejects_non_increasing_positions() -> None:
    with pytest.raises(InferenceRequestError, match="strictly increasing"):
        _regular_grid_positions_to_indices(
            torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64),
            mode_name="continuous_float",
        )


def test_regular_grid_tolerance_scales_with_the_step() -> None:
    accepted = torch.tensor([0.0, 1.0e-12, 2.0e-12 + 5.0e-22], dtype=torch.float64)

    assert torch.equal(
        _regular_grid_positions_to_indices(accepted, mode_name="continuous_float"),
        torch.tensor([0, 1, 2]),
    )

    rejected = torch.tensor([0.0, 1.0e-12, 2.0e-12 + 5.0e-21], dtype=torch.float64)
    with pytest.raises(InferenceRequestError, match="regular step grid"):
        _regular_grid_positions_to_indices(rejected, mode_name="continuous_float")


def test_index_bound_reports_the_largest_representable_index() -> None:
    _validate_grid_index_bound(
        indices=torch.tensor([4]),
        max_size=5,
        field="history_rows",
        label="observed time indices",
    )

    with pytest.raises(InferenceRequestError, match="exceeds capacity 4"):
        _validate_grid_index_bound(
            indices=torch.tensor([5]),
            max_size=5,
            field="history_rows",
            label="observed time indices",
        )
