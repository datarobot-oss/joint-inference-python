"""Request times as numeric positions, and forecast query rows after the history.

``ordinal`` times are integer steps, ``continuous_float`` times are numbers,
and ``absolute_datetime`` times are timestamps measured from the last history
timestamp in units of ``time_scale_seconds``.
"""

from __future__ import annotations

import zoneinfo
from copy import deepcopy
from typing import Any, Final, Sequence

import numpy as np
import pandas as pd
import torch

from jointfm_inference.data.types import (
    INERT_VALUE_FILLER,
    METADATA_TIME_INDEX_KEY,
    CanonicalSample,
)

DEFAULT_ABSOLUTE_DATETIME_SCALE_SECONDS: Final = 365.25 * 24.0 * 60.0 * 60.0
RELATIVE_GRID_STEP_TOLERANCE: Final = 1e-9
NANOSECONDS_PER_SECOND: Final = 1e9

TIME_INDEX_ORIGIN_NS_KEY: Final = "time_origin_ns"
TIME_INDEX_SCALE_SECONDS_KEY: Final = "time_scale_seconds"
TIME_INDEX_TIMEZONE_KEY: Final = "timezone"
# The zone the history timestamps themselves carry (``None`` when naive).
# Positions come from UTC instants of aware timestamps but from wall-clock
# values of naive ones, so query timestamps must match it exactly.
TIME_INDEX_DATA_TIMEZONE_KEY: Final = "data_timezone"


def _data_timezone(dt_index: pd.DatetimeIndex) -> str | None:
    """Return the zone a ``DatetimeIndex`` carries, ``None`` when naive."""
    return None if dt_index.tz is None else str(dt_index.tz)


def positions_within_relative_step_tolerance(
    actual_positions: torch.Tensor,
    expected_positions: torch.Tensor,
    *,
    step_scale: float,
    relative_tolerance: float = RELATIVE_GRID_STEP_TOLERANCE,
) -> bool:
    """Return whether every position residual stays within a relative grid step."""
    if actual_positions.shape != expected_positions.shape:
        raise ValueError("actual_positions and expected_positions must have one shape")
    if step_scale <= 0.0:
        raise ValueError("step_scale must be positive")
    residual = torch.abs(actual_positions - expected_positions)
    return bool(
        torch.all(residual <= float(step_scale) * float(relative_tolerance)).item()
    )


def _to_datetime_index(raw_time: pd.Series | Sequence[Any]) -> pd.DatetimeIndex:
    """Coerce timestamps to a nanosecond-resolution ``DatetimeIndex``.

    Positions are computed from nanosecond integers (``asi8``); pandas may
    otherwise carry second or microsecond resolution, so it is pinned here.
    """
    if isinstance(raw_time, pd.Series):
        if not pd.api.types.is_datetime64_any_dtype(raw_time):
            raise TypeError(
                "absolute_datetime indices must use a pandas datetime dtype"
            )
        dt_index = pd.DatetimeIndex(raw_time)
    else:
        dt_index = pd.DatetimeIndex(pd.to_datetime(list(raw_time), errors="raise"))
    if bool(dt_index.isna().any()):
        raise ValueError("absolute_datetime timestamps must not be missing (NaT)")
    return dt_index.as_unit("ns")


def _resolve_timezone(
    dt_index: pd.DatetimeIndex, *, explicit_timezone: str | None
) -> str | None:
    """Reconcile an explicit timezone with the one the timestamps carry.

    Raises:
        ValueError: when the explicit timezone is not a known zone name or
            differs from the zone the timestamps carry.
    """
    if explicit_timezone is not None:
        try:
            zoneinfo.ZoneInfo(explicit_timezone)
        except (zoneinfo.ZoneInfoNotFoundError, ValueError) as error:
            raise ValueError(f"unknown timezone {explicit_timezone!r}") from error
    index_timezone = None if dt_index.tz is None else str(dt_index.tz)
    if explicit_timezone is not None and index_timezone is not None:
        if explicit_timezone != index_timezone:
            raise ValueError(
                "Provided timezone does not match timezone carried by the datetime index"
            )
    return explicit_timezone or index_timezone


def absolute_datetime_positions(
    raw_time: pd.Series,
    *,
    time_scale_seconds: float | None,
    timezone: str | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return history positions and the ``time_index`` block that produced them.

    The origin is the last history timestamp.

    Raises:
        ValueError: when the series is empty, holds a missing timestamp, the
            scale is not positive, or the timezones disagree.
    """
    dt_index = _to_datetime_index(raw_time)
    if len(dt_index) == 0:
        raise ValueError("absolute_datetime index must contain at least one timestamp")
    scale_seconds = float(
        DEFAULT_ABSOLUTE_DATETIME_SCALE_SECONDS
        if time_scale_seconds is None
        else time_scale_seconds
    )
    if scale_seconds <= 0:
        raise ValueError("time_scale_seconds must be positive")
    timestamps_ns = dt_index.asi8.astype(np.int64)
    origin_ns = int(timestamps_ns[-1])
    positions = torch.tensor(
        (timestamps_ns - origin_ns) / NANOSECONDS_PER_SECOND / scale_seconds,
        dtype=torch.float64,
    )
    metadata = {
        TIME_INDEX_ORIGIN_NS_KEY: origin_ns,
        TIME_INDEX_SCALE_SECONDS_KEY: scale_seconds,
        TIME_INDEX_TIMEZONE_KEY: _resolve_timezone(
            dt_index, explicit_timezone=timezone
        ),
        TIME_INDEX_DATA_TIMEZONE_KEY: _data_timezone(dt_index),
    }
    return positions, metadata


def _require_int_like(value: Any) -> int:
    """Return ``value`` as an integer step, refusing booleans and fractions."""
    if isinstance(value, bool):
        raise ValueError("ordinal query_times must use integer step values")
    int_value = int(value)
    if float(value) != float(int_value):
        raise ValueError("ordinal query_times must use integer step values")
    return int_value


def _query_positions(
    sample: CanonicalSample, query_times: Sequence[Any]
) -> list[float]:
    """Convert request query times to positions in the sample's frame."""
    if sample.time_index_mode == "ordinal":
        return [float(_require_int_like(value)) for value in query_times]
    if sample.time_index_mode == "continuous_float":
        positions = []
        for value in query_times:
            if isinstance(value, (pd.Timestamp, np.datetime64)):
                raise ValueError(
                    "continuous_float query_times must be numeric, not datetime values"
                )
            positions.append(float(value))
        return positions
    time_meta = sample.metadata[METADATA_TIME_INDEX_KEY]
    dt_index = _to_datetime_index(query_times)
    if _data_timezone(dt_index) != time_meta[TIME_INDEX_DATA_TIMEZONE_KEY]:
        raise ValueError(
            "Query timestamps must use the same timezone semantics as the sample: "
            "both naive, or both carrying the same zone"
        )
    offsets_ns = dt_index.asi8.astype(np.int64) - int(
        time_meta[TIME_INDEX_ORIGIN_NS_KEY]
    )
    return [
        float(offset)
        / NANOSECONDS_PER_SECOND
        / float(time_meta[TIME_INDEX_SCALE_SECONDS_KEY])
        for offset in offsets_ns.tolist()
    ]


def build_canonical_query_sample(
    sample: CanonicalSample, *, query_times: Sequence[Any]
) -> CanonicalSample:
    """Append one query row per query time to a history-only ``sample``.

    Raises:
        ValueError: when ``sample`` already carries query rows, or when
            ``query_times`` is empty, holds duplicates, is not strictly
            increasing, or does not start strictly after the last history time.
    """
    if bool(sample.query_row_mask.any()):
        raise ValueError("the sample already carries query rows")
    query_list = list(query_times)
    if not query_list:
        raise ValueError("query_times must not be empty")
    positions = _query_positions(sample, query_list)
    if len(set(positions)) != len(positions):
        raise ValueError("Duplicate query times are not allowed")
    if any(later <= earlier for earlier, later in zip(positions, positions[1:])):
        raise ValueError(
            "Forecast query times must be strictly monotonically increasing"
        )
    if positions[0] <= float(sample.time_positions[-1].item()):
        raise ValueError(
            "Forecast query times must be strictly after the last observed time"
        )
    n_new, n_cols = len(positions), int(sample.values.shape[1])
    return CanonicalSample(
        values=torch.cat(
            [
                sample.values,
                torch.full(
                    (n_new, n_cols), INERT_VALUE_FILLER, dtype=sample.values.dtype
                ),
            ]
        ),
        time_positions=torch.cat(
            [
                sample.time_positions,
                torch.tensor(positions, dtype=sample.time_positions.dtype),
            ]
        ),
        time_index_mode=sample.time_index_mode,
        query_row_mask=torch.cat(
            [sample.query_row_mask, torch.ones(n_new, dtype=torch.bool)]
        ),
        metadata=deepcopy(sample.metadata),
    )


__all__ = [
    "DEFAULT_ABSOLUTE_DATETIME_SCALE_SECONDS",
    "RELATIVE_GRID_STEP_TOLERANCE",
    "absolute_datetime_positions",
    "build_canonical_query_sample",
    "positions_within_relative_step_tolerance",
]
