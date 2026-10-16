"""Serialization of request query times into response payloads."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd


def _serialize_query_times(
    query_times: Sequence[Any],
    *,
    time_index_mode: str,
) -> list[Any]:
    """Serialize query times according to the request time-index semantics."""
    if time_index_mode == "absolute_datetime":
        return [pd.Timestamp(value).isoformat() for value in query_times]
    if time_index_mode == "ordinal":
        return [int(value) for value in query_times]
    return [float(value) for value in query_times]
