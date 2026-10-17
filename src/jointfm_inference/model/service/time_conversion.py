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
