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

"""Module-level constants for the packaged inference HTTP service.

The wire vocabularies (request fields, modes, condition kinds) live in
``wire_schema``, which the OpenAPI document is generated from.
"""

from __future__ import annotations

DEFAULT_SCHEMA_VERSION = "v6"
DEFAULT_BIND_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
DEFAULT_DEVICE = "cpu"
DEFAULT_MAX_SAMPLE_COUNT = 1000
# Maximum prediction requests permitted in-flight on one container at the same
# time. The packaged service serializes beyond this cap so per-request GPU
# memory (most notably the per-draw Cholesky expansion of the Student-t head's
# ``sample`` path) cannot stack across threads and OOM the device.
# The default of one matches the safe budget on a single small GPU; deployments
# on larger devices should raise it via ``JOINTFM_INFERENCE_MAX_CONCURRENT_REQUESTS`` to
# the number of largest requests whose peak memory fits the device at once.
DEFAULT_MAX_CONCURRENT_REQUESTS = 1
# The default ``max_request_bytes`` (``default_max_request_bytes`` in
# ``config.py``) admits a request at every limit of the capacity envelope.
# A cell is one value keyed by its column name in a ``history_rows`` or
# ``query_rows`` mapping, or one ``query_times`` entry. Its budget covers the
# longest value, an ISO 8601 timestamp with nanoseconds and a UTC offset
# (37 bytes quoted; a full-precision float64 takes at most 24), a column name
# of up to ``REQUEST_COLUMN_NAME_BYTES``, and the quotes and the ``": "`` and
# ``", "`` separators the standard library's ``json.dumps`` writes.
REQUEST_COLUMN_NAME_BYTES = 64
REQUEST_CELL_BYTES = 128
# Each row holds one value per column plus its time value, and each query row
# also has its ``query_times`` entry.
REQUEST_TIME_CELLS_PER_ROW = 2
# One interval condition of its own on one column at one query time, the
# largest a column and position can carry: the column name, the kind, two
# full-precision bounds, and a one-element ``query_time_indices``.
REQUEST_CONDITION_BYTES = 256
# One column descriptor with a long name, its role, and both bounds, plus the
# column's ``requested_columns`` entry.
REQUEST_COLUMN_BYTES = 4096
# Every other top-level field: versions, modes, quantiles, the seed.
REQUEST_BASE_BYTES = 65536
# Seconds a connection may go without sending or accepting a byte before the
# handler drops it, so a stalled client cannot hold a server thread.
REQUEST_TIMEOUT_SECONDS = 60.0
# Response bytes per socket send: ``sendall`` bounds a whole call's duration
# by the timeout, so chunking keeps that timeout a stall bound instead of a
# cap on how long a large response may take to download.
RESPONSE_WRITE_CHUNK_BYTES = 1 << 20
