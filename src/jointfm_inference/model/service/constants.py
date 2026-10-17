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

"""Module-level constants for the packaged inference HTTP service."""

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
QUERY_MODE_FORECAST = "forecast"
# Conditional queries against the emitted joint at the future positions the
# conditions cover. A deployment advertises this mode only when the mounted head
# and backend can answer it, so the name is a vocabulary constant rather than a
# served set.
QUERY_MODE_CONDITION = "condition"
CONDITION_KIND_EQUALITY = "equality"
CONDITION_KIND_INTERVAL = "interval"
SUPPORTED_CONDITION_KINDS = (CONDITION_KIND_EQUALITY, CONDITION_KIND_INTERVAL)
PREDICT_REQUEST_TYPE = "predict"
HEALTH_REQUEST_TYPE = "health"
SUPPORTED_REQUEST_TYPES = frozenset({PREDICT_REQUEST_TYPE, HEALTH_REQUEST_TYPE})
# Every top-level key a prediction request may carry. A key outside this set is
# refused rather than ignored, so a misspelled option fails visibly.
REQUEST_FIELDS = frozenset(
    {
        "request_type",
        "schema_version",
        "model_version",
        "query_mode",
        "return_mode",
        "time_index_mode",
        "time_column",
        "time_scale_seconds",
        "use_local_normalized_time",
        "calendar_id",
        "timezone",
        "columns",
        "requested_columns",
        "history_rows",
        "query_times",
        "query_rows",
        "conditions",
        "n_samples",
        "quantiles",
        "seed",
    }
)
