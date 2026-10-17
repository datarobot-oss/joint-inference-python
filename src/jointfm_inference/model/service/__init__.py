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

"""HTTP service over one loaded JointFM checkpoint.

- ``http_service``: ``JointFMHTTPService``, request validation and responses.
- ``http_handler`` and ``server``: the HTTP transport and process entry points.
- ``config``: ``InferenceServiceConfig``, read from the environment.
- ``errors``: ``ServiceError`` and its error codes.
"""

from .config import InferenceServiceConfig
from .constants import (
    DEFAULT_BIND_HOST,
    DEFAULT_DEVICE,
    DEFAULT_MAX_CONCURRENT_REQUESTS,
    DEFAULT_MAX_SAMPLE_COUNT,
    DEFAULT_PORT,
    DEFAULT_SCHEMA_VERSION,
    QUERY_MODE_CONDITION,
    QUERY_MODE_FORECAST,
    SUPPORTED_CONDITION_KINDS,
)
from .errors import ServiceError, ServiceErrorCode
from .http_handler import JointFMRequestHandler
from .http_service import JointFMHTTPService
from .serialization import encode_response
from .server import create_service_from_env, main, serve

__all__ = [
    "DEFAULT_BIND_HOST",
    "DEFAULT_DEVICE",
    "DEFAULT_MAX_CONCURRENT_REQUESTS",
    "DEFAULT_MAX_SAMPLE_COUNT",
    "DEFAULT_PORT",
    "DEFAULT_SCHEMA_VERSION",
    "InferenceServiceConfig",
    "JointFMHTTPService",
    "JointFMRequestHandler",
    "QUERY_MODE_CONDITION",
    "QUERY_MODE_FORECAST",
    "SUPPORTED_CONDITION_KINDS",
    "ServiceError",
    "ServiceErrorCode",
    "create_service_from_env",
    "encode_response",
    "main",
    "serve",
]
