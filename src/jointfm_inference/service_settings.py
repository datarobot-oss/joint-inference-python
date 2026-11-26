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

"""The settings of one inference service: their environment variables and their Python form.

``jointfm-inference serve`` reads every setting from a ``JOINTFM_INFERENCE_*``
environment variable. This module owns those names, so the service that reads
them (``jointfm_inference.model.service.config``) and the launcher that writes
them (``jointfm_inference.local_service``) cannot drift apart. It imports
neither torch nor the model, so an application can start a service without
loading either into its own process.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

ENV_PREFIX = "JOINTFM_INFERENCE_"
CHECKPOINT_PATH_ENV = f"{ENV_PREFIX}CHECKPOINT_PATH"
CHECKPOINT_VERSION_ENV = f"{ENV_PREFIX}CHECKPOINT_VERSION"
CHECKPOINT_URL_ENV = f"{ENV_PREFIX}CHECKPOINT_URL"
CHECKPOINT_SHA256_ENV = f"{ENV_PREFIX}CHECKPOINT_SHA256"
CHECKPOINT_TOKEN_ENV = f"{ENV_PREFIX}CHECKPOINT_TOKEN"
DEVICE_ENV = f"{ENV_PREFIX}DEVICE"
PRECISION_ENV = f"{ENV_PREFIX}PRECISION"
MAP_LOCATION_ENV = f"{ENV_PREFIX}MAP_LOCATION"
BIND_HOST_ENV = f"{ENV_PREFIX}BIND_HOST"
PORT_ENV = f"{ENV_PREFIX}PORT"
MAX_SAMPLE_COUNT_ENV = f"{ENV_PREFIX}MAX_SAMPLE_COUNT"
MAX_CONCURRENT_REQUESTS_ENV = f"{ENV_PREFIX}MAX_CONCURRENT_REQUESTS"
MAX_REQUEST_BYTES_ENV = f"{ENV_PREFIX}MAX_REQUEST_BYTES"
MODEL_VERSION_ENV = f"{ENV_PREFIX}MODEL_VERSION"
LOG_DIR_ENV = f"{ENV_PREFIX}LOG_DIR"


@dataclass(frozen=True)
class ServiceSettings:
    """The settings of one inference service, as Python values.

    Each field stands for the ``JOINTFM_INFERENCE_*`` variable of the same
    name, with the meaning and default the README's configuration table gives
    it; ``None`` leaves the variable unset, so the service applies that
    default. Nothing is validated here: the service checks every value when it
    starts and names the variable of a value it refuses, so the rules live in
    one place. ``checkpoint_token`` is left out of ``repr``, because it is a
    credential.
    """

    checkpoint_path: str | os.PathLike[str]
    checkpoint_version: str | None = None
    checkpoint_url: str | None = None
    checkpoint_sha256: str | None = None
    checkpoint_token: str | None = field(default=None, repr=False)
    device: str | None = None
    map_location: str | None = None
    precision: str | None = None
    bind_host: str | None = None
    port: int | None = None
    max_sample_count: int | None = None
    max_concurrent_requests: int | None = None
    max_request_bytes: int | None = None
    model_version: str | None = None
    log_dir: str | os.PathLike[str] | None = None

    def environment(self) -> dict[str, str]:
        """Return the ``JOINTFM_INFERENCE_*`` variables that the set fields stand for."""
        variables = {
            CHECKPOINT_PATH_ENV: os.fspath(self.checkpoint_path),
            CHECKPOINT_VERSION_ENV: self.checkpoint_version,
            CHECKPOINT_URL_ENV: self.checkpoint_url,
            CHECKPOINT_SHA256_ENV: self.checkpoint_sha256,
            CHECKPOINT_TOKEN_ENV: self.checkpoint_token,
            DEVICE_ENV: self.device,
            MAP_LOCATION_ENV: self.map_location,
            PRECISION_ENV: self.precision,
            BIND_HOST_ENV: self.bind_host,
            PORT_ENV: self.port,
            MAX_SAMPLE_COUNT_ENV: self.max_sample_count,
            MAX_CONCURRENT_REQUESTS_ENV: self.max_concurrent_requests,
            MAX_REQUEST_BYTES_ENV: self.max_request_bytes,
            MODEL_VERSION_ENV: self.model_version,
            LOG_DIR_ENV: self.log_dir,
        }
        return {
            name: str(value) for name, value in variables.items() if value is not None
        }
