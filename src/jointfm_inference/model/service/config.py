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

"""Runtime configuration of the inference service, read from the environment.

Every setting is a ``JOINTFM_INFERENCE_*`` environment variable, so one
git-ignored ``.env`` file configures a local process and a container alike.
Nothing names a default checkpoint: which weights a service loads, and from
where, is always the operator's configuration.

``package_version`` is imported at module scope so tests can patch the symbol
at ``jointfm_inference.model.service.config.package_version`` to control the reported
image version without booting a full container.
"""

from __future__ import annotations

import os
import re
from http import HTTPStatus
from pathlib import Path
from urllib.parse import urlsplit

from jointfm_inference.core.version import package_version
from jointfm_inference.model.checkpoint import file_sha256
from jointfm_inference.model.inference import DEFAULT_INFERENCE_PRECISION
from jointfm_inference.model.precision import MODEL_PRECISIONS

from .constants import (
    DEFAULT_BIND_HOST,
    DEFAULT_DEVICE,
    DEFAULT_MAX_CONCURRENT_REQUESTS,
    DEFAULT_MAX_SAMPLE_COUNT,
    DEFAULT_PORT,
    DEFAULT_SCHEMA_VERSION,
)
from .errors import ServiceError
from .validation import (
    _choice_from_env,
    _optional_env,
    _positive_int_from_env,
    _required_env,
)

ENV_PREFIX = "JOINTFM_INFERENCE_"
CHECKPOINT_PATH_ENV = f"{ENV_PREFIX}CHECKPOINT_PATH"
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
MODEL_VERSION_ENV = f"{ENV_PREFIX}MODEL_VERSION"
DOWNLOAD_URL_SCHEMES = ("https", "http")
# A bearer token travels only over TLS.
TOKEN_URL_SCHEME = "https"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
# Enough of the digest to tell checkpoints apart while staying readable.
CHECKPOINT_VERSION_LENGTH = 12


class InferenceServiceConfig:
    """Runtime configuration of one inference service process."""

    def __init__(
        self,
        *,
        checkpoint_path: str,
        checkpoint_version: str,
        device: str,
        precision: str,
        map_location: str,
        bind_host: str,
        port: int,
        schema_version: str,
        image_version: str,
        model_version: str,
        max_sample_count: int,
        max_concurrent_requests: int,
        checkpoint_url: str | None = None,
        checkpoint_sha256: str | None = None,
        checkpoint_token: str | None = None,
    ) -> None:
        """Store validated runtime settings of one inference service process.

        ``precision`` is the model precision (``JOINTFM_INFERENCE_PRECISION``)
        every forward runs at; the session re-validates it against the model's
        device. ``checkpoint_url``, ``checkpoint_sha256``, and
        ``checkpoint_token`` describe where startup downloads the checkpoint to
        ``checkpoint_path`` when the file is not there yet; with no URL, the
        file must already be at ``checkpoint_path``, for example mounted into a
        container.
        """
        self.checkpoint_path = checkpoint_path
        self.checkpoint_version = checkpoint_version
        self.device = device
        self.precision = precision
        self.map_location = map_location
        self.bind_host = bind_host
        self.port = port
        self.schema_version = schema_version
        self.image_version = image_version
        self.model_version = model_version
        self.max_sample_count = max_sample_count
        self.max_concurrent_requests = max_concurrent_requests
        self.checkpoint_url = checkpoint_url
        self.checkpoint_sha256 = checkpoint_sha256
        self.checkpoint_token = checkpoint_token

    @classmethod
    def from_env(cls) -> "InferenceServiceConfig":
        """Build validated runtime configuration from environment variables.

        ``JOINTFM_INFERENCE_CHECKPOINT_PATH`` is required. A download
        source is ``JOINTFM_INFERENCE_CHECKPOINT_URL`` together with the
        file's ``JOINTFM_INFERENCE_CHECKPOINT_SHA256``, optionally with a
        ``JOINTFM_INFERENCE_CHECKPOINT_TOKEN`` sent as a bearer token.

        The checkpoint version is the start of the file's SHA-256 digest, so
        it changes exactly when the weights do: the configured digest for a
        download source, otherwise the digest of the file at the path.

        Raises:
            ServiceError: A required variable is missing, a value is malformed,
                no download source is set and no file is at the path, or the
                download variables are set inconsistently: a URL without its
                digest, a digest or token without a URL, a URL scheme other
                than ``https`` or ``http``, or a token for a URL that is not
                ``https``.
        """
        checkpoint_path = _required_env(CHECKPOINT_PATH_ENV)
        checkpoint_url, checkpoint_sha256, checkpoint_token = (
            _download_source_from_env()
        )
        checkpoint_version = _checkpoint_digest(checkpoint_path, checkpoint_sha256)[
            :CHECKPOINT_VERSION_LENGTH
        ]
        device = os.getenv(DEVICE_ENV, DEFAULT_DEVICE)
        precision = _choice_from_env(
            PRECISION_ENV,
            DEFAULT_INFERENCE_PRECISION,
            MODEL_PRECISIONS,
        )
        map_location = os.getenv(MAP_LOCATION_ENV, device)
        bind_host = os.getenv(BIND_HOST_ENV, DEFAULT_BIND_HOST)
        port = _positive_int_from_env(PORT_ENV, DEFAULT_PORT)
        max_sample_count = _positive_int_from_env(
            MAX_SAMPLE_COUNT_ENV,
            DEFAULT_MAX_SAMPLE_COUNT,
        )
        max_concurrent_requests = _positive_int_from_env(
            MAX_CONCURRENT_REQUESTS_ENV,
            DEFAULT_MAX_CONCURRENT_REQUESTS,
        )

        image_version = package_version("jointfm-inference")
        model_version = os.getenv(
            MODEL_VERSION_ENV,
            f"jointfm-inference:{image_version}+ckpt.{checkpoint_version}",
        )
        return cls(
            checkpoint_path=checkpoint_path,
            checkpoint_version=checkpoint_version,
            device=device,
            precision=precision,
            map_location=map_location,
            bind_host=bind_host,
            port=port,
            schema_version=DEFAULT_SCHEMA_VERSION,
            image_version=image_version,
            model_version=model_version,
            max_sample_count=max_sample_count,
            max_concurrent_requests=max_concurrent_requests,
            checkpoint_url=checkpoint_url,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_token=checkpoint_token,
        )


def _checkpoint_digest(checkpoint_path: str, configured_sha256: str | None) -> str:
    """Return the configured download digest, or the digest of the local file."""
    if configured_sha256 is not None:
        return configured_sha256
    path = Path(checkpoint_path)
    if not path.is_file():
        raise _config_error(
            CHECKPOINT_PATH_ENV,
            f"{CHECKPOINT_PATH_ENV} names no file and {CHECKPOINT_URL_ENV} is "
            f"not set: {checkpoint_path}",
        )
    return file_sha256(path)


def _download_source_from_env() -> tuple[str | None, str | None, str | None]:
    """Return the validated ``(url, sha256, token)`` checkpoint download source."""
    url = _optional_env(CHECKPOINT_URL_ENV)
    sha256 = _optional_env(CHECKPOINT_SHA256_ENV)
    token = _optional_env(CHECKPOINT_TOKEN_ENV)
    if url is None:
        for name, value in (
            (CHECKPOINT_SHA256_ENV, sha256),
            (CHECKPOINT_TOKEN_ENV, token),
        ):
            if value is not None:
                raise _config_error(
                    name, f"{name} is set but {CHECKPOINT_URL_ENV} is not"
                )
        return None, None, None
    scheme = urlsplit(url).scheme
    if scheme not in DOWNLOAD_URL_SCHEMES:
        raise _config_error(
            CHECKPOINT_URL_ENV,
            f"{CHECKPOINT_URL_ENV} must use one of the schemes "
            f"{DOWNLOAD_URL_SCHEMES}, got {scheme!r}",
        )
    if sha256 is None:
        raise _config_error(
            CHECKPOINT_SHA256_ENV,
            f"{CHECKPOINT_SHA256_ENV} is required with {CHECKPOINT_URL_ENV}, "
            "so a downloaded checkpoint is verified before it is loaded",
        )
    if _SHA256_PATTERN.fullmatch(sha256) is None:
        raise _config_error(
            CHECKPOINT_SHA256_ENV,
            f"{CHECKPOINT_SHA256_ENV} must be 64 lowercase hexadecimal digits",
        )
    if token is not None and scheme != TOKEN_URL_SCHEME:
        raise _config_error(
            CHECKPOINT_TOKEN_ENV,
            f"{CHECKPOINT_TOKEN_ENV} is sent only to an {TOKEN_URL_SCHEME} "
            f"{CHECKPOINT_URL_ENV}, got {scheme!r}",
        )
    return url, sha256, token


def _config_error(name: str, message: str) -> ServiceError:
    """Return the startup error for one inconsistent environment variable."""
    return ServiceError(
        "VALIDATION_ERROR",
        message,
        status=HTTPStatus.INTERNAL_SERVER_ERROR,
        field=name,
    )
