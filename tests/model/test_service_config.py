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

"""The service's environment configuration."""

from __future__ import annotations

import hashlib
import json
from http import HTTPStatus
from pathlib import Path

import pytest

import jointfm_inference.model.service.config as service_config
from jointfm_inference.model.inference import CapacityEnvelope
from jointfm_inference.model.service import (
    DEFAULT_MAX_CONCURRENT_REQUESTS,
    DEFAULT_MAX_SAMPLE_COUNT,
    InferenceServiceConfig,
    ServiceError,
)
from jointfm_inference.model.service.config import (
    CHECKPOINT_PATH_ENV,
    CHECKPOINT_SHA256_ENV,
    CHECKPOINT_TOKEN_ENV,
    CHECKPOINT_URL_ENV,
    CHECKPOINT_VERSION_ENV,
    MAX_CONCURRENT_REQUESTS_ENV,
    MAX_REQUEST_BYTES_ENV,
    PRECISION_ENV,
    default_max_request_bytes,
)
from jointfm_inference.model.service.constants import REQUEST_COLUMN_NAME_BYTES
from tests.support.service import clear_service_environment

DIGEST = "ab" * 32
CHECKPOINT_BYTES = b"checkpoint bytes"
CHECKPOINT_DIGEST = hashlib.sha256(CHECKPOINT_BYTES).hexdigest()


@pytest.fixture
def checkpoint_path(tmp_path: Path) -> Path:
    path = tmp_path / "jointfm.pt"
    path.write_bytes(CHECKPOINT_BYTES)
    return path


@pytest.fixture(autouse=True)
def _configured_environment(
    monkeypatch: pytest.MonkeyPatch, checkpoint_path: Path
) -> None:
    """Start every test from a present checkpoint file and nothing else."""
    clear_service_environment(monkeypatch)

    def _package_version(_name: str) -> str:
        return "9.9.9"

    monkeypatch.setattr(service_config, "package_version", _package_version)
    monkeypatch.setenv(CHECKPOINT_PATH_ENV, str(checkpoint_path))


def _refused_field() -> str | None:
    with pytest.raises(ServiceError) as exc_info:
        InferenceServiceConfig.from_env()
    assert exc_info.value.status == HTTPStatus.INTERNAL_SERVER_ERROR
    return exc_info.value.field


def test_from_env_applies_the_defaults_of_every_optional_setting(
    checkpoint_path: Path,
) -> None:
    config = InferenceServiceConfig.from_env()

    assert config.checkpoint_path == str(checkpoint_path)
    assert config.max_sample_count == DEFAULT_MAX_SAMPLE_COUNT
    assert config.max_concurrent_requests == DEFAULT_MAX_CONCURRENT_REQUESTS
    assert config.precision == "bf16"
    assert config.checkpoint_url is None
    assert config.checkpoint_sha256 is None
    assert config.checkpoint_token is None
    assert config.max_request_bytes is None


def test_from_env_requires_the_checkpoint_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No checkpoint is a default: the operator always names the one served."""
    monkeypatch.delenv(CHECKPOINT_PATH_ENV)
    assert _refused_field() == CHECKPOINT_PATH_ENV


def test_from_env_versions_a_local_checkpoint_by_its_digest() -> None:
    """The version changes exactly when the weights do, with nothing to configure."""
    config = InferenceServiceConfig.from_env()

    assert config.checkpoint_version == CHECKPOINT_DIGEST[:12]
    assert (
        config.model_version == f"jointfm-inference:9.9.9+ckpt.{CHECKPOINT_DIGEST[:12]}"
    )


def test_from_env_versions_a_download_by_its_configured_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A download source names its digest, so the file need not exist yet."""
    monkeypatch.setenv(CHECKPOINT_PATH_ENV, str(tmp_path / "absent.pt"))
    monkeypatch.setenv(CHECKPOINT_URL_ENV, "https://example.com/jointfm.pt")
    monkeypatch.setenv(CHECKPOINT_SHA256_ENV, DIGEST)

    assert InferenceServiceConfig.from_env().checkpoint_version == DIGEST[:12]


def test_from_env_uses_a_configured_version_label(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A set label wins, and the file is not read until the service loads it."""
    monkeypatch.setenv(CHECKPOINT_PATH_ENV, str(tmp_path / "absent.pt"))
    monkeypatch.setenv(CHECKPOINT_VERSION_ENV, "example-1")

    config = InferenceServiceConfig.from_env()

    assert config.checkpoint_version == "example-1"
    assert config.model_version == "jointfm-inference:9.9.9+ckpt.example-1"


def test_from_env_treats_an_empty_version_label_as_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(CHECKPOINT_VERSION_ENV, "")
    assert (
        InferenceServiceConfig.from_env().checkpoint_version == CHECKPOINT_DIGEST[:12]
    )


def test_from_env_refuses_a_missing_local_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(CHECKPOINT_PATH_ENV, str(tmp_path / "absent.pt"))
    assert _refused_field() == CHECKPOINT_PATH_ENV


def test_from_env_reads_max_concurrent_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MAX_CONCURRENT_REQUESTS_ENV, "4")
    assert InferenceServiceConfig.from_env().max_concurrent_requests == 4


def test_from_env_refuses_a_non_positive_max_concurrent_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero concurrency cap is a startup error, not a silent fallback."""
    monkeypatch.setenv(MAX_CONCURRENT_REQUESTS_ENV, "0")
    assert _refused_field() == MAX_CONCURRENT_REQUESTS_ENV


@pytest.mark.parametrize("precision", ["fp32", "bf16", "fp16"])
def test_from_env_reads_precision(
    monkeypatch: pytest.MonkeyPatch, precision: str
) -> None:
    monkeypatch.setenv(PRECISION_ENV, precision)
    assert InferenceServiceConfig.from_env().precision == precision


@pytest.mark.parametrize("precision", ["tf32", "BF16", "float32", ""])
def test_from_env_refuses_an_unknown_precision(
    monkeypatch: pytest.MonkeyPatch, precision: str
) -> None:
    """An unknown precision is a startup error, not a silent fp32 fallback."""
    monkeypatch.setenv(PRECISION_ENV, precision)
    assert _refused_field() == PRECISION_ENV


def test_from_env_reads_a_download_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CHECKPOINT_URL_ENV, "https://example.com/jointfm.pt")
    monkeypatch.setenv(CHECKPOINT_SHA256_ENV, DIGEST)
    monkeypatch.setenv(CHECKPOINT_TOKEN_ENV, "secret")

    config = InferenceServiceConfig.from_env()

    assert config.checkpoint_url == "https://example.com/jointfm.pt"
    assert config.checkpoint_sha256 == DIGEST
    assert config.checkpoint_token == "secret"


def test_from_env_treats_an_empty_download_variable_as_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configuration file that lists a variable without a value sets nothing."""
    for name in (CHECKPOINT_URL_ENV, CHECKPOINT_SHA256_ENV, CHECKPOINT_TOKEN_ENV):
        monkeypatch.setenv(name, "")
    assert InferenceServiceConfig.from_env().checkpoint_url is None


@pytest.mark.parametrize(
    ("settings", "field"),
    [
        ({CHECKPOINT_SHA256_ENV: DIGEST}, CHECKPOINT_SHA256_ENV),
        ({CHECKPOINT_TOKEN_ENV: "secret"}, CHECKPOINT_TOKEN_ENV),
        ({CHECKPOINT_URL_ENV: "https://example.com/c.pt"}, CHECKPOINT_SHA256_ENV),
        (
            {
                CHECKPOINT_URL_ENV: "https://example.com/c.pt",
                CHECKPOINT_SHA256_ENV: "AB",
            },
            CHECKPOINT_SHA256_ENV,
        ),
        (
            {
                CHECKPOINT_URL_ENV: "https://example.com/c.pt",
                CHECKPOINT_SHA256_ENV: DIGEST.upper(),
            },
            CHECKPOINT_SHA256_ENV,
        ),
        (
            {CHECKPOINT_URL_ENV: "file:///models/c.pt", CHECKPOINT_SHA256_ENV: DIGEST},
            CHECKPOINT_URL_ENV,
        ),
        (
            {
                CHECKPOINT_URL_ENV: "http://example.com/c.pt",
                CHECKPOINT_SHA256_ENV: DIGEST,
                CHECKPOINT_TOKEN_ENV: "secret",
            },
            CHECKPOINT_TOKEN_ENV,
        ),
    ],
    ids=[
        "digest-without-url",
        "token-without-url",
        "url-without-digest",
        "short-digest",
        "uppercase-digest",
        "file-scheme",
        "token-over-http",
    ],
)
def test_from_env_refuses_an_inconsistent_download_source(
    monkeypatch: pytest.MonkeyPatch, settings: dict[str, str], field: str
) -> None:
    for name, value in settings.items():
        monkeypatch.setenv(name, value)
    assert _refused_field() == field


def test_from_env_reads_max_request_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A set request cap replaces the one derived from the capacity envelope."""
    monkeypatch.setenv(MAX_REQUEST_BYTES_ENV, "4096")
    assert InferenceServiceConfig.from_env().max_request_bytes == 4096


def test_from_env_treats_an_empty_max_request_bytes_as_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MAX_REQUEST_BYTES_ENV, "")
    assert InferenceServiceConfig.from_env().max_request_bytes is None


@pytest.mark.parametrize("raw_value", ["0", "-1", "lots"])
def test_from_env_refuses_a_malformed_max_request_bytes(
    monkeypatch: pytest.MonkeyPatch, raw_value: str
) -> None:
    """A request cap that is not a positive integer is a startup error."""
    monkeypatch.setenv(MAX_REQUEST_BYTES_ENV, raw_value)
    assert _refused_field() == MAX_REQUEST_BYTES_ENV


# The longest JSON spellings of a float64 at full precision and of an ISO 8601
# timestamp with nanoseconds and a UTC offset.
_LONGEST_FLOAT = -2.2250738585072014e-308
_LONGEST_TIMESTAMP = "2026-11-18T12:34:56.123456789+05:30"


def _envelope_limit_request(envelope: CapacityEnvelope) -> dict[str, object]:
    """Return a request at least as large as any a client sizes to ``envelope``.

    Every limit is reached at once: ``max_series`` columns whose names take
    the budgeted length, ``n_input`` history rows, ``n_output`` query times
    with their scored query rows, and an interval condition of its own on
    every column at every query time, each value at full float64 precision.
    """
    time_column = "t" * REQUEST_COLUMN_NAME_BYTES
    names = [
        f"{index:0{REQUEST_COLUMN_NAME_BYTES}d}" for index in range(envelope.max_series)
    ]
    columns = [
        {
            "name": name,
            "modality": "numeric",
            "role": "target",
            "nullable": False,
            "lower_bound": _LONGEST_FLOAT,
            "upper_bound": -_LONGEST_FLOAT,
        }
        for name in names
    ]
    conditions = [
        {
            "column": name,
            "kind": "interval",
            "lower": _LONGEST_FLOAT,
            "upper": -_LONGEST_FLOAT,
            "query_time_indices": [position],
        }
        for name in names
        for position in range(envelope.n_output)
    ]

    def row() -> dict[str, object]:
        return {time_column: _LONGEST_TIMESTAMP, **dict.fromkeys(names, _LONGEST_FLOAT)}

    return {
        "request_type": "predict",
        "schema_version": "v6",
        "model_version": "jointfm-inference:0.0.0+ckpt.0123456789ab",
        "query_mode": "condition",
        "return_mode": "log_prob",
        "time_index_mode": "absolute_datetime",
        "time_column": time_column,
        "time_scale_seconds": _LONGEST_FLOAT,
        "use_local_normalized_time": False,
        "calendar_id": "pandas-default",
        "timezone": "America/Argentina/ComodRivadavia",
        "columns": columns,
        "requested_columns": names,
        "history_rows": [row() for _ in range(envelope.n_input)],
        "query_times": [_LONGEST_TIMESTAMP] * envelope.n_output,
        "query_rows": [row() for _ in range(envelope.n_output)],
        "conditions": conditions,
        "n_samples": 1000,
        "quantiles": [0.05, 0.5, 0.95],
        "seed": 2**63 - 1,
    }


@pytest.mark.parametrize(
    "envelope",
    [
        CapacityEnvelope(
            min_series=1, max_series=1, t_input=1.0, t_output=1.0, n_input=1, n_output=1
        ),
        CapacityEnvelope(
            min_series=2,
            max_series=7,
            t_input=1.0,
            t_output=1.0,
            n_input=50,
            n_output=9,
        ),
        CapacityEnvelope(
            min_series=2,
            max_series=100,
            t_input=1.0,
            t_output=1.0,
            n_input=732,
            n_output=288,
        ),
    ],
    ids=["single-cell", "narrow", "wide"],
)
def test_the_default_request_cap_admits_a_request_at_every_envelope_limit(
    envelope: CapacityEnvelope,
) -> None:
    """No request the client sizes to the envelope is refused for its size.

    The body is written as ``requests`` writes ``json=``, with the standard
    library's ``", "`` and ``": "`` separators.
    """
    body = json.dumps(_envelope_limit_request(envelope)).encode("utf-8")

    assert len(body) <= default_max_request_bytes(envelope)
