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

from http import HTTPStatus

import pytest

import jointfm_inference.model.service.config as service_config
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
    PRECISION_ENV,
)
from tests.support.service import clear_service_environment

DIGEST = "ab" * 32


@pytest.fixture(autouse=True)
def _configured_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from the two required variables and nothing else."""
    clear_service_environment(monkeypatch)

    def _package_version(_name: str) -> str:
        return "9.9.9"

    monkeypatch.setattr(service_config, "package_version", _package_version)
    monkeypatch.setenv(CHECKPOINT_PATH_ENV, "/models/jointfm.pt")
    monkeypatch.setenv(CHECKPOINT_VERSION_ENV, "example-1")


def _refused_field() -> str | None:
    with pytest.raises(ServiceError) as exc_info:
        InferenceServiceConfig.from_env()
    assert exc_info.value.status == HTTPStatus.INTERNAL_SERVER_ERROR
    return exc_info.value.field


def test_from_env_applies_the_defaults_of_every_optional_setting() -> None:
    config = InferenceServiceConfig.from_env()

    assert config.checkpoint_path == "/models/jointfm.pt"
    assert config.checkpoint_version == "example-1"
    assert config.max_sample_count == DEFAULT_MAX_SAMPLE_COUNT
    assert config.max_concurrent_requests == DEFAULT_MAX_CONCURRENT_REQUESTS
    assert config.precision == "bf16"
    assert config.model_version == "jointfm-inference:9.9.9+ckpt.example-1"
    assert config.checkpoint_url is None
    assert config.checkpoint_sha256 is None
    assert config.checkpoint_token is None


@pytest.mark.parametrize("name", [CHECKPOINT_PATH_ENV, CHECKPOINT_VERSION_ENV])
def test_from_env_requires_the_checkpoint_path_and_version(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """No checkpoint is a default: the operator always names the one served."""
    monkeypatch.delenv(name)
    assert _refused_field() == name


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
