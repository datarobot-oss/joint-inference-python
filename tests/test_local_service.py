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

"""Starting a service from Python with ``LocalService`` and ``ServiceSettings``."""

from __future__ import annotations

import dataclasses
import signal
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from jointfm_inference import (
    LocalService,
    LocalServiceError,
    ServiceSettings,
    cli,
    local_service,
    service_settings,
)
from jointfm_inference.model.service import InferenceServiceConfig
from jointfm_inference.service_settings import (
    BIND_HOST_ENV,
    CHECKPOINT_PATH_ENV,
    CHECKPOINT_URL_ENV,
    DEVICE_ENV,
    PORT_ENV,
)

from tests.support.service import clear_service_environment

# Tiny, with random weights, and owned by the backward compatibility tests.
FROZEN_CHECKPOINT = (
    Path(__file__).parent
    / "compatibility"
    / "jointfm_1_checkpoints"
    / "column-equivariant.pt"
)
# Covers importing torch and loading the checkpoint in a fresh process.
REAL_SERVICE_READY_SECONDS = 120.0
SHORT_TIMEOUT_SECONDS = 0.5
# A stand-in service: answers every GET with 200 on the address the launcher sets.
STAND_IN_SERVICE = """
import os, signal, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

class Healthy(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        pass

if "--ignore-sigterm" in sys.argv:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
address = (os.environ["JOINTFM_INFERENCE_BIND_HOST"], int(os.environ["JOINTFM_INFERENCE_PORT"]))
print("stand-in serving", flush=True)
HTTPServer(address, Healthy).serve_forever()
"""


class _BodyError(Exception):
    """An error raised inside a ``with`` block."""


class _HealthyHandler(BaseHTTPRequestHandler):
    """Answers every GET with 200, as another service on the port would."""

    def do_GET(self) -> None:
        """Answer the health check."""
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        """Keep the test output free of access logs."""
        del format, args


@pytest.fixture
def started() -> Iterator[list[subprocess.Popen[bytes]]]:
    """Record every child process the launcher starts during the test."""
    processes: list[subprocess.Popen[bytes]] = []

    class _RecordingPopen(subprocess.Popen):
        """``subprocess.Popen`` that records each process it starts."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Start the process and record it."""
            super().__init__(*args, **kwargs)
            processes.append(self)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(local_service.subprocess, "Popen", _RecordingPopen)
        yield processes


@pytest.fixture
def occupied_port() -> Iterator[int]:
    """Return a loopback port another service answers health checks on."""
    server = HTTPServer((local_service.LOOPBACK_HOST, 0), _HealthyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


def _serve_stand_in(monkeypatch: pytest.MonkeyPatch, *flags: str) -> None:
    """Make the launcher start the stand-in service instead of the model."""

    def _command() -> list[str]:
        return [sys.executable, "-c", STAND_IN_SERVICE, *flags]

    monkeypatch.setattr(local_service, "_serve_command", _command)


def _every_setting() -> ServiceSettings:
    """Return settings with every field set to a value the service accepts."""
    return ServiceSettings(
        checkpoint_path="/models/weights.pt",
        checkpoint_version="weights-1",
        checkpoint_url="https://example.com/weights.pt",
        checkpoint_sha256="0" * 64,
        checkpoint_token="secret-token",
        device="cpu",
        map_location="cpu",
        precision="fp32",
        bind_host="127.0.0.1",
        port=9090,
        max_sample_count=32,
        max_concurrent_requests=2,
        max_request_bytes=4096,
        model_version="model-1",
    )


def test_every_setting_reaches_the_service_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each field arrives under the variable the service reads it from."""
    clear_service_environment(monkeypatch)
    settings = _every_setting()
    for name, value in settings.environment().items():
        monkeypatch.setenv(name, value)
    config = InferenceServiceConfig.from_env()
    for field in dataclasses.fields(ServiceSettings):
        assert getattr(config, field.name) == getattr(settings, field.name), field.name


def test_settings_cover_every_service_variable() -> None:
    """A variable the service reads but no field sets would be unreachable from Python."""
    variables = {
        value for name, value in vars(service_settings).items() if name.endswith("_ENV")
    }
    assert set(_every_setting().environment()) == variables


def test_unset_settings_leave_the_service_defaults() -> None:
    """Only the checkpoint is required; every other variable stays unset."""
    settings = ServiceSettings(checkpoint_path=Path("weights.pt"))
    assert settings.environment() == {CHECKPOINT_PATH_ENV: "weights.pt"}


def test_settings_keep_the_token_out_of_their_repr() -> None:
    """Logging the settings must not leak the download credential."""
    assert "secret-token" not in repr(_every_setting())


def test_importing_the_launcher_keeps_torch_out_of_the_application() -> None:
    """Only the service process pays for torch and the model."""
    check = (
        "import sys, jointfm_inference; "
        "from jointfm_inference import LocalService, ServiceSettings; "
        "sys.exit('torch' in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", check], check=True)


def test_the_callers_service_variables_do_not_reach_the_service() -> None:
    """Only the settings configure the child, whatever the caller exported."""
    base = {
        "PATH": "/usr/bin",
        DEVICE_ENV: "cuda:0",
        CHECKPOINT_URL_ENV: "https://example.com/other.pt",
    }
    environment = local_service._child_environment(
        base, ServiceSettings(checkpoint_path="weights.pt"), "127.0.0.1", 9090
    )
    assert environment == {
        "PATH": "/usr/bin",
        CHECKPOINT_PATH_ENV: "weights.pt",
        BIND_HOST_ENV: "127.0.0.1",
        PORT_ENV: "9090",
    }


def test_serves_a_checkpoint_until_the_block_ends(
    started: list[subprocess.Popen[bytes]], tmp_path: Path
) -> None:
    """The real service answers every return mode, then stops with the block."""
    log_path = tmp_path / "service.log"
    with LocalService(
        ServiceSettings(checkpoint_path=FROZEN_CHECKPOINT),
        log_path=log_path,
        ready_timeout_seconds=REAL_SERVICE_READY_SECONDS,
    ) as service:
        base_url = service.base_url
        cli.main(["probe", "--service-url", base_url])
    (process,) = started
    assert process.returncode == 0
    assert not local_service._answers_health(base_url)
    assert f"Serving on {base_url}" in log_path.read_text(encoding="utf-8")


def test_reports_why_the_service_did_not_start(
    started: list[subprocess.Popen[bytes]], tmp_path: Path
) -> None:
    """The error carries the service's own refusal of a missing checkpoint."""
    with pytest.raises(
        LocalServiceError, match=r"(?s)exited with status .*names no file"
    ):
        with LocalService(
            ServiceSettings(checkpoint_path=tmp_path / "missing.pt"),
            ready_timeout_seconds=REAL_SERVICE_READY_SECONDS,
        ):
            pass
    (process,) = started
    assert process.returncode != 0


def test_stops_the_service_when_the_block_raises(
    started: list[subprocess.Popen[bytes]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An error in the block stops the service and propagates unchanged."""
    _serve_stand_in(monkeypatch)
    with pytest.raises(_BodyError):
        with LocalService(ServiceSettings(checkpoint_path="unused.pt")):
            raise _BodyError
    (process,) = started
    assert process.returncode == -signal.SIGTERM


def test_kills_a_service_that_ignores_sigterm(
    started: list[subprocess.Popen[bytes]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A service that outlasts the grace period is killed."""
    _serve_stand_in(monkeypatch, "--ignore-sigterm")
    with LocalService(
        ServiceSettings(checkpoint_path="unused.pt"),
        stop_timeout_seconds=SHORT_TIMEOUT_SECONDS,
    ):
        pass
    (process,) = started
    assert process.returncode == -signal.SIGKILL


def test_gives_up_on_a_service_that_never_answers(
    started: list[subprocess.Popen[bytes]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A silent service is stopped, its output reported, and its temporary log removed."""

    def _silent() -> list[str]:
        return [
            sys.executable,
            "-c",
            "import time; print('still loading', flush=True); time.sleep(60)",
        ]

    monkeypatch.setattr(local_service, "_serve_command", _silent)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    with pytest.raises(LocalServiceError, match=r"(?s)did not answer .*still loading"):
        with LocalService(
            ServiceSettings(checkpoint_path="unused.pt"),
            ready_timeout_seconds=SHORT_TIMEOUT_SECONDS,
        ):
            pass
    (process,) = started
    assert process.returncode == -signal.SIGTERM
    assert list(tmp_path.glob("jointfm-inference-*.log")) == []


def test_refuses_a_port_another_service_answers_on(
    started: list[subprocess.Popen[bytes]], occupied_port: int
) -> None:
    """Waiting on someone else's service would report a start that never happened."""
    with pytest.raises(LocalServiceError, match="already answers"):
        with LocalService(
            ServiceSettings(checkpoint_path="unused.pt", port=occupied_port)
        ):
            pass
    assert started == []


def test_base_url_exists_only_inside_the_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A service bound to every interface is reached at the loopback address."""
    _serve_stand_in(monkeypatch)
    service = LocalService(
        ServiceSettings(checkpoint_path="unused.pt", bind_host="0.0.0.0")
    )
    with pytest.raises(LocalServiceError, match="inside its with block"):
        service.base_url
    with service:
        assert service.base_url.startswith(f"http://{local_service.LOOPBACK_HOST}:")
        with pytest.raises(LocalServiceError, match="already running"):
            service.__enter__()
    with pytest.raises(LocalServiceError, match="inside its with block"):
        service.base_url
