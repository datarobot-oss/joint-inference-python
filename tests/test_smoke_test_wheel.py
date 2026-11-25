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

"""The wheel smoke test that ``task build`` runs before every upload."""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

import smoke_test_wheel


def _free_port() -> int:
    with socket.socket() as probe_socket:
        probe_socket.bind((smoke_test_wheel.LOOPBACK_HOST, 0))
        return probe_socket.getsockname()[1]


class _HealthyHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        del format, args


@pytest.fixture
def occupied_port() -> Iterator[int]:
    server = HTTPServer((smoke_test_wheel.LOOPBACK_HOST, 0), _HealthyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


def test_service_environment_replaces_the_callers_service_settings() -> None:
    base = {
        "PATH": "/usr/bin",
        "JOINTFM_INFERENCE_DEVICE": "cuda:0",
        "JOINTFM_INFERENCE_CHECKPOINT_URL": "https://example.com/model.pt",
    }
    environment = smoke_test_wheel.service_environment(base, Path("model.pt"), 18080)
    assert environment == {
        "PATH": "/usr/bin",
        "JOINTFM_INFERENCE_CHECKPOINT_PATH": str(Path("model.pt").resolve()),
        "JOINTFM_INFERENCE_BIND_HOST": "127.0.0.1",
        "JOINTFM_INFERENCE_PORT": "18080",
    }


@pytest.mark.parametrize(
    "wheel_names", [(), ("a-1-py3-none-any.whl", "b-1-py3-none-any.whl")]
)
def test_find_wheel_refuses_an_ambiguous_dist_dir(
    tmp_path: Path, wheel_names: tuple[str, ...]
) -> None:
    for name in wheel_names:
        (tmp_path / name).touch()
    with pytest.raises(SystemExit, match=f"found {len(wheel_names)}"):
        smoke_test_wheel.find_wheel(tmp_path)


def test_main_refuses_a_port_another_service_answers_on(
    tmp_path: Path, occupied_port: int
) -> None:
    (tmp_path / "a-1-py3-none-any.whl").touch()
    with pytest.raises(SystemExit, match="already answers"):
        smoke_test_wheel.main(
            [
                "--dist-dir",
                str(tmp_path),
                "--checkpoint",
                "model.pt",
                "--python",
                "3.13",
                "--port",
                str(occupied_port),
            ]
        )


def test_wait_until_ready_fails_when_the_service_exits_first() -> None:
    service = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"])
    service.wait()
    url = f"http://{smoke_test_wheel.LOOPBACK_HOST}:{_free_port()}"
    with pytest.raises(SystemExit, match="exited with status 3"):
        smoke_test_wheel.wait_until_ready(service, url, timeout_seconds=5.0)


def test_wait_until_ready_returns_once_the_service_answers(occupied_port: int) -> None:
    service = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    url = f"http://{smoke_test_wheel.LOOPBACK_HOST}:{occupied_port}"
    try:
        smoke_test_wheel.wait_until_ready(service, url, timeout_seconds=5.0)
    finally:
        service.kill()
        service.wait()
