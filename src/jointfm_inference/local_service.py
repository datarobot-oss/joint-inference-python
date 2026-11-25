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

"""Start an inference service from Python for the length of a ``with`` block.

An application that installs this package beside ``jointfm-client`` can serve
a checkpoint itself instead of running a separate service. The service runs as
a child process of the caller's interpreter (``python -m jointfm_inference
serve``), not inside the caller's process: it handles SIGINT and SIGTERM,
which Python allows only on the main thread, and it keeps the model's memory
and threads out of the application. A child that the application cannot clean
up after, because the application was killed without unwinding (SIGKILL),
keeps running until it is stopped.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Self

from jointfm_inference.service_settings import (
    BIND_HOST_ENV,
    ENV_PREFIX,
    PORT_ENV,
    ServiceSettings,
)

LOOPBACK_HOST = "127.0.0.1"
# A service bound to every interface is reached through the loopback address.
WILDCARD_HOST = "0.0.0.0"
HEALTH_ROUTE = "/healthz"
# Long enough for a first start that downloads the weights before loading them.
DEFAULT_READY_TIMEOUT_SECONDS = 600.0
DEFAULT_STOP_TIMEOUT_SECONDS = 30.0
HEALTH_TIMEOUT_SECONDS = 1.0
POLL_INTERVAL_SECONDS = 0.25
LOG_TAIL_LINES = 40


class LocalServiceError(RuntimeError):
    """A local service did not start, or was used outside its ``with`` block."""


class LocalService:
    """An inference service that runs while its ``with`` block does.

    Entering the block starts ``jointfm-inference serve`` with ``settings``
    under the caller's interpreter and returns once the service answers
    ``GET /healthz``; leaving it, also through an exception, sends SIGTERM
    and kills the service if it has not stopped within
    ``stop_timeout_seconds``. Only ``settings`` configure the service: the
    caller's own ``JOINTFM_INFERENCE_*`` variables are not passed on. Unlike
    ``jointfm-inference serve``, the service listens on the loopback address
    unless ``settings.bind_host`` says otherwise, and on a free port unless
    ``settings.port`` names one.

    The service's output goes to ``log_path``, appended, or else to a
    temporary file that is removed when the block ends.

    Raises:
        LocalServiceError: on entering, when a service already answers at the
            address, the service exits before it answers, or it does not
            answer within ``ready_timeout_seconds``; the message ends with the
            last lines of the service's output, which say why. A service that
            did not start is stopped before the error is raised.
    """

    def __init__(
        self,
        settings: ServiceSettings,
        *,
        log_path: str | os.PathLike[str] | None = None,
        ready_timeout_seconds: float = DEFAULT_READY_TIMEOUT_SECONDS,
        stop_timeout_seconds: float = DEFAULT_STOP_TIMEOUT_SECONDS,
    ) -> None:
        """Store the service's settings; nothing starts before the ``with`` block."""
        self._settings = settings
        self._log_path = None if log_path is None else Path(log_path)
        self._ready_timeout_seconds = ready_timeout_seconds
        self._stop_timeout_seconds = stop_timeout_seconds
        self._process: subprocess.Popen[bytes] | None = None
        self._base_url: str | None = None
        self._temporary_log: Path | None = None

    @property
    def base_url(self) -> str:
        """Return the address the service answers at, such as ``http://127.0.0.1:41234``.

        Raises:
            LocalServiceError: outside the ``with`` block.
        """
        if self._base_url is None:
            raise LocalServiceError("the service runs only inside its with block")
        return self._base_url

    def __enter__(self) -> Self:
        """Start the service and return once it answers its health check."""
        if self._process is not None:
            raise LocalServiceError("the service is already running")
        host = (
            LOOPBACK_HOST
            if self._settings.bind_host is None
            else self._settings.bind_host
        )
        port = _free_port(host) if self._settings.port is None else self._settings.port
        base_url = f"http://{_client_host(host)}:{port}"
        # Waiting on someone else's service would report a start that never happened.
        if _answers_health(base_url):
            raise LocalServiceError(
                f"a service already answers on {base_url}; set another port"
            )
        log_path = self._log_path
        if log_path is None:
            descriptor, name = tempfile.mkstemp(
                prefix="jointfm-inference-", suffix=".log"
            )
            os.close(descriptor)
            log_path = self._temporary_log = Path(name)
        with log_path.open("ab") as log:
            self._process = subprocess.Popen(
                _serve_command(),
                env=_child_environment(os.environ, self._settings, host, port),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        try:
            _wait_until_ready(
                self._process, base_url, self._ready_timeout_seconds, log_path
            )
        except BaseException:
            self._stop()
            raise
        self._base_url = base_url
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop the service; an exception raised in the block propagates."""
        self._stop()

    def _stop(self) -> None:
        """Stop the child, killing it after the grace period, and remove a temporary log."""
        process = self._process
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=self._stop_timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if self._temporary_log is not None:
            self._temporary_log.unlink(missing_ok=True)
        self._process = None
        self._base_url = None
        self._temporary_log = None


def _child_environment(
    base: Mapping[str, str], settings: ServiceSettings, host: str, port: int
) -> dict[str, str]:
    """Return ``base`` with its service variables replaced by those of ``settings``, ``host``, and ``port``."""
    environment = {
        name: value for name, value in base.items() if not name.startswith(ENV_PREFIX)
    }
    environment.update(settings.environment())
    environment[BIND_HOST_ENV] = host
    environment[PORT_ENV] = str(port)
    return environment


def _serve_command() -> list[str]:
    """Return the command that serves from the caller's own environment."""
    return [sys.executable, "-m", "jointfm_inference", "serve"]


def _client_host(bind_host: str) -> str:
    """Return the host a client on this machine reaches a service bound to ``bind_host`` at."""
    return LOOPBACK_HOST if bind_host == WILDCARD_HOST else bind_host


def _free_port(host: str) -> int:
    """Return a port on ``host`` that no socket is bound to at this moment."""
    with socket.socket() as probe:
        probe.bind((host, 0))
        return probe.getsockname()[1]


def _answers_health(base_url: str) -> bool:
    """Return whether a service at ``base_url`` answers ``GET /healthz`` with HTTP 200."""
    try:
        with urllib.request.urlopen(
            f"{base_url}{HEALTH_ROUTE}", timeout=HEALTH_TIMEOUT_SECONDS
        ) as response:
            return response.status == 200
    except OSError:
        return False


def _wait_until_ready(
    process: subprocess.Popen[bytes],
    base_url: str,
    timeout_seconds: float,
    log_path: Path,
) -> None:
    """Return once ``process`` answers its health check at ``base_url``.

    Raises:
        LocalServiceError: when the process exits first, or does not answer
            within ``timeout_seconds``.
    """
    deadline = time.monotonic() + timeout_seconds
    while not _answers_health(base_url):
        if process.poll() is not None:
            raise LocalServiceError(
                f"the service exited with status {process.returncode} before "
                f"answering {base_url}{HEALTH_ROUTE}:\n{_log_tail(log_path)}"
            )
        if time.monotonic() > deadline:
            raise LocalServiceError(
                f"the service did not answer {base_url}{HEALTH_ROUTE} within "
                f"{timeout_seconds:g} s:\n{_log_tail(log_path)}"
            )
        time.sleep(POLL_INTERVAL_SECONDS)
    # A health answer from a process that has already exited came from another service.
    if process.poll() is not None:
        raise LocalServiceError(
            f"the service exited with status {process.returncode} while another "
            f"process answered {base_url}{HEALTH_ROUTE}:\n{_log_tail(log_path)}"
        )


def _log_tail(log_path: Path) -> str:
    """Return the last lines of the service's output, which give the reason it failed."""
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-LOG_TAIL_LINES:])
