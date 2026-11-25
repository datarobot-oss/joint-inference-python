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

"""Install the built wheel into a fresh environment, serve a checkpoint from it, and probe it.

``task build`` runs this on the wheel it has just built, and the publish
workflow runs ``task build`` before every upload, so a release cannot reach
PyPI when its wheel lacks a module, package data, or an entry point, or needs
a package that only the development environment provides. The fresh
environment resolves the wheel's dependencies as ``pip install`` does for a
user, without ``uv.lock``, but with PyTorch's CPU build to keep it small.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import time
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path

SERVICE_SETTING_PREFIX = "JOINTFM_INFERENCE_"
LOOPBACK_HOST = "127.0.0.1"
DEFAULT_PORT = 18080
HEALTH_TIMEOUT_SECONDS = 1.0
READY_TIMEOUT_SECONDS = 120.0
POLL_INTERVAL_SECONDS = 0.5
STOP_TIMEOUT_SECONDS = 30.0
LOG_TAIL_LINES = 40


def find_wheel(dist_dir: Path) -> Path:
    """Return the one wheel in ``dist_dir``.

    Raises:
        SystemExit: when ``dist_dir`` holds no wheel or more than one, since
            which one would be tested is then ambiguous.
    """
    wheels = sorted(dist_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(
            f"expected exactly one wheel in {dist_dir}, found {len(wheels)}"
        )
    return wheels[0]


def service_environment(
    base: Mapping[str, str], checkpoint: Path, port: int
) -> dict[str, str]:
    """Return ``base`` with every service setting replaced by those that serve ``checkpoint``.

    The caller's own ``JOINTFM_INFERENCE_*`` settings, such as a CUDA device or
    a download URL, are dropped, so the service under test serves exactly the
    given checkpoint on the CPU at ``port`` of the loopback address.
    """
    environment = {
        name: value
        for name, value in base.items()
        if not name.startswith(SERVICE_SETTING_PREFIX)
    }
    environment[f"{SERVICE_SETTING_PREFIX}CHECKPOINT_PATH"] = str(checkpoint.resolve())
    environment[f"{SERVICE_SETTING_PREFIX}BIND_HOST"] = LOOPBACK_HOST
    environment[f"{SERVICE_SETTING_PREFIX}PORT"] = str(port)
    return environment


def answers_health(url: str) -> bool:
    """Return whether a service at ``url`` answers ``GET /healthz`` with HTTP 200."""
    try:
        with urllib.request.urlopen(
            f"{url}/healthz", timeout=HEALTH_TIMEOUT_SECONDS
        ) as response:
            return response.status == 200
    except OSError:
        return False


def install(wheel: Path, environment_dir: Path, python: str) -> Path:
    """Create a fresh environment at ``environment_dir``, install ``wheel`` into it, and return its command.

    Raises:
        subprocess.CalledProcessError: when the environment cannot be created
            or the wheel or one of its dependencies does not install.
    """
    subprocess.run(
        ["uv", "venv", "--quiet", "--python", python, str(environment_dir)], check=True
    )
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--quiet",
            "--python",
            str(environment_dir),
            "--torch-backend",
            "cpu",
            str(wheel),
        ],
        check=True,
    )
    return environment_dir / "bin" / "jointfm-inference"


def wait_until_ready(
    service: subprocess.Popen[bytes], url: str, timeout_seconds: float
) -> None:
    """Return once ``service`` answers its health check at ``url``.

    Raises:
        SystemExit: when ``service`` exits first, or does not answer within
            ``timeout_seconds``.
    """
    deadline = time.monotonic() + timeout_seconds
    while True:
        if service.poll() is not None:
            raise SystemExit(
                f"the service exited with status {service.returncode} before answering {url}/healthz"
            )
        if answers_health(url):
            return
        if time.monotonic() > deadline:
            raise SystemExit(
                f"the service did not answer {url}/healthz within {timeout_seconds:.0f} s"
            )
        time.sleep(POLL_INTERVAL_SECONDS)


def serve_and_probe(command: Path, checkpoint: Path, port: int, log_path: Path) -> int:
    """Serve ``checkpoint`` with ``command``, probe every return mode, stop the service, and return the probe's status.

    Raises:
        SystemExit: when the service does not come up.
    """
    url = f"http://{LOOPBACK_HOST}:{port}"
    with log_path.open("wb") as log:
        service = subprocess.Popen(
            [str(command), "serve"],
            env=service_environment(os.environ, checkpoint, port),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            wait_until_ready(service, url, READY_TIMEOUT_SECONDS)
            return subprocess.run(
                [str(command), "probe", "--service-url", url], check=False
            ).returncode
        finally:
            service.terminate()
            service.wait(timeout=STOP_TIMEOUT_SECONDS)


def _print_log_tail(log_path: Path) -> None:
    """Print the end of the service log, which holds the reason a failed service gave."""
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    print(f"--- last {LOG_TAIL_LINES} lines of the service log ---")
    print("\n".join(lines[-LOG_TAIL_LINES:]))


def main(argv: Sequence[str] | None = None) -> None:
    """Run the smoke test the command-line arguments describe.

    Raises:
        SystemExit: when another service already answers on the port, the
            service does not come up, or the probe fails.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=Path("dist"),
        help="Directory holding the one wheel to test",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Serving checkpoint the installed wheel serves",
    )
    parser.add_argument(
        "--python", required=True, help="Python version of the fresh environment"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Loopback port to serve on (default: %(default)s)",
    )
    arguments = parser.parse_args(argv)

    wheel = find_wheel(arguments.dist_dir)
    url = f"http://{LOOPBACK_HOST}:{arguments.port}"
    # Probing someone else's service would pass without testing the wheel.
    if answers_health(url):
        raise SystemExit(f"a service already answers on {url}; pass a free --port")
    with tempfile.TemporaryDirectory(prefix="jointfm-inference-wheel-") as work:
        command = install(wheel, Path(work) / "venv", arguments.python)
        log_path = Path(work) / "serve.log"
        try:
            status = serve_and_probe(
                command, arguments.checkpoint, arguments.port, log_path
            )
        except SystemExit:
            _print_log_tail(log_path)
            raise
        if status != 0:
            _print_log_tail(log_path)
            raise SystemExit(
                f"the probe of the service installed from {wheel.name} failed"
            )
    print(
        f"{wheel.name} installed into a fresh environment, served {arguments.checkpoint}, and passed the probe"
    )


if __name__ == "__main__":
    main()
