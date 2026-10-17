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

"""Checkpoint download and reuse before the service loads its weights."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError

import pytest

from jointfm_inference.model.checkpoint import JOINTFM_1_GENERATION
from jointfm_inference.model.service import create_service_from_env
from jointfm_inference.model.service.checkpoint_source import (
    CHECKPOINT_FILE_MODE,
    download_checkpoint,
    file_sha256,
    resolve_checkpoint,
)
from jointfm_inference.model.service.config import (
    CHECKPOINT_PATH_ENV,
    CHECKPOINT_SHA256_ENV,
    CHECKPOINT_URL_ENV,
    CHECKPOINT_VERSION_ENV,
    DEVICE_ENV,
    PRECISION_ENV,
)
from tests.support.models import write_checkpoint
from tests.support.service import (
    SERVICE_PRECISION,
    clear_service_environment,
    make_service_config,
)

BODY_PATH = "/weights/jointfm.pt"


@dataclass
class _Origin:
    """A local HTTP origin serving one body and recording each request's headers."""

    url: str
    body: bytes = b""
    authorizations: list[str | None] = field(default_factory=list)


@pytest.fixture
def origin() -> Iterator[_Origin]:
    served = _Origin(url="")

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            served.authorizations.append(self.headers.get("Authorization"))
            if self.path != BODY_PATH:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Length", str(len(served.body)))
            self.end_headers()
            self.wfile.write(served.body)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    host, port = server.server_address[:2]
    served.url = f"http://{host!s}:{port}{BODY_PATH}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield served
    server.shutdown()
    server.server_close()
    thread.join()


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def test_download_writes_the_verified_body_and_sends_the_token(
    origin: _Origin, tmp_path: Path
) -> None:
    origin.body = b"weights" * 1000
    target = tmp_path / "nested" / "jointfm.pt"

    download_checkpoint(origin.url, target, sha256=_sha256(origin.body), token="t0k")

    assert target.read_bytes() == origin.body
    assert target.stat().st_mode & 0o777 == CHECKPOINT_FILE_MODE
    assert origin.authorizations == ["Bearer t0k"]
    assert list(target.parent.iterdir()) == [target]


def test_download_without_a_token_sends_no_authorization(
    origin: _Origin, tmp_path: Path
) -> None:
    origin.body = b"weights"
    download_checkpoint(
        origin.url, tmp_path / "c.pt", sha256=_sha256(origin.body), token=None
    )
    assert origin.authorizations == [None]


def test_a_digest_mismatch_leaves_no_file_behind(
    origin: _Origin, tmp_path: Path
) -> None:
    """A corrupted download must not become a file a later start would load."""
    origin.body = b"tampered"
    with pytest.raises(RuntimeError, match="expected " + "0" * 64):
        download_checkpoint(origin.url, tmp_path / "c.pt", sha256="0" * 64, token=None)
    assert list(tmp_path.iterdir()) == []


def test_a_failed_request_leaves_no_file_behind(
    origin: _Origin, tmp_path: Path
) -> None:
    with pytest.raises(HTTPError):
        download_checkpoint(
            origin.url + ".missing", tmp_path / "c.pt", sha256="0" * 64, token=None
        )
    assert list(tmp_path.iterdir()) == []


def test_resolve_requires_a_present_file_without_a_download_source(
    tmp_path: Path,
) -> None:
    config = make_service_config(checkpoint_path=str(tmp_path / "missing.pt"))
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        resolve_checkpoint(config)

    present = tmp_path / "present.pt"
    present.write_bytes(b"weights")
    assert (
        resolve_checkpoint(make_service_config(checkpoint_path=str(present))) == present
    )


def test_resolve_downloads_once_and_then_reuses_the_matching_file(
    origin: _Origin, tmp_path: Path
) -> None:
    origin.body = b"weights"
    config = make_service_config(
        checkpoint_path=str(tmp_path / "c.pt"),
        checkpoint_url=origin.url,
        checkpoint_sha256=_sha256(origin.body),
    )

    assert resolve_checkpoint(config).read_bytes() == origin.body
    assert resolve_checkpoint(config).read_bytes() == origin.body
    assert len(origin.authorizations) == 1


def test_resolve_refuses_a_present_file_of_another_digest(
    origin: _Origin, tmp_path: Path
) -> None:
    """A changed download source must never serve the weights already on disk."""
    stale = tmp_path / "c.pt"
    stale.write_bytes(b"older weights")
    origin.body = b"newer weights"
    config = make_service_config(
        checkpoint_path=str(stale),
        checkpoint_url=origin.url,
        checkpoint_sha256=_sha256(origin.body),
    )

    with pytest.raises(RuntimeError, match="remove the file"):
        resolve_checkpoint(config)
    assert stale.read_bytes() == b"older weights"
    assert origin.authorizations == []


def test_file_sha256_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "c.pt"
    path.write_bytes(b"x" * 3_000_000)
    assert file_sha256(path) == _sha256(path.read_bytes())


def test_service_startup_downloads_and_serves_the_configured_checkpoint(
    origin: _Origin, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pt"
    write_checkpoint(source)
    origin.body = source.read_bytes()
    target = tmp_path / "models" / "jointfm.pt"
    clear_service_environment(monkeypatch)
    monkeypatch.setenv(CHECKPOINT_PATH_ENV, str(target))
    monkeypatch.setenv(CHECKPOINT_VERSION_ENV, "example-1")
    monkeypatch.setenv(CHECKPOINT_URL_ENV, origin.url)
    monkeypatch.setenv(CHECKPOINT_SHA256_ENV, _sha256(origin.body))
    monkeypatch.setenv(DEVICE_ENV, "cpu")
    monkeypatch.setenv(PRECISION_ENV, SERVICE_PRECISION)

    health = create_service_from_env().health_payload()

    assert health["generation"] == JOINTFM_1_GENERATION
    assert health["checkpoint_path"] == str(target)
    assert target.read_bytes() == origin.body
