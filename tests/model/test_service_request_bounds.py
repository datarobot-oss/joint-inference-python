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

"""The HTTP handler's bounds on what one request and one connection may cost."""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Buffer, Iterator
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from io import BufferedIOBase, BytesIO
from typing import cast

import pytest

import jointfm_inference.model.service.http_handler as http_handler
from jointfm_inference.model.service import JointFMRequestHandler
from jointfm_inference.model.service.http_handler import _handler_class
from tests.support.http_handler import (
    GENEROUS_REQUEST_CAP,
    FakeHTTPHandler,
    capped_service,
    response_body,
)


class _UnreadableBody(BytesIO):
    """A request body that fails the test if the handler reads any of it."""

    def read(self, size: int | None = -1) -> bytes:
        raise AssertionError(f"the handler read {size} bytes of a refused body")


def _posting_handler(max_request_bytes: int, content_length: str) -> FakeHTTPHandler:
    """Return a ``/predict`` handler whose body must not be read."""
    handler = FakeHTTPHandler()
    handler.service = capped_service(max_request_bytes)
    handler.path = "/predict"
    handler.headers["Content-Length"] = content_length
    handler.rfile = _UnreadableBody()
    return handler


def test_a_negative_content_length_is_refused_like_a_missing_one() -> None:
    """A negative length answers 400 on ``Content-Length`` before any read."""
    handler = _posting_handler(GENEROUS_REQUEST_CAP, "-1")

    handler.do_POST()

    assert handler.response_codes == [HTTPStatus.BAD_REQUEST]
    [error] = response_body(handler)["errors"]
    assert error["code"] == "VALIDATION_ERROR"
    assert error["field"] == "Content-Length"
    assert "non-negative" in error["message"]


def test_a_body_above_the_cap_is_refused_unread_with_413() -> None:
    """One byte over ``max_request_bytes`` answers 413 naming both sizes, unread."""
    handler = _posting_handler(64, "65")

    handler.do_POST()

    assert handler.response_codes == [HTTPStatus.REQUEST_ENTITY_TOO_LARGE]
    [error] = response_body(handler)["errors"]
    assert error["code"] == "INPUT_SIZE_EXCEEDED"
    assert error["field"] == "Content-Length"
    assert "65" in error["message"]
    assert "max_request_bytes 64" in error["message"]


def test_a_body_at_the_cap_is_read_and_answered() -> None:
    """The cap is inclusive: a body of exactly ``max_request_bytes`` is served."""
    raw_body = b'{"value": 3}'
    handler = FakeHTTPHandler()
    handler.service = capped_service(len(raw_body))
    handler.path = "/predict"
    handler.headers["Content-Length"] = str(len(raw_body))
    handler.rfile = BytesIO(raw_body)

    handler.do_POST()

    assert handler.response_codes == [HTTPStatus.OK]
    assert response_body(handler) == {"value": 3}


class _RecordingWriter(BufferedIOBase):
    """Response stream double that keeps each write separately."""

    def __init__(self) -> None:
        super().__init__()
        self.writes: list[bytes] = []

    def write(self, data: Buffer, /) -> int:
        chunk = bytes(data)
        self.writes.append(chunk)
        return len(chunk)


def test_a_response_is_written_in_chunks_so_the_timeout_bounds_stalls_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each chunk is its own socket send, so a large response to a slow reader completes."""
    monkeypatch.setattr(http_handler, "RESPONSE_WRITE_CHUNK_BYTES", 4)
    handler = FakeHTTPHandler()
    writer = _RecordingWriter()
    handler.wfile = writer

    handler._send_body(HTTPStatus.OK, b"0123456789")

    assert writer.writes == [b"0123", b"4567", b"89"]


_LIVE_HANDLER_TIMEOUT_SECONDS = 0.3
_LIVE_REQUEST_CAP = 1024
# How long a test client waits for an answer: well above the handler timeout,
# well below a test hang.
_LIVE_CLIENT_TIMEOUT_SECONDS = 10.0


@pytest.fixture
def live_server(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[str, int]]:
    """Serve a capped echo service on a loopback port with a short handler timeout."""
    monkeypatch.setattr(JointFMRequestHandler, "timeout", _LIVE_HANDLER_TIMEOUT_SECONDS)
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), _handler_class(capped_service(_LIVE_REQUEST_CAP))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield cast(tuple[str, int], server.server_address[:2])
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _read_until_closed(connection: socket.socket) -> bytes:
    """Return everything the server sends before it closes the connection."""
    chunks = []
    while chunk := connection.recv(65536):
        chunks.append(chunk)
    return b"".join(chunks)


def _post_head(content_length: int) -> bytes:
    """Return the request line and headers of a ``/predict`` POST."""
    return (
        "POST /predict HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {content_length}\r\n\r\n"
    ).encode("ascii")


def test_a_stalled_body_is_answered_408_and_its_connection_closed(
    live_server: tuple[str, int],
) -> None:
    """A client that stops sending mid-body releases its thread after the handler timeout."""
    with socket.create_connection(
        live_server, timeout=_LIVE_CLIENT_TIMEOUT_SECONDS
    ) as connection:
        connection.sendall(_post_head(100) + b'{"value":')
        response = _read_until_closed(connection)

    head, _, body = response.partition(b"\r\n\r\n")
    assert head.startswith(b"HTTP/1.0 408")
    [error] = json.loads(body)["errors"]
    assert error["code"] == "VALIDATION_ERROR"
    assert error["field"] == "request"


def test_an_oversized_body_is_refused_before_it_is_sent(
    live_server: tuple[str, int],
) -> None:
    """The 413 arrives from the headers alone, without waiting for any body byte."""
    with socket.create_connection(
        live_server, timeout=_LIVE_CLIENT_TIMEOUT_SECONDS
    ) as connection:
        connection.sendall(_post_head(_LIVE_REQUEST_CAP + 1))
        response = _read_until_closed(connection)

    head, _, body = response.partition(b"\r\n\r\n")
    assert head.startswith(b"HTTP/1.0 413")
    [error] = json.loads(body)["errors"]
    assert error["code"] == "INPUT_SIZE_EXCEEDED"
    assert error["field"] == "Content-Length"
