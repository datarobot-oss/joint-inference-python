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

"""A request-handler double that records what the HTTP handler writes."""

from __future__ import annotations

import json
import re
from http import HTTPStatus
from http.client import HTTPMessage
from io import BytesIO
from types import SimpleNamespace
from typing import Any, cast

from jointfm_inference.model.service import (
    JointFMHTTPService,
    JointFMRequestHandler,
    encode_response,
)

# A request cap far above every body the handler tests send.
GENEROUS_REQUEST_CAP = 1 << 20
HANDLER_LOGGER = "jointfm_inference.model.service.http_handler"
# The whole message of a server failure: a reference and nothing else.
INTERNAL_ERROR_MESSAGE = re.compile(
    r"internal server error \(reference ([0-9a-f]{32})\)"
)
# What a server failure may carry and a client must never see.
INTERNAL_DETAIL = "CUDA out of memory on cuda:3 while reading /models/jointfm.pt"


class FakeHTTPHandler(JointFMRequestHandler):
    """Minimal request-handler double that records response writes."""

    def __init__(self) -> None:
        self.headers = HTTPMessage()
        self.rfile = BytesIO()
        self.wfile = BytesIO()
        self.path = "/"
        self.response_codes: list[int] = []
        self.sent_headers: list[tuple[str, str]] = []
        self.ended = False

    def send_response(self, code: int, message: str | None = None) -> None:
        del message
        self.response_codes.append(code)

    def send_header(self, keyword: str, value: str) -> None:
        self.sent_headers.append((keyword, value))

    def end_headers(self) -> None:
        self.ended = True


def response_body(handler: FakeHTTPHandler) -> dict[str, Any]:
    """Decode the JSON body one fake handler wrote."""
    return json.loads(cast(BytesIO, handler.wfile).getvalue())


def capped_service(max_request_bytes: int) -> JointFMHTTPService:
    """Return a service double that echoes a request under ``max_request_bytes``."""
    return cast(
        JointFMHTTPService,
        SimpleNamespace(
            max_request_bytes=max_request_bytes,
            predict_response_bytes=encode_response,
        ),
    )


def failing_handler(path: str, failing_call: str) -> FakeHTTPHandler:
    """Return a handler for ``path`` whose service fails with ``INTERNAL_DETAIL``.

    ``failing_call`` names the service method that raises: ``health_payload``
    or ``predict_response_bytes``. The handler holds a small JSON body, so a
    ``POST`` reaches the service.
    """

    def fail(*_args: object) -> bytes:
        raise RuntimeError(INTERNAL_DETAIL)

    handler = FakeHTTPHandler()
    handler.service = cast(
        JointFMHTTPService,
        SimpleNamespace(max_request_bytes=GENEROUS_REQUEST_CAP, **{failing_call: fail}),
    )
    handler.path = path
    raw_body = b'{"value": 3}'
    handler.headers["Content-Length"] = str(len(raw_body))
    handler.rfile = BytesIO(raw_body)
    return handler


def internal_error_reference(handler: FakeHTTPHandler) -> str:
    """Return the reference of the one ``INTERNAL_ERROR`` a handler answered.

    Fails unless the answer is a 500 whose single error carries no field and a
    message that names the reference and nothing else.
    """
    assert handler.response_codes == [HTTPStatus.INTERNAL_SERVER_ERROR]
    [error] = response_body(handler)["errors"]
    assert error["code"] == "INTERNAL_ERROR"
    assert error["field"] is None
    match = INTERNAL_ERROR_MESSAGE.fullmatch(error["message"])
    assert match is not None, error["message"]
    return match.group(1)
