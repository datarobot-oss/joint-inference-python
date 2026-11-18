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
