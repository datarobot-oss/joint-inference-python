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

"""HTTP request handler bound to a configured :class:`JointFMHTTPService`.

The handler bounds what one connection can cost: a body above the service's
``max_request_bytes`` is refused from its ``Content-Length`` before any of it
is read, and a connection that sends or accepts no byte for
``REQUEST_TIMEOUT_SECONDS`` is dropped, a stalled request body with a 408.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from typing import Any

from .constants import (
    DEFAULT_SCHEMA_VERSION,
    REQUEST_TIMEOUT_SECONDS,
    RESPONSE_WRITE_CHUNK_BYTES,
)
from .errors import ServiceError
from .http_service import JointFMHTTPService
from .serialization import encode_response
from .validation import _require_mapping


class JointFMRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler that delegates to a shared service instance."""

    service: JointFMHTTPService | None = None
    timeout = REQUEST_TIMEOUT_SECONDS

    def do_GET(self) -> None:  # noqa: N802
        """Serve container health metadata from the readiness endpoint."""
        if self.path != "/healthz":
            self._send_error(
                status=HTTPStatus.NOT_FOUND,
                code="VALIDATION_ERROR",
                message=f"Unknown endpoint: {self.path}",
            )
            return
        self._send_json(HTTPStatus.OK, _service(self).health_payload())

    def do_POST(self) -> None:  # noqa: N802
        """Serve one prediction request from the packaged inference transport."""
        if self.path != "/predict":
            self._send_error(
                status=HTTPStatus.NOT_FOUND,
                code="VALIDATION_ERROR",
                message=f"Unknown endpoint: {self.path}",
            )
            return

        try:
            body = self._read_json_body()
            payload = _require_mapping(body, field="request")
            response_body = _service(self).predict_response_bytes(payload)
        except ServiceError as error:
            self._send_error(
                status=error.status,
                code=error.code,
                message=error.message,
                field=error.field,
                values=error.values,
            )
            return
        except Exception as error:  # pragma: no cover - defensive HTTP boundary
            self._send_error(
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
                code="INTERNAL_ERROR",
                message=str(error),
            )
            return

        self._send_body(HTTPStatus.OK, response_body)

    def log_message(self, format: str, *args: object) -> None:
        """Suppress default access logging from the standard library handler."""
        del format, args

    def _read_json_body(self) -> Any:
        """Read and decode one JSON request body from the HTTP input stream.

        Raises:
            ServiceError: ``VALIDATION_ERROR`` (400) on ``Content-Length`` when
                it is missing, not an integer, or negative;
                ``INPUT_SIZE_EXCEEDED`` (413) on ``Content-Length`` when it
                names more than the service's ``max_request_bytes``, before
                any body byte is read; ``VALIDATION_ERROR`` (408) on
                ``request`` when the body stops arriving for the handler's
                timeout; ``VALIDATION_ERROR`` (400) on ``request`` when the
                body is not JSON.
        """
        length_header = self.headers.get("Content-Length")
        if length_header is None:
            raise ServiceError(
                "VALIDATION_ERROR",
                "Content-Length header is required",
                status=HTTPStatus.BAD_REQUEST,
                field="Content-Length",
            )
        try:
            content_length = int(length_header)
        except ValueError as error:
            raise ServiceError(
                "VALIDATION_ERROR",
                "Content-Length must be an integer",
                status=HTTPStatus.BAD_REQUEST,
                field="Content-Length",
            ) from error

        if content_length < 0:
            raise ServiceError(
                "VALIDATION_ERROR",
                f"Content-Length must be non-negative, got {content_length}",
                status=HTTPStatus.BAD_REQUEST,
                field="Content-Length",
            )
        max_request_bytes = _service(self).max_request_bytes
        if content_length > max_request_bytes:
            raise ServiceError(
                "INPUT_SIZE_EXCEEDED",
                (
                    f"request body of {content_length} bytes exceeds the "
                    f"service's max_request_bytes {max_request_bytes}; "
                    "/healthz advertises the cap"
                ),
                status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                field="Content-Length",
                values={"requested": content_length, "allowed": max_request_bytes},
            )
        try:
            raw_body = self.rfile.read(content_length)
        except TimeoutError as error:
            raise ServiceError(
                "VALIDATION_ERROR",
                (
                    f"request body stalled: no byte arrived for {self.timeout} s "
                    f"of the {content_length} bytes Content-Length announced"
                ),
                status=HTTPStatus.REQUEST_TIMEOUT,
                field="request",
            ) from error
        try:
            return json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as error:
            raise ServiceError(
                "VALIDATION_ERROR",
                f"Request body is not valid JSON: {error.msg}",
                status=HTTPStatus.BAD_REQUEST,
                field="request",
            ) from error

    def _send_error(
        self,
        *,
        status: int,
        code: str,
        message: str,
        field: str | None = None,
        values: Mapping[str, int] | None = None,
    ) -> None:
        """Send a structured JSON error payload using the public API schema.

        ``values`` are the error's value fields, written beside its message.
        """
        error_payload = {
            "schema_version": DEFAULT_SCHEMA_VERSION,
            "errors": [
                {
                    "code": code,
                    "message": message,
                    "field": field,
                    **({} if values is None else values),
                }
            ],
        }
        self._send_json(status, error_payload)

    def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
        """Serialize one response payload and write it back to the client."""
        self._send_body(status, encode_response(payload))

    def _send_body(self, status: int, body: bytes) -> None:
        """Write one already-encoded JSON response back to the client."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        view = memoryview(body)
        for start in range(0, len(view), RESPONSE_WRITE_CHUNK_BYTES):
            self.wfile.write(view[start : start + RESPONSE_WRITE_CHUNK_BYTES])


def _handler_class(service: JointFMHTTPService) -> type[JointFMRequestHandler]:
    """Bind the shared service instance onto a dedicated request-handler subclass."""

    class _BoundJointFMRequestHandler(JointFMRequestHandler):
        """Request handler class bound to a specific service instance."""

        pass

    _BoundJointFMRequestHandler.service = service
    return _BoundJointFMRequestHandler


def _service(handler: JointFMRequestHandler) -> JointFMHTTPService:
    """Return the configured service instance for one bound request handler."""
    if handler.service is None:
        raise RuntimeError("HTTP handler was created without a service instance")
    return handler.service
