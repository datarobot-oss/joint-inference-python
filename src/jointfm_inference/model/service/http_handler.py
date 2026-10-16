"""HTTP request handler bound to a configured :class:`JointFMHTTPService`."""

from __future__ import annotations

import json
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from typing import Any

from .constants import DEFAULT_SCHEMA_VERSION
from .errors import ServiceError
from .http_service import JointFMHTTPService
from .serialization import encode_response
from .validation import _require_mapping


class JointFMRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler that delegates to a shared service instance."""

    service: JointFMHTTPService | None = None

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
        """Read and decode one JSON request body from the HTTP input stream."""
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

        raw_body = self.rfile.read(content_length)
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
    ) -> None:
        """Send a structured JSON error payload using the public API schema."""
        error_payload = {
            "schema_version": DEFAULT_SCHEMA_VERSION,
            "errors": [
                {
                    "code": code,
                    "message": message,
                    "field": field,
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
        self.wfile.write(body)


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
