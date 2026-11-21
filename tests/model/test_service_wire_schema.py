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

"""The OpenAPI document against what a running service accepts and sends.

A service on the tests' small checkpoint answers every example request of the
document, its health endpoint, and requests it refuses; every payload it sends
must validate against the schema the document gives for that path and status.
"""

from __future__ import annotations

import http.client
import json
import threading
from collections.abc import Iterator, Mapping
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import jsonschema
from jsonschema.protocols import Validator
import pytest

from jointfm_inference.cli import main
from jointfm_inference.model.inference.jointfm_1_backend import DECODING_STRATEGY
from jointfm_inference.model.service import JointFMHTTPService, ServiceError
from jointfm_inference.model.service.http_handler import _handler_class
from jointfm_inference.model.service.wire_schema import (
    HEALTH_REQUEST_EXAMPLE,
    OPENAPI_FILE_NAME,
    example_requests,
    openapi_document,
    packaged_openapi,
    render_openapi,
)
from jointfm_inference.model.service.wire_schema.fields import described_values
from tests.support.http_handler import FakeHTTPHandler, response_body
from tests.support.models import tiny_model
from tests.support.service import SERVICE_MAX_SAMPLE_COUNT, mount_service

_DOCUMENT = openapi_document()
_PREDICT = _DOCUMENT["paths"]["/predict"]["post"]
# A cap above every example request and below the oversized one the tests send.
_LIVE_REQUEST_CAP = 1 << 16
_CLIENT_TIMEOUT_SECONDS = 30.0


def _validator(schema: Mapping[str, Any]) -> Validator:
    """Return a validator for ``schema``, resolving the document's references."""
    return jsonschema.Draft202012Validator(
        {**schema, "components": _DOCUMENT["components"]}
    )


def _component(name: str) -> Validator:
    """Return a validator for one schema under ``components.schemas``."""
    return _validator({"$ref": f"#/components/schemas/{name}"})


def _predict_response(status: int) -> Validator:
    """Return a validator for the body ``POST /predict`` documents for ``status``."""
    responses = _PREDICT["responses"]
    assert str(status) in responses, f"POST /predict does not document {status}"
    return _validator(responses[str(status)]["content"]["application/json"]["schema"])


_REQUEST = _validator(_PREDICT["requestBody"]["content"]["application/json"]["schema"])


@pytest.fixture(scope="module")
def service_address() -> Iterator[tuple[str, int]]:
    """Serve the tiny model on a loopback port for the whole module."""
    service = mount_service(tiny_model(), max_request_bytes=_LIVE_REQUEST_CAP)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_class(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield cast(tuple[str, int], server.server_address[:2])
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _exchange(
    address: tuple[str, int],
    method: str,
    path: str,
    body: bytes | None = None,
    headers: Mapping[str, str] | None = None,
) -> tuple[int, Any]:
    """Send one request and return the status and decoded JSON body."""
    connection = http.client.HTTPConnection(*address, timeout=_CLIENT_TIMEOUT_SECONDS)
    try:
        connection.request(method, path, body=body, headers=dict(headers or {}))
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def _post(address: tuple[str, int], payload: Any) -> tuple[int, Any]:
    """POST ``payload`` as JSON to ``/predict``."""
    return _exchange(
        address,
        "POST",
        "/predict",
        json.dumps(payload).encode("utf-8"),
        {"Content-Type": "application/json"},
    )


def _assert_valid(validator: Validator, instance: Any) -> None:
    """Fail with every validation error at once, each with its path."""
    errors = [
        f"{'/'.join(map(str, error.absolute_path))}: {error.message}"
        for error in validator.iter_errors(instance)
    ]
    assert not errors, errors


def test_the_packaged_document_is_the_generated_one() -> None:
    """The ``openapi.json`` shipped as package data is current."""
    assert packaged_openapi() == render_openapi()


@pytest.mark.parametrize("name", sorted(_DOCUMENT["components"]["schemas"]))
def test_every_component_is_valid_json_schema(name: str) -> None:
    jsonschema.Draft202012Validator.check_schema(
        _DOCUMENT["components"]["schemas"][name]
    )


@pytest.mark.parametrize("name", sorted(example_requests()))
def test_every_example_is_accepted_and_answered_as_documented(
    service_address: tuple[str, int], name: str
) -> None:
    request = example_requests()[name]
    _assert_valid(_REQUEST, request)

    status, response = _post(service_address, request)

    assert status == HTTPStatus.OK, response
    _assert_valid(_predict_response(status), response)
    if name == HEALTH_REQUEST_EXAMPLE:
        _assert_valid(_component("HealthResponse"), response)
    else:
        _assert_valid(_component("PredictResponse"), response)
        assert response["query_mode"] == request["query_mode"]
        assert response["return_mode"] == request["return_mode"]


def test_healthz_is_answered_as_documented(service_address: tuple[str, int]) -> None:
    status, response = _exchange(service_address, "GET", "/healthz")

    assert status == HTTPStatus.OK
    schema = _DOCUMENT["paths"]["/healthz"]["get"]["responses"]["200"]
    _assert_valid(_validator(schema["content"]["application/json"]["schema"]), response)


def _example(**changes: Any) -> dict[str, Any]:
    """Return the forecast mean example with ``changes`` applied."""
    request = example_requests()["forecast_mean"]
    request.update(changes)
    return request


@pytest.mark.parametrize(
    ("request_payload", "code", "field"),
    [
        (_example(horizon=3), "VALIDATION_ERROR", "horizon"),
        ([], "VALIDATION_ERROR", "request"),
        (_example(schema_version="v0"), "SCHEMA_VERSION_MISMATCH", "schema_version"),
        (_example(model_version="other"), "MODEL_VERSION_MISMATCH", "model_version"),
        (
            _example(n_samples=SERVICE_MAX_SAMPLE_COUNT + 1),
            "INPUT_SIZE_EXCEEDED",
            "n_samples",
        ),
    ],
)
def test_a_refusal_is_answered_as_documented(
    service_address: tuple[str, int], request_payload: Any, code: str, field: str
) -> None:
    status, response = _post(service_address, request_payload)

    assert status == HTTPStatus.BAD_REQUEST
    _assert_valid(_predict_response(status), response)
    [error] = response["errors"]
    assert (error["code"], error["field"]) == (code, field)


def test_a_sample_cap_refusal_carries_both_sample_counts(
    service_address: tuple[str, int],
) -> None:
    """The counts the message names are fields, so a client need not parse text."""
    requested = SERVICE_MAX_SAMPLE_COUNT + 1

    _, response = _post(service_address, _example(n_samples=requested))

    [error] = response["errors"]
    assert (error["requested"], error["allowed"]) == (
        requested,
        SERVICE_MAX_SAMPLE_COUNT,
    )


def test_an_oversized_body_is_refused_as_documented(
    service_address: tuple[str, int],
) -> None:
    """The 413 names the announced size and the cap, from the headers alone."""
    oversized = _LIVE_REQUEST_CAP + 1
    connection = http.client.HTTPConnection(
        *service_address, timeout=_CLIENT_TIMEOUT_SECONDS
    )
    try:
        connection.putrequest("POST", "/predict")
        connection.putheader("Content-Length", str(oversized))
        connection.endheaders()
        response = connection.getresponse()
        status, body = response.status, json.loads(response.read())
    finally:
        connection.close()

    assert status == HTTPStatus.REQUEST_ENTITY_TOO_LARGE
    _assert_valid(_predict_response(status), body)
    [error] = body["errors"]
    assert (error["code"], error["requested"], error["allowed"]) == (
        "INPUT_SIZE_EXCEEDED",
        oversized,
        _LIVE_REQUEST_CAP,
    )


def test_an_unknown_path_answers_an_error_response(
    service_address: tuple[str, int],
) -> None:
    status, response = _exchange(service_address, "GET", "/metrics")

    assert status == HTTPStatus.NOT_FOUND
    _assert_valid(_component("ErrorResponse"), response)


class _StalledBody(BytesIO):
    """A request body whose next byte never arrives."""

    def read(self, size: int | None = -1) -> bytes:
        raise TimeoutError


class _FailingService:
    """A service double that fails on every request it accepts."""

    max_request_bytes = _LIVE_REQUEST_CAP

    def predict_response_bytes(self, payload: Mapping[str, Any]) -> bytes:
        del payload
        raise RuntimeError("the forward failed")


@pytest.mark.parametrize(
    ("rfile", "status"),
    [
        (_StalledBody(), HTTPStatus.REQUEST_TIMEOUT),
        (BytesIO(b"{}"), HTTPStatus.INTERNAL_SERVER_ERROR),
    ],
)
def test_a_stall_and_a_failure_are_answered_as_documented(
    rfile: BytesIO, status: HTTPStatus
) -> None:
    handler = FakeHTTPHandler()
    handler.service = cast(JointFMHTTPService, _FailingService())
    handler.path = "/predict"
    handler.headers["Content-Length"] = "2"
    handler.rfile = rfile

    handler.do_POST()

    assert handler.response_codes == [status]
    _assert_valid(_predict_response(status), response_body(handler))


def test_a_request_field_the_service_refuses_fails_the_document() -> None:
    errors = list(_REQUEST.iter_errors(_example(horizon=3)))

    assert errors


def test_the_document_explains_the_decoding_strategy_a_jointfm_1_service_reports() -> (
    None
):
    """The description names the backend's value, which it cannot import."""
    health = _DOCUMENT["components"]["schemas"]["HealthResponse"]["properties"]

    assert f"`{DECODING_STRATEGY}`" in health["decoding_strategy"]["description"]


@pytest.mark.parametrize(
    "error",
    [
        {"code": "INPUT_SIZE_EXCEEDED", "message": "too large", "field": "n_samples"},
        {
            "code": "VALIDATION_ERROR",
            "message": "malformed",
            "field": "columns",
            "requested": 3,
            "allowed": 2,
        },
    ],
)
def test_the_document_ties_error_values_to_their_codes(error: dict[str, Any]) -> None:
    assert list(_component("Error").iter_errors(error))


@pytest.mark.parametrize(
    ("code", "values"),
    [
        ("INPUT_SIZE_EXCEEDED", None),
        ("INPUT_SIZE_EXCEEDED", {"requested": 2}),
        ("VALIDATION_ERROR", {"requested": 2, "allowed": 1}),
    ],
)
def test_an_error_carries_exactly_the_values_of_its_code(
    code: str, values: dict[str, int] | None
) -> None:
    with pytest.raises(ValueError, match="carries exactly the values"):
        ServiceError(code, "refused", status=HTTPStatus.BAD_REQUEST, values=values)


@pytest.mark.parametrize(
    "descriptions", [{"a": "A."}, {"a": "A.", "b": "B.", "c": "C."}]
)
def test_a_vocabulary_must_describe_exactly_the_served_values(
    descriptions: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match="undescribed"):
        described_values(descriptions, owned={"a", "b"}, vocabulary="test")


def test_schema_writes_the_document_and_check_accepts_it(tmp_path: Path) -> None:
    output = tmp_path / OPENAPI_FILE_NAME

    main(["schema", "--output", str(output)])
    main(["schema", "--check", "--output", str(output)])

    assert output.read_text(encoding="utf-8") == render_openapi()


@pytest.mark.parametrize("stale", [None, "{}\n"])
def test_schema_check_fails_on_a_missing_or_stale_file(
    tmp_path: Path, stale: str | None
) -> None:
    output = tmp_path / OPENAPI_FILE_NAME
    if stale is not None:
        output.write_text(stale, encoding="utf-8")

    with pytest.raises(SystemExit, match="rewrite it with"):
        main(["schema", "--check", "--output", str(output)])
