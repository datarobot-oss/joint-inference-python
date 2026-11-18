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

"""The OpenAPI 3.1 document of the service, assembled from the wire schema.

OpenAPI (https://spec.openapis.org/oas/v3.1.0) is the standard description
of an HTTP API: its paths, operations, and payload schemas, the latter in
JSON Schema. The document is generated rather than written by hand so it
cannot drift from what the service does, and rendered deterministically so a
committed copy can be checked for staleness by comparing text.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any, Final

from ..constants import DEFAULT_PORT, DEFAULT_SCHEMA_VERSION
from .examples import HEALTH_REQUEST_EXAMPLE, example_requests
from .fields import ref
from .request import request_components
from .response import response_components

OPENAPI_VERSION: Final = "3.1.0"
OPENAPI_FILE_NAME: Final = "openapi.json"
JSON_MEDIA_TYPE: Final = "application/json"

# The statuses ``POST /predict`` answers with an ``ErrorResponse``.
PREDICT_ERROR_STATUSES: Final = {
    "400": "The request is malformed, asks for something the service does not "
    "serve, names a version it does not speak, or asks for more samples than "
    "`max_sample_count`.",
    "408": "The request body stopped arriving before `Content-Length` bytes came.",
    "413": "`Content-Length` exceeds `max_request_bytes`; the body is not read.",
    "500": "The service failed on a request it accepted.",
}

_DESCRIPTION: Final = (
    "The HTTP contract of a `jointfm-inference` service. `GET /healthz` reports "
    "what the service serves and the limits a request must keep; "
    "`POST /predict` answers forecast and condition queries over one table of "
    "rows. Every prediction request names the contract version in "
    "`schema_version`, and this document describes the version in `info.version`. "
    "An unknown path answers 404 with an `ErrorResponse`.\n\n"
    "The service generates this document from its own code "
    "(`jointfm-inference schema`)."
)


def _json_content(schema: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """Return a ``content`` map holding one JSON media type."""
    return {JSON_MEDIA_TYPE: {"schema": schema, **extra}}


def _request_examples() -> dict[str, dict[str, Any]]:
    """Return the request body's examples in OpenAPI's ``examples`` form."""
    examples: dict[str, dict[str, Any]] = {}
    for name, request in example_requests().items():
        if name == HEALTH_REQUEST_EXAMPLE:
            summary = "Health through the prediction route"
        else:
            summary = (
                f"`{request['query_mode']}` query, `{request['return_mode']}` answer"
            )
        examples[name] = {"summary": summary, "value": request}
    return examples


def openapi_document() -> dict[str, Any]:
    """Return the OpenAPI 3.1 document of the service's wire contract."""
    error_responses = {
        status: {
            "description": description,
            "content": _json_content(ref("ErrorResponse")),
        }
        for status, description in PREDICT_ERROR_STATUSES.items()
    }
    return {
        "openapi": OPENAPI_VERSION,
        "info": {
            "title": "JointFM inference service",
            "version": DEFAULT_SCHEMA_VERSION,
            "description": _DESCRIPTION,
        },
        "servers": [
            {
                "url": f"http://127.0.0.1:{DEFAULT_PORT}",
                "description": "A local service on its default port.",
            }
        ],
        "paths": {
            "/healthz": {
                "get": {
                    "operationId": "health",
                    "summary": "Report what the service serves",
                    "responses": {
                        "200": {
                            "description": "The service is ready.",
                            "content": _json_content(ref("HealthResponse")),
                        }
                    },
                }
            },
            "/predict": {
                "post": {
                    "operationId": "predict",
                    "summary": "Answer a forecast or condition query",
                    "description": (
                        "Answers a `PredictRequest` with a `PredictResponse`, "
                        "and a `HealthRequest` with the `HealthResponse` of "
                        "`GET /healthz`."
                    ),
                    "requestBody": {
                        "required": True,
                        "content": _json_content(
                            ref("Request"), examples=_request_examples()
                        ),
                    },
                    "responses": {
                        "200": {
                            "description": "The answer.",
                            "content": _json_content(
                                {
                                    "oneOf": [
                                        ref("PredictResponse"),
                                        ref("HealthResponse"),
                                    ]
                                }
                            ),
                        },
                        **error_responses,
                    },
                }
            },
        },
        "components": {"schemas": {**request_components(), **response_components()}},
    }


def render_openapi() -> str:
    """Return the document as the text ``jointfm-inference schema`` writes.

    Keys keep the order the builders give them and the text ends in a
    newline, so equal documents render to equal text.
    """
    return json.dumps(openapi_document(), indent=2, ensure_ascii=False) + "\n"


def packaged_openapi() -> str:
    """Return the text of the ``openapi.json`` shipped inside the package."""
    return resources.files(__package__).joinpath(OPENAPI_FILE_NAME).read_text("utf-8")
