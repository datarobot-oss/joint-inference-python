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

"""The service's wire contract: every field, vocabulary, and example, as data.

One package holds every field of ``POST /predict`` and ``GET /healthz``
requests and responses with its JSON type, enumeration, requiredness, and
description, each return mode's response as its own variant, the error
object, and one example request per query mode and return mode. The request
parsers read their key sets and vocabularies from here, and
:func:`openapi_document` assembles the OpenAPI 3.1 document from the same
data, so the document cannot list a field the service refuses or omit one it
reads.

- ``vocabulary``: request types, query and return modes, time-index modes,
  condition kinds, roles, precisions, and error codes.
- ``request`` and ``response``: the fields of every object on the wire.
- ``examples``: the example requests.
- ``document``: the OpenAPI document and the ``openapi.json`` shipped with the
  package.

The schema is plain data rather than pydantic models because the runner keeps
pydantic out of its runtime dependencies.
"""

from .document import (
    OPENAPI_FILE_NAME,
    openapi_document,
    packaged_openapi,
    render_openapi,
)
from .examples import HEALTH_REQUEST_EXAMPLE, example_name, example_requests
from .request import (
    COLUMN_DESCRIPTOR_FIELD_NAMES,
    EQUALITY_CONDITION_FIELD_NAMES,
    INTERVAL_CONDITION_FIELD_NAMES,
    QUERY_TIME_INDICES_FIELD,
    REQUEST_FIELDS,
)
from .vocabulary import (
    CONDITION_KIND_EQUALITY,
    CONDITION_KIND_INTERVAL,
    CONDITION_KINDS,
    DEFAULT_QUERY_MODE,
    DEFAULT_REQUEST_TYPE,
    DEFAULT_RETURN_MODE,
    DEFAULT_ROLE,
    QUERY_MODE_CONDITION,
    QUERY_MODE_FORECAST,
    QUERY_MODES,
    REQUEST_TYPE_HEALTH,
    REQUEST_TYPE_PREDICT,
    REQUEST_TYPES,
    RETURN_MODES,
    SERVED_MODALITY,
)

__all__ = [
    "COLUMN_DESCRIPTOR_FIELD_NAMES",
    "CONDITION_KINDS",
    "CONDITION_KIND_EQUALITY",
    "CONDITION_KIND_INTERVAL",
    "DEFAULT_QUERY_MODE",
    "DEFAULT_REQUEST_TYPE",
    "DEFAULT_RETURN_MODE",
    "DEFAULT_ROLE",
    "EQUALITY_CONDITION_FIELD_NAMES",
    "HEALTH_REQUEST_EXAMPLE",
    "INTERVAL_CONDITION_FIELD_NAMES",
    "OPENAPI_FILE_NAME",
    "QUERY_MODES",
    "QUERY_MODE_CONDITION",
    "QUERY_MODE_FORECAST",
    "QUERY_TIME_INDICES_FIELD",
    "REQUEST_FIELDS",
    "REQUEST_TYPES",
    "REQUEST_TYPE_HEALTH",
    "REQUEST_TYPE_PREDICT",
    "RETURN_MODES",
    "SERVED_MODALITY",
    "example_name",
    "example_requests",
    "openapi_document",
    "packaged_openapi",
    "render_openapi",
]
