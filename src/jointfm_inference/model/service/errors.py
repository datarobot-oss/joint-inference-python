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

"""API-facing error type for the packaged inference HTTP service."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum


class ServiceErrorCode(StrEnum):
    """The enumerated set of codes a rejection may carry.

    A client gates on these, so they are an enumeration rather than free text:
    a code that only ever appears in one service version is a code no caller
    can branch on. The catalog is part of the public service contract, so
    adding, renaming, or removing a code changes what every client sees.
    """

    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNSUPPORTED_HEAD_QUERY_COMBINATION = "UNSUPPORTED_HEAD_QUERY_COMBINATION"
    UNSUPPORTED_RETURN_MODE = "UNSUPPORTED_RETURN_MODE"
    INPUT_SIZE_EXCEEDED = "INPUT_SIZE_EXCEEDED"
    SCHEMA_VERSION_MISMATCH = "SCHEMA_VERSION_MISMATCH"
    MODEL_VERSION_MISMATCH = "MODEL_VERSION_MISMATCH"
    INTERNAL_ERROR = "INTERNAL_ERROR"


# The values an error object of each code carries as fields beside its
# message, so a client reads them instead of parsing the message text. A code
# not listed carries none. Each value is in the unit of the error's ``field``:
# a sample count for ``n_samples``, a byte count for ``Content-Length``.
ERROR_VALUE_FIELDS: Mapping[ServiceErrorCode, tuple[str, ...]] = {
    ServiceErrorCode.INPUT_SIZE_EXCEEDED: ("requested", "allowed"),
}


class ServiceError(Exception):
    """Request or startup error with an explicit API-facing code."""

    def __init__(
        self,
        code: ServiceErrorCode | str,
        message: str,
        *,
        status: int,
        field: str | None = None,
        values: Mapping[str, int] | None = None,
    ) -> None:
        """Initialize one API-facing error with its HTTP and field metadata.

        ``values`` holds exactly the value fields ``ERROR_VALUE_FIELDS`` names
        for ``code``; the error object on the wire carries them beside the
        message.

        Raises:
            ValueError: when ``code`` is outside the enumerated catalog, or
                ``values`` does not hold exactly the code's value fields. Both
                are programming errors rather than request errors: an
                unenumerated code would reach a client that cannot branch on
                it, and a missing value would break the documented contract.
        """
        super().__init__(message)
        error_code = ServiceErrorCode(code)
        carried = {} if values is None else dict(values)
        expected = ERROR_VALUE_FIELDS.get(error_code, ())
        if set(carried) != set(expected):
            raise ValueError(
                f"a {error_code.value} error carries exactly the values "
                f"{list(expected)}, got {sorted(carried)}"
            )
        self.code = error_code.value
        self.message = message
        self.status = status
        self.field = field
        self.values = carried
