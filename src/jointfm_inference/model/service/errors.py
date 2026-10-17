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


class ServiceError(Exception):
    """Request or startup error with an explicit API-facing code."""

    def __init__(
        self,
        code: ServiceErrorCode | str,
        message: str,
        *,
        status: int,
        field: str | None = None,
    ) -> None:
        """Initialize one API-facing error with its HTTP and field metadata.

        Raises:
            ValueError: when ``code`` is outside the enumerated catalog, which
                is a programming error rather than a request error: an
                unenumerated code would reach a client that cannot branch on
                it.
        """
        super().__init__(message)
        self.code = ServiceErrorCode(code).value
        self.message = message
        self.status = status
        self.field = field
