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

"""The tiny service the client captures its fixtures from."""

from __future__ import annotations

from pathlib import Path

import pytest

from jointfm_inference.model.service.server import create_service_from_env
from jointfm_inference.model.service.wire_schema import (
    HEALTH_REQUEST_EXAMPLE,
    example_requests,
)
from tests.support.serve_tiny import TINY_SERVICE_DEVICE, tiny_service_environment


def test_the_tiny_environment_serves_every_example_on_the_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in tiny_service_environment(tmp_path).items():
        monkeypatch.setenv(name, value)

    service = create_service_from_env()

    assert service.health_payload()["device"] == TINY_SERVICE_DEVICE
    for name, request in example_requests().items():
        if name != HEALTH_REQUEST_EXAMPLE:
            assert service.predict(request)["errors"] == [], name
