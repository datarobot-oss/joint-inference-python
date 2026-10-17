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

"""The service process: clean shutdown on the signals that stop it."""

from __future__ import annotations

import os
import signal

import pytest

import jointfm_inference.model.service.server as service_server
from jointfm_inference.model.service import serve
from tests.support.models import tiny_model
from tests.support.service import mount_service


@pytest.mark.parametrize("stop_signal", service_server.STOP_SIGNALS)
def test_a_stop_signal_closes_the_server_and_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch, stop_signal: signal.Signals
) -> None:
    """``docker stop`` sends SIGTERM, which a container's first process ignores unhandled."""
    events: list[str] = []

    class SignaledServer:
        def __init__(self, address, handler_class) -> None:
            del address, handler_class

        def serve_forever(self) -> None:
            os.kill(os.getpid(), stop_signal)
            events.append("still serving")

        def server_close(self) -> None:
            events.append("closed")

    handlers_before = {s: signal.getsignal(s) for s in service_server.STOP_SIGNALS}
    monkeypatch.setattr(service_server, "ThreadingHTTPServer", SignaledServer)

    with pytest.raises(SystemExit) as exc_info:
        serve(mount_service(tiny_model()))

    assert exc_info.value.code == 0
    assert events == ["closed"]
    assert {
        s: signal.getsignal(s) for s in service_server.STOP_SIGNALS
    } == handlers_before
