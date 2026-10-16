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
