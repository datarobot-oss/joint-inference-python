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

"""Process entrypoints for the packaged inference HTTP server.

``ThreadingHTTPServer`` is imported at module scope so tests can patch the
symbol at ``jointfm_inference.model.service.server.ThreadingHTTPServer`` for transport-
level integration without binding a real socket.
"""

from __future__ import annotations

import signal
import sys
from http.server import ThreadingHTTPServer
from types import FrameType

from jointfm_inference.model.inference import JointFMInferenceSession

from .checkpoint_source import resolve_checkpoint
from .config import InferenceServiceConfig
from .http_handler import _handler_class
from .http_service import JointFMHTTPService

STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)


def create_service_from_env() -> JointFMHTTPService:
    """Load the configured checkpoint and return the ready HTTP service.

    The checkpoint is downloaded first when the environment names a download
    source and the file is not yet at the configured path.
    """
    config = InferenceServiceConfig.from_env()
    checkpoint_path = resolve_checkpoint(config)

    session = JointFMInferenceSession.from_checkpoint(
        checkpoint_path,
        map_location=config.map_location,
        device=config.device,
        precision=config.precision,
    )
    return JointFMHTTPService(config=config, session=session)


def _stop_serving(signum: int, frame: FrameType | None) -> None:
    """Leave ``serve_forever`` with a clean exit when the process is told to stop."""
    del frame
    print(f"Stopping on {signal.Signals(signum).name}", file=sys.stderr)
    raise SystemExit(0)


def serve(service: JointFMHTTPService) -> None:
    """Run the threaded HTTP server until SIGINT or SIGTERM stops it.

    Both signals end the process with status 0 after closing the server.
    SIGTERM needs its own handler because a container's first process
    ignores signals it does not handle, so ``docker stop`` would otherwise
    wait for its timeout and kill the process. The previous handlers are
    restored on return.
    """
    handler = _handler_class(service)
    address = (service._config.bind_host, service._config.port)
    server = ThreadingHTTPServer(address, handler)
    previous_handlers = {
        stop_signal: signal.signal(stop_signal, _stop_serving)
        for stop_signal in STOP_SIGNALS
    }
    print(f"Serving on http://{address[0]}:{address[1]}", file=sys.stderr)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        for stop_signal, previous_handler in previous_handlers.items():
            signal.signal(stop_signal, previous_handler)


def main() -> None:
    """Launch the packaged inference HTTP service from process entrypoint."""
    service = create_service_from_env()
    serve(service)
