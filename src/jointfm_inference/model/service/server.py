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

"""Process entrypoints and process log of the packaged inference HTTP server.

``ThreadingHTTPServer`` is imported at module scope so tests can patch the
symbol at ``jointfm_inference.model.service.server.ThreadingHTTPServer`` for transport-
level integration without binding a real socket.
"""

from __future__ import annotations

import logging
import signal
import sys
from http.server import ThreadingHTTPServer
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import FrameType

from jointfm_inference.model.inference import JointFMInferenceSession

from .checkpoint_source import resolve_checkpoint
from .config import InferenceServiceConfig
from .http_handler import _handler_class
from .http_service import JointFMHTTPService

STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)
SERVICE_LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_FILE_NAME = "jointfm-inference.log"
# Bounds the disk the log takes: a file rolls over at this size, and the
# service keeps LOG_FILE_COUNT files, the one it writes included.
LOG_FILE_MAX_BYTES = 32 * 1024 * 1024
LOG_FILE_COUNT = 3

logger = logging.getLogger(__name__)


def configure_logging(log_dir: str | None) -> None:
    """Send the process's log records to stderr and, with ``log_dir``, a rolling file.

    Records from ``INFO`` up reach the root logger's handlers, tracebacks
    included, so a server failure leaves its traceback under the reference
    its response names. With ``log_dir`` (``JOINTFM_INFERENCE_LOG_DIR``), the
    directory is created if missing and records also go to
    ``<log_dir>/jointfm-inference.log``, which rolls over at
    ``LOG_FILE_MAX_BYTES``, keeping ``LOG_FILE_COUNT`` files. Only the
    process entry point calls it, because it configures the whole process.

    Raises:
        OSError: ``log_dir`` cannot be created or its log file not opened.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_dir is not None:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(
                directory / LOG_FILE_NAME,
                maxBytes=LOG_FILE_MAX_BYTES,
                backupCount=LOG_FILE_COUNT - 1,
                encoding="utf-8",
            )
        )
    root = logging.getLogger()
    formatter = logging.Formatter(LOG_FORMAT)
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(SERVICE_LOG_LEVEL)


def create_service(config: InferenceServiceConfig) -> JointFMHTTPService:
    """Load the checkpoint ``config`` names and return the ready HTTP service.

    The checkpoint is downloaded first when ``config`` names a download
    source and the file is not yet at the configured path.
    """
    checkpoint_path = resolve_checkpoint(config)

    session = JointFMInferenceSession.from_checkpoint(
        checkpoint_path,
        map_location=config.map_location,
        device=config.device,
        precision=config.precision,
    )
    return JointFMHTTPService(config=config, session=session)


def create_service_from_env() -> JointFMHTTPService:
    """Return the ready HTTP service of the checkpoint the environment configures."""
    return create_service(InferenceServiceConfig.from_env())


def _stop_serving(signum: int, frame: FrameType | None) -> None:
    """Leave ``serve_forever`` with a clean exit when the process is told to stop."""
    del frame
    logger.info("Stopping on %s", signal.Signals(signum).name)
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
    logger.info("Serving on http://%s:%d", *address)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        for stop_signal, previous_handler in previous_handlers.items():
            signal.signal(stop_signal, previous_handler)


def main() -> None:
    """Configure the process log, then load and serve the configured checkpoint.

    The log is configured before the checkpoint loads, so the records of a
    download or a load reach it too.
    """
    config = InferenceServiceConfig.from_env()
    configure_logging(config.log_dir)
    serve(create_service(config))
