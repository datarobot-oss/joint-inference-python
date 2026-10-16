"""Process entrypoints for the packaged inference HTTP server.

``ThreadingHTTPServer`` is imported at module scope so tests can patch the
symbol at ``jointfm_inference.model.service.server.ThreadingHTTPServer`` for transport-
level integration without binding a real socket.
"""

from __future__ import annotations

from http.server import ThreadingHTTPServer
from pathlib import Path

from jointfm_inference.model.inference import JointFMInferenceSession

from .config import InferenceServiceConfig
from .http_handler import _handler_class
from .http_service import JointFMHTTPService


def create_service_from_env() -> JointFMHTTPService:
    """Load the configured checkpoint and return the ready HTTP service."""
    config = InferenceServiceConfig.from_env()
    checkpoint_path = Path(config.checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found at configured path: {config.checkpoint_path}"
        )

    session = JointFMInferenceSession.from_checkpoint(
        checkpoint_path,
        map_location=config.map_location,
        device=config.device,
        precision=config.precision,
    )
    return JointFMHTTPService(config=config, session=session)


def serve(service: JointFMHTTPService) -> None:
    """Start the threaded HTTP server for the packaged inference runtime."""
    handler = _handler_class(service)
    server = ThreadingHTTPServer(
        (service._config.bind_host, service._config.port),
        handler,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main() -> None:
    """Launch the packaged inference HTTP service from process entrypoint."""
    service = create_service_from_env()
    serve(service)
