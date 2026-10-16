"""Runtime configuration loader for the packaged inference container.

``package_version`` is imported at module scope so tests can patch the symbol
at ``jointfm_inference.model.service.config.package_version`` to control the reported
image version without booting a full container.
"""

from __future__ import annotations

import os

from jointfm_inference.core.version import package_version
from jointfm_inference.model.inference import DEFAULT_INFERENCE_PRECISION
from jointfm_inference.model.precision import MODEL_PRECISIONS

from .constants import (
    DEFAULT_BIND_HOST,
    DEFAULT_CHECKPOINT_MOUNT_PATH,
    DEFAULT_DEVICE,
    DEFAULT_MAX_CONCURRENT_REQUESTS,
    DEFAULT_MAX_SAMPLE_COUNT,
    DEFAULT_PORT,
    DEFAULT_SCHEMA_VERSION,
)
from .validation import _choice_from_env, _positive_int_from_env, _required_env


class InferenceServiceConfig:
    """Runtime configuration for the packaged inference container."""

    def __init__(
        self,
        *,
        checkpoint_path: str,
        checkpoint_version: str,
        device: str,
        precision: str,
        map_location: str,
        bind_host: str,
        port: int,
        schema_version: str,
        image_version: str,
        model_version: str,
        max_sample_count: int,
        max_concurrent_requests: int,
    ) -> None:
        """Store validated runtime settings for the packaged inference container.

        ``precision`` is the model precision (``JOINT_PRECISION``) every forward
        runs at; the session re-validates it against the model's device.
        """
        self.checkpoint_path = checkpoint_path
        self.checkpoint_version = checkpoint_version
        self.device = device
        self.precision = precision
        self.map_location = map_location
        self.bind_host = bind_host
        self.port = port
        self.schema_version = schema_version
        self.image_version = image_version
        self.model_version = model_version
        self.max_sample_count = max_sample_count
        self.max_concurrent_requests = max_concurrent_requests

    @classmethod
    def from_env(cls) -> "InferenceServiceConfig":
        """Build validated runtime configuration from environment variables."""
        checkpoint_path = os.getenv(
            "JOINT_CHECKPOINT_PATH",
            DEFAULT_CHECKPOINT_MOUNT_PATH,
        )
        checkpoint_version = _required_env("JOINT_CHECKPOINT_VERSION")
        device = os.getenv("JOINT_DEVICE", DEFAULT_DEVICE)
        precision = _choice_from_env(
            "JOINT_PRECISION",
            DEFAULT_INFERENCE_PRECISION,
            MODEL_PRECISIONS,
        )
        map_location = os.getenv("JOINT_MAP_LOCATION", device)
        bind_host = os.getenv("JOINT_BIND_HOST", DEFAULT_BIND_HOST)
        port = _positive_int_from_env("JOINT_PORT", DEFAULT_PORT)
        max_sample_count = _positive_int_from_env(
            "JOINT_MAX_SAMPLE_COUNT",
            DEFAULT_MAX_SAMPLE_COUNT,
        )
        max_concurrent_requests = _positive_int_from_env(
            "JOINT_MAX_CONCURRENT_REQUESTS",
            DEFAULT_MAX_CONCURRENT_REQUESTS,
        )

        image_version = package_version("jointfm-inference")
        model_version = os.getenv(
            "JOINT_MODEL_VERSION",
            f"jointfm-inference:{image_version}+ckpt.{checkpoint_version}",
        )
        return cls(
            checkpoint_path=checkpoint_path,
            checkpoint_version=checkpoint_version,
            device=device,
            precision=precision,
            map_location=map_location,
            bind_host=bind_host,
            port=port,
            schema_version=DEFAULT_SCHEMA_VERSION,
            image_version=image_version,
            model_version=model_version,
            max_sample_count=max_sample_count,
            max_concurrent_requests=max_concurrent_requests,
        )
