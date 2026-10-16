"""The HTTP service mounted over a test network, with its runtime configuration."""

from typing import Any

from jointfm_inference.model.checkpoint import JOINTFM_1_GENERATION
from jointfm_inference.model.inference import JointFMInferenceSession
from jointfm_inference.model.jointfm_1.model import JointFM
from jointfm_inference.model.service import (
    DEFAULT_SCHEMA_VERSION,
    InferenceServiceConfig,
    JointFMHTTPService,
)

from tests.support.models import checkpoint_mapping

CHECKPOINT_VERSION = "example-1"
MODEL_VERSION = f"jointfm-inference:0.2.0+ckpt.{CHECKPOINT_VERSION}"
# Full precision keeps a served number comparable with an in-process oracle.
SERVICE_PRECISION = "fp32"
SERVICE_MAX_SAMPLE_COUNT = 64


def make_service_config(**overrides: Any) -> InferenceServiceConfig:
    """Return a CPU runtime configuration; ``overrides`` replace single fields."""
    fields: dict[str, Any] = {
        "checkpoint_path": "/models/jointfm.pt",
        "checkpoint_version": CHECKPOINT_VERSION,
        "device": "cpu",
        "precision": SERVICE_PRECISION,
        "map_location": "cpu",
        "bind_host": "127.0.0.1",
        "port": 8080,
        "schema_version": DEFAULT_SCHEMA_VERSION,
        "image_version": "0.2.0",
        "model_version": MODEL_VERSION,
        "max_sample_count": SERVICE_MAX_SAMPLE_COUNT,
        "max_concurrent_requests": 1,
    }
    fields.update(overrides)
    return InferenceServiceConfig(**fields)


def make_session(
    model: JointFM,
    *,
    checkpoint: dict[str, Any] | None = None,
    precision: str = SERVICE_PRECISION,
) -> JointFMInferenceSession:
    """Mount ``model`` with ``checkpoint``, by default the one a release of it holds."""
    return JointFMInferenceSession(
        model,
        model.cfg,
        checkpoint_mapping(model) if checkpoint is None else checkpoint,
        generation=JOINTFM_1_GENERATION,
        precision=precision,
    )


def mount_service(
    model: JointFM,
    *,
    checkpoint: dict[str, Any] | None = None,
    precision: str = SERVICE_PRECISION,
    **config_overrides: Any,
) -> JointFMHTTPService:
    """Serve ``model`` at ``precision``; ``config_overrides`` reach the config."""
    return JointFMHTTPService(
        config=make_service_config(precision=precision, **config_overrides),
        session=make_session(model, checkpoint=checkpoint, precision=precision),
    )
