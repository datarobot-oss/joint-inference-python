"""Tiny JointFM 1 networks and checkpoints for tests."""

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from jointfm_inference.model.checkpoint import (
    CHECKPOINT_CAPACITY_ENVELOPE_KEY,
    CHECKPOINT_CFG_KEY,
    CHECKPOINT_GENERATION_KEY,
    CHECKPOINT_INFERENCE_METADATA_KEY,
    CHECKPOINT_MODEL_STATE_KEY,
    CHECKPOINT_RUNTIME_METADATA_KEY,
    CHECKPOINT_TIME_INDEX_SUPPORT_KEY,
    JOINTFM_1_GENERATION,
)
from jointfm_inference.model.inference.time_index import (
    DISCRETE_GRID_TIME_INDEX_ENCODING,
)
from jointfm_inference.model.jointfm_1.config import JointFMConfig
from jointfm_inference.model.jointfm_1.model import JointFM

TINY_CONFIG = JointFMConfig(
    num_layers=2,
    column_capacity=4,
    n_mixtures=2,
    covariance_rank=2,
    d_value=4,
    d_model=8,
    nhead=2,
    dim_feedforward=16,
    layer_norm_eps=1e-5,
    max_time_ctx=24,
    max_time_fut=6,
    attention_time_blocks=1,
    attention_row_blocks=1,
    attention_cnn_kernel_size=3,
    attention_cnn_first_layers=1,
    horizon_embed_scale=1.0,
    time_freq_base=100.0,
    horizon_freq_base=100.0,
    head_trunk_layers=1,
    head_trunk_hidden=None,
    head_min_df=2.0,
    head_min_scale=1e-4,
    pooling_nhead=2,
    pooling_seeds=1,
    pooling_layers=1,
)

TINY_CAPACITY_ENVELOPE = {
    "min_series": 1,
    "max_series": 4,
    "t_input": 1.0,
    "t_output": 0.25,
    "n_input": 24,
    "n_output": 6,
}


def tiny_config(**overrides: Any) -> JointFMConfig:
    """Return ``TINY_CONFIG`` with ``overrides`` applied."""
    return replace(TINY_CONFIG, **overrides)


def tiny_model(cfg: JointFMConfig = TINY_CONFIG, *, seed: int = 0) -> JointFM:
    """Return a network with seeded random weights, in evaluation mode."""
    torch.manual_seed(seed)
    return JointFM(cfg).eval()


def checkpoint_mapping(
    model: JointFM, *, capacity_envelope: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return the checkpoint mapping a release of ``model`` holds."""
    cfg = model.cfg
    inference_metadata: dict[str, Any] = {
        CHECKPOINT_TIME_INDEX_SUPPORT_KEY: {
            "encoding": DISCRETE_GRID_TIME_INDEX_ENCODING,
            "supported_request_time_index_modes": [
                "ordinal",
                "continuous_float",
                "absolute_datetime",
            ],
            "max_time_ctx": cfg.max_time_ctx,
            "max_time_fut": cfg.max_time_fut,
        },
        CHECKPOINT_CAPACITY_ENVELOPE_KEY: dict(
            TINY_CAPACITY_ENVELOPE if capacity_envelope is None else capacity_envelope
        ),
    }
    return {
        CHECKPOINT_CFG_KEY: asdict(cfg),
        CHECKPOINT_MODEL_STATE_KEY: model.state_dict(),
        CHECKPOINT_RUNTIME_METADATA_KEY: {
            CHECKPOINT_GENERATION_KEY: JOINTFM_1_GENERATION
        },
        CHECKPOINT_INFERENCE_METADATA_KEY: inference_metadata,
    }


def write_checkpoint(path: Path, model: JointFM | None = None) -> JointFM:
    """Save a checkpoint of ``model`` (a fresh tiny one by default) to ``path``."""
    model = tiny_model() if model is None else model
    torch.save(checkpoint_mapping(model), path)
    return model
