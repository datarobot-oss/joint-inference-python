"""Rebuild a JointFM 1 network from a checkpoint mapping.

A JointFM 1 checkpoint holds exactly these entries:

- ``cfg``: the :class:`JointFMConfig` fields as a flat mapping;
- ``model_state``: the network's state dict;
- ``runtime_metadata``: ``{"generation": "jointfm-1"}``;
- ``inference_metadata``: the time-index support and the capacity envelope
  that the serving backend and the health payload read.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, Optional

import torch

from jointfm_inference.model.checkpoint import (
    CHECKPOINT_CFG_KEY,
    CHECKPOINT_INFERENCE_METADATA_KEY,
    CHECKPOINT_MODEL_STATE_KEY,
    CHECKPOINT_RUNTIME_METADATA_KEY,
    JOINTFM_1_GENERATION,
    checkpoint_generation,
)
from jointfm_inference.model.jointfm_1.config import JointFMConfig
from jointfm_inference.model.jointfm_1.model import JointFM

CHECKPOINT_KEYS: Final = frozenset(
    {
        CHECKPOINT_CFG_KEY,
        CHECKPOINT_MODEL_STATE_KEY,
        CHECKPOINT_RUNTIME_METADATA_KEY,
        CHECKPOINT_INFERENCE_METADATA_KEY,
    }
)


def load_model_from_checkpoint(
    checkpoint: Mapping[str, Any], *, device: Optional[torch.device] = None
) -> tuple[JointFM, JointFMConfig]:
    """Return the network rebuilt from ``checkpoint``, in evaluation mode.

    Every parameter must match the checkpoint exactly (``strict=True``).

    Raises:
        ValueError: when the checkpoint carries a missing or unknown top-level
            entry, names another generation, or its ``cfg`` does not
            describe a JointFM 1 network.
        TypeError: when ``cfg`` or ``model_state`` is not a mapping.
        RuntimeError: when the state dict does not match the network.
    """
    missing = sorted(CHECKPOINT_KEYS - set(checkpoint))
    unknown = sorted(set(checkpoint) - CHECKPOINT_KEYS)
    if missing or unknown:
        raise ValueError(
            f"not a JointFM 1 checkpoint: missing {missing}, unknown {unknown}"
        )
    generation = checkpoint_generation(checkpoint)
    if generation != JOINTFM_1_GENERATION:
        raise ValueError(
            f"the JointFM 1 loader requires generation '{JOINTFM_1_GENERATION}', "
            f"got '{generation}'"
        )
    raw_cfg = checkpoint[CHECKPOINT_CFG_KEY]
    state = checkpoint[CHECKPOINT_MODEL_STATE_KEY]
    if not isinstance(raw_cfg, Mapping):
        raise TypeError(f"checkpoint '{CHECKPOINT_CFG_KEY}' must be a mapping")
    if not isinstance(state, Mapping):
        raise TypeError(f"checkpoint '{CHECKPOINT_MODEL_STATE_KEY}' must be a mapping")

    cfg = JointFMConfig.from_mapping(raw_cfg)
    model = JointFM(cfg)
    model.load_state_dict(state, strict=True)
    if device is not None:
        model.to(device)
    model.eval()
    return model, cfg


__all__ = ["CHECKPOINT_KEYS", "load_model_from_checkpoint"]
