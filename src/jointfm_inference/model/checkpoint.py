"""Checkpoint file reading and the generation marker.

A checkpoint is a plain-data mapping saved with ``torch.save``: tensors,
strings, numbers, lists, and mappings, so it loads with ``weights_only=True``.
It names the JointFM generation that wrote it under
``runtime_metadata.generation`` (``jointfm-1``, for instance); the inference
backend registry resolves that marker to the backend that reads the rest of
the mapping.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import torch

CHECKPOINT_CFG_KEY: Final = "cfg"
CHECKPOINT_MODEL_STATE_KEY: Final = "model_state"
CHECKPOINT_RUNTIME_METADATA_KEY: Final = "runtime_metadata"
CHECKPOINT_GENERATION_KEY: Final = "generation"
CHECKPOINT_INFERENCE_METADATA_KEY: Final = "inference_metadata"
CHECKPOINT_TIME_INDEX_SUPPORT_KEY: Final = "time_index_support"
CHECKPOINT_CAPACITY_ENVELOPE_KEY: Final = "data_generation"
JOINTFM_1_GENERATION: Final = "jointfm-1"


def read_checkpoint_file(
    path: str, *, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Load the checkpoint mapping at ``path`` without unpickling code objects.

    Raises:
        TypeError: when the file does not hold a mapping.
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("a checkpoint file must hold a mapping")
    return dict(checkpoint)


def checkpoint_generation(checkpoint: Mapping[str, Any]) -> str:
    """Return the generation marker of ``checkpoint``.

    Raises:
        ValueError: when the marker is missing or empty.
        TypeError: when the runtime metadata is not a mapping or the marker
            is not a string.
    """
    runtime_metadata = checkpoint.get(CHECKPOINT_RUNTIME_METADATA_KEY)
    if runtime_metadata is None:
        raise ValueError(f"checkpoint missing '{CHECKPOINT_RUNTIME_METADATA_KEY}'")
    if not isinstance(runtime_metadata, Mapping):
        raise TypeError(
            f"checkpoint '{CHECKPOINT_RUNTIME_METADATA_KEY}' must be a mapping"
        )
    if CHECKPOINT_GENERATION_KEY not in runtime_metadata:
        raise ValueError(
            f"checkpoint '{CHECKPOINT_RUNTIME_METADATA_KEY}' missing "
            f"'{CHECKPOINT_GENERATION_KEY}'"
        )
    generation = runtime_metadata[CHECKPOINT_GENERATION_KEY]
    if not isinstance(generation, str):
        raise TypeError("checkpoint generation must be a string")
    if not generation:
        raise ValueError("checkpoint generation must not be empty")
    return generation


__all__ = [
    "CHECKPOINT_CAPACITY_ENVELOPE_KEY",
    "CHECKPOINT_CFG_KEY",
    "CHECKPOINT_GENERATION_KEY",
    "CHECKPOINT_INFERENCE_METADATA_KEY",
    "CHECKPOINT_MODEL_STATE_KEY",
    "CHECKPOINT_RUNTIME_METADATA_KEY",
    "CHECKPOINT_TIME_INDEX_SUPPORT_KEY",
    "JOINTFM_1_GENERATION",
    "checkpoint_generation",
    "read_checkpoint_file",
]
