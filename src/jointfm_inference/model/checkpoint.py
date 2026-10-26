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

"""The checkpoint format the runner reads, and checkpoint file reading.

A checkpoint is a plain-data mapping saved with ``torch.save``: tensors,
strings, numbers, lists, and mappings, so it loads with ``weights_only=True``.
Its layout is versioned by ``runtime_metadata.format_version``, and the runner
reads exactly one version, :data:`CHECKPOINT_FORMAT_VERSION`:

- ``cfg``: the network sizes, whose fields the generation's backend defines;
- ``model_state``: the network's state dict, every floating tensor stored in
  the dtype ``runtime_metadata.weights_dtype`` names;
- ``runtime_metadata``: exactly ``generation`` (the JointFM generation that
  wrote the file, such as ``jointfm-1``), ``format_version``, and
  ``weights_dtype``;
- ``inference_metadata``: exactly ``time_index_support`` and
  ``data_generation`` (the capacity envelope), whose readers own their fields.

:func:`require_checkpoint_format` checks this layout before the inference
backend registry resolves the generation marker to the backend that reads
``cfg`` and ``model_state``, so a file of an unknown version or with an entry
outside the layout fails before any backend is imported or network is built.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import torch

CHECKPOINT_CFG_KEY: Final = "cfg"
CHECKPOINT_MODEL_STATE_KEY: Final = "model_state"
CHECKPOINT_RUNTIME_METADATA_KEY: Final = "runtime_metadata"
CHECKPOINT_INFERENCE_METADATA_KEY: Final = "inference_metadata"
CHECKPOINT_KEYS: Final = frozenset(
    {
        CHECKPOINT_CFG_KEY,
        CHECKPOINT_MODEL_STATE_KEY,
        CHECKPOINT_RUNTIME_METADATA_KEY,
        CHECKPOINT_INFERENCE_METADATA_KEY,
    }
)

CHECKPOINT_GENERATION_KEY: Final = "generation"
CHECKPOINT_FORMAT_VERSION_KEY: Final = "format_version"
CHECKPOINT_WEIGHTS_DTYPE_KEY: Final = "weights_dtype"
CHECKPOINT_RUNTIME_METADATA_KEYS: Final = frozenset(
    {
        CHECKPOINT_GENERATION_KEY,
        CHECKPOINT_FORMAT_VERSION_KEY,
        CHECKPOINT_WEIGHTS_DTYPE_KEY,
    }
)

CHECKPOINT_TIME_INDEX_SUPPORT_KEY: Final = "time_index_support"
CHECKPOINT_CAPACITY_ENVELOPE_KEY: Final = "data_generation"
CHECKPOINT_INFERENCE_METADATA_KEYS: Final = frozenset(
    {CHECKPOINT_TIME_INDEX_SUPPORT_KEY, CHECKPOINT_CAPACITY_ENVELOPE_KEY}
)

CHECKPOINT_FORMAT_VERSION: Final = 1
"""The one checkpoint format version this runner reads."""

CHECKPOINT_WEIGHTS_DTYPES: Final[Mapping[str, torch.dtype]] = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}
"""Weights dtype labels and the dtype each names; loading upcasts to float32."""

JOINTFM_1_GENERATION: Final = "jointfm-1"

_DIGEST_CHUNK_BYTES: Final = 1 << 20


def file_sha256(path: Path) -> str:
    """Return the lowercase hexadecimal SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_DIGEST_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def read_checkpoint_file(
    path: str, *, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Load the checkpoint mapping at ``path`` without unpickling code objects.

    The mapping is returned as read; :func:`require_checkpoint_format` checks
    its layout.

    Raises:
        TypeError: when the file does not hold a mapping.
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("a checkpoint file must hold a mapping")
    return dict(checkpoint)


def _require_exact_keys(
    mapping: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> None:
    """Refuse ``mapping`` unless its keys are exactly ``expected``."""
    missing = sorted(expected - set(mapping))
    unknown = sorted(set(mapping) - expected)
    if missing or unknown:
        raise ValueError(
            f"{label} does not match checkpoint format version "
            f"{CHECKPOINT_FORMAT_VERSION}: missing {missing}, unknown {unknown}"
        )


def _require_mapping(checkpoint: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return the entry ``key`` of ``checkpoint`` when it is a mapping."""
    value = checkpoint[key]
    if not isinstance(value, Mapping):
        raise TypeError(f"checkpoint '{key}' must be a mapping")
    return value


def _require_format_version(runtime_metadata: Mapping[str, Any]) -> None:
    """Refuse every format version other than :data:`CHECKPOINT_FORMAT_VERSION`."""
    version = runtime_metadata[CHECKPOINT_FORMAT_VERSION_KEY]
    if isinstance(version, bool) or not isinstance(version, int):
        raise TypeError(
            f"checkpoint format version must be an int, got {type(version).__name__}"
        )
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format version {version}; this runner reads "
            f"version {CHECKPOINT_FORMAT_VERSION}"
        )


def _require_generation(runtime_metadata: Mapping[str, Any]) -> str:
    """Return the generation marker when it is a non-empty string."""
    generation = runtime_metadata[CHECKPOINT_GENERATION_KEY]
    if not isinstance(generation, str):
        raise TypeError("checkpoint generation must be a string")
    if not generation:
        raise ValueError("checkpoint generation must not be empty")
    return generation


def _require_weights_dtype(
    runtime_metadata: Mapping[str, Any], model_state: Mapping[str, Any]
) -> None:
    """Refuse an unknown dtype label or a floating tensor stored in another dtype."""
    label = runtime_metadata[CHECKPOINT_WEIGHTS_DTYPE_KEY]
    if not isinstance(label, str) or label not in CHECKPOINT_WEIGHTS_DTYPES:
        raise ValueError(
            f"checkpoint weights dtype must be one of "
            f"{sorted(CHECKPOINT_WEIGHTS_DTYPES)}, got {label!r}"
        )
    dtype = CHECKPOINT_WEIGHTS_DTYPES[label]
    for name, tensor in model_state.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"checkpoint '{CHECKPOINT_MODEL_STATE_KEY}' entry {name!r} must be "
                f"a tensor, got {type(tensor).__name__}"
            )
        if tensor.is_floating_point() and tensor.dtype != dtype:
            raise ValueError(
                f"checkpoint weights dtype is {label!r}, but "
                f"'{CHECKPOINT_MODEL_STATE_KEY}' entry {name!r} is {tensor.dtype}"
            )


def require_checkpoint_format(checkpoint: Mapping[str, Any]) -> str:
    """Check that ``checkpoint`` follows the format this runner reads.

    Checks every part of the layout the module docstring describes except the
    fields of ``cfg``, ``time_index_support``, and ``data_generation``, which
    the generation's backend and the metadata readers own. The format version
    is checked before the other runtime metadata, so a file of another version
    is refused for its version, not for whatever its layout changed.

    Returns:
        The checkpoint generation marker.

    Raises:
        ValueError: when an entry is missing or unknown, the format version is
            not :data:`CHECKPOINT_FORMAT_VERSION`, the generation marker is
            empty, the weights dtype label is unknown, or a floating tensor is
            stored in another dtype than the label names.
        TypeError: when an entry has the wrong type.
    """
    _require_exact_keys(checkpoint, CHECKPOINT_KEYS, label="checkpoint")
    runtime_metadata = _require_mapping(checkpoint, CHECKPOINT_RUNTIME_METADATA_KEY)
    if CHECKPOINT_FORMAT_VERSION_KEY not in runtime_metadata:
        raise ValueError(
            f"checkpoint '{CHECKPOINT_RUNTIME_METADATA_KEY}' missing "
            f"'{CHECKPOINT_FORMAT_VERSION_KEY}'"
        )
    _require_format_version(runtime_metadata)
    _require_exact_keys(
        runtime_metadata,
        CHECKPOINT_RUNTIME_METADATA_KEYS,
        label=f"checkpoint '{CHECKPOINT_RUNTIME_METADATA_KEY}'",
    )
    generation = _require_generation(runtime_metadata)
    _require_mapping(checkpoint, CHECKPOINT_CFG_KEY)
    model_state = _require_mapping(checkpoint, CHECKPOINT_MODEL_STATE_KEY)
    _require_weights_dtype(runtime_metadata, model_state)
    inference_metadata = _require_mapping(checkpoint, CHECKPOINT_INFERENCE_METADATA_KEY)
    _require_exact_keys(
        inference_metadata,
        CHECKPOINT_INFERENCE_METADATA_KEYS,
        label=f"checkpoint '{CHECKPOINT_INFERENCE_METADATA_KEY}'",
    )
    return generation


__all__ = [
    "CHECKPOINT_CAPACITY_ENVELOPE_KEY",
    "CHECKPOINT_CFG_KEY",
    "CHECKPOINT_FORMAT_VERSION",
    "CHECKPOINT_FORMAT_VERSION_KEY",
    "CHECKPOINT_GENERATION_KEY",
    "CHECKPOINT_INFERENCE_METADATA_KEY",
    "CHECKPOINT_INFERENCE_METADATA_KEYS",
    "CHECKPOINT_KEYS",
    "CHECKPOINT_MODEL_STATE_KEY",
    "CHECKPOINT_RUNTIME_METADATA_KEY",
    "CHECKPOINT_RUNTIME_METADATA_KEYS",
    "CHECKPOINT_TIME_INDEX_SUPPORT_KEY",
    "CHECKPOINT_WEIGHTS_DTYPES",
    "CHECKPOINT_WEIGHTS_DTYPE_KEY",
    "JOINTFM_1_GENERATION",
    "file_sha256",
    "read_checkpoint_file",
    "require_checkpoint_format",
]
