"""The time-index contract a mounted model reads request times under."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Final, Sequence

from jointfm_inference.model.checkpoint import (
    CHECKPOINT_INFERENCE_METADATA_KEY,
    CHECKPOINT_TIME_INDEX_SUPPORT_KEY,
)

DISCRETE_GRID_TIME_INDEX_ENCODING: Final = "discrete_grid"
"""Encoding of models that read time as integer steps on a regular grid."""


@dataclass(frozen=True)
class TimeIndexSupport:
    """Time-index encoding, accepted request modes, and grid capacities.

    ``max_time_ctx`` bounds the history length and ``max_time_fut`` the
    furthest forecast step, both in grid steps.
    """

    encoding: str
    supported_request_time_index_modes: tuple[str, ...]
    max_time_ctx: int
    max_time_fut: int


def time_index_support_from_checkpoint(
    checkpoint: Mapping[str, Any],
) -> TimeIndexSupport:
    """Read the ``inference_metadata.time_index_support`` block of a checkpoint.

    Raises:
        ValueError: when the block is missing, lacks a field, carries a key
            outside :class:`TimeIndexSupport`, or names another encoding.
        TypeError: when a field has the wrong type.
    """
    inference_metadata = checkpoint.get(CHECKPOINT_INFERENCE_METADATA_KEY)
    if not isinstance(inference_metadata, Mapping):
        raise ValueError(f"checkpoint missing '{CHECKPOINT_INFERENCE_METADATA_KEY}'")
    raw = inference_metadata.get(CHECKPOINT_TIME_INDEX_SUPPORT_KEY)
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"checkpoint '{CHECKPOINT_INFERENCE_METADATA_KEY}' missing "
            f"'{CHECKPOINT_TIME_INDEX_SUPPORT_KEY}'"
        )
    names = {field.name for field in fields(TimeIndexSupport)}
    missing = sorted(names - set(raw))
    unknown = sorted(set(raw) - names)
    if missing or unknown:
        raise ValueError(
            f"checkpoint '{CHECKPOINT_TIME_INDEX_SUPPORT_KEY}' block must hold "
            f"exactly the time-index support: missing {missing}, unknown {unknown}"
        )
    encoding = raw["encoding"]
    if encoding != DISCRETE_GRID_TIME_INDEX_ENCODING:
        raise ValueError(
            f"unsupported time-index encoding: expected "
            f"{DISCRETE_GRID_TIME_INDEX_ENCODING!r}, got {encoding!r}"
        )
    modes = raw.get("supported_request_time_index_modes")
    if not isinstance(modes, Sequence) or isinstance(modes, (str, bytes)):
        raise TypeError(
            "'supported_request_time_index_modes' must be a list of strings"
        )
    capacities = {}
    for field in ("max_time_ctx", "max_time_fut"):
        value = raw.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise TypeError(f"'{field}' must be a positive int, got {value!r}")
        capacities[field] = value
    return TimeIndexSupport(
        encoding=DISCRETE_GRID_TIME_INDEX_ENCODING,
        supported_request_time_index_modes=tuple(str(mode) for mode in modes),
        max_time_ctx=capacities["max_time_ctx"],
        max_time_fut=capacities["max_time_fut"],
    )


__all__ = [
    "DISCRETE_GRID_TIME_INDEX_ENCODING",
    "TimeIndexSupport",
    "time_index_support_from_checkpoint",
]
