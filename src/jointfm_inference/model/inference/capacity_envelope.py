"""The capacity envelope a checkpoint records and the health payload reports.

The envelope is the ``inference_metadata.data_generation`` block of a
checkpoint: the request sizes the model was built for. The service reports it
as the ``data_generation`` block of ``/healthz`` so a client can size and split
its requests.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

from jointfm_inference.model.checkpoint import (
    CHECKPOINT_CAPACITY_ENVELOPE_KEY,
    CHECKPOINT_INFERENCE_METADATA_KEY,
)


@dataclass(frozen=True)
class CapacityEnvelope:
    """Request sizes a model was built for.

    ``min_series`` and ``max_series`` bound the total column count,
    ``n_input`` and ``n_output`` are the longest history and horizon in steps,
    and ``t_input`` and ``t_output`` their time spans. The checkpoint block
    holds exactly these six fields.
    """

    min_series: int
    max_series: int
    t_input: float
    t_output: float
    n_input: int
    n_output: int

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable payload mirroring the checkpoint block."""
        return asdict(self)


def _require_int(payload: Mapping[str, Any], field: str) -> int:
    """Extract one non-negative int field from a checkpoint payload."""
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"capacity envelope field {field!r} must be an int")
    if value < 0:
        raise ValueError(f"capacity envelope field {field!r} must be non-negative")
    return value


def _require_positive_float(payload: Mapping[str, Any], field: str) -> float:
    """Extract one strictly positive float field from a checkpoint payload."""
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"capacity envelope field {field!r} must be a number")
    if value <= 0:
        raise ValueError(f"capacity envelope field {field!r} must be positive")
    return float(value)


def _capacity_envelope_from_payload(
    raw_payload: Mapping[str, Any],
) -> CapacityEnvelope:
    """Parse one checkpoint capacity-envelope block holding exactly its fields."""
    names = {field.name for field in fields(CapacityEnvelope)}
    missing = sorted(names - set(raw_payload))
    unknown = sorted(set(raw_payload) - names)
    if missing or unknown:
        raise ValueError(
            f"checkpoint {CHECKPOINT_CAPACITY_ENVELOPE_KEY!r} block must hold "
            f"exactly the capacity envelope: missing {missing}, unknown {unknown}"
        )
    return CapacityEnvelope(
        min_series=_require_int(raw_payload, "min_series"),
        max_series=_require_int(raw_payload, "max_series"),
        t_input=_require_positive_float(raw_payload, "t_input"),
        t_output=_require_positive_float(raw_payload, "t_output"),
        n_input=_require_int(raw_payload, "n_input"),
        n_output=_require_int(raw_payload, "n_output"),
    )


def capacity_envelope_from_checkpoint(
    checkpoint: Mapping[str, Any],
) -> CapacityEnvelope:
    """Read the ``inference_metadata.data_generation`` block of a checkpoint.

    Raises:
        ValueError: when the block is missing, lacks a field, or carries a
            key outside the envelope.
        TypeError: when the block or a field has the wrong type.
    """
    inference_metadata = checkpoint.get(CHECKPOINT_INFERENCE_METADATA_KEY)
    if not isinstance(inference_metadata, Mapping):
        raise ValueError(f"checkpoint missing '{CHECKPOINT_INFERENCE_METADATA_KEY}'")
    if CHECKPOINT_CAPACITY_ENVELOPE_KEY not in inference_metadata:
        raise ValueError(
            f"checkpoint '{CHECKPOINT_INFERENCE_METADATA_KEY}' missing "
            f"'{CHECKPOINT_CAPACITY_ENVELOPE_KEY}'"
        )
    raw_envelope = inference_metadata[CHECKPOINT_CAPACITY_ENVELOPE_KEY]
    if not isinstance(raw_envelope, Mapping):
        raise TypeError(
            "checkpoint inference metadata "
            f"{CHECKPOINT_CAPACITY_ENVELOPE_KEY!r} must be a mapping"
        )
    return _capacity_envelope_from_payload(raw_envelope)


__all__ = ["CapacityEnvelope", "capacity_envelope_from_checkpoint"]
