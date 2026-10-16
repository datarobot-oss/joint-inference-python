"""The capacity envelope a checkpoint records and the health payload reports.

The envelope is the ``inference_metadata.data_generation`` block of a
checkpoint: the request sizes the model was built for. The service reports it
as the ``data_generation`` block of ``/healthz`` so a client can size and split
its requests.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Optional

from jointfm_inference.model.checkpoint import (
    CHECKPOINT_CAPACITY_ENVELOPE_KEY,
    CHECKPOINT_INFERENCE_METADATA_KEY,
)


@dataclass(frozen=True)
class CapacityEnvelope:
    """Request sizes a model was built for.

    ``min_series`` and ``max_series`` bound the total column count,
    ``n_input`` and ``n_output`` are the longest history and horizon in steps,
    and ``t_input`` and ``t_output`` their time spans. Keys of the checkpoint
    block outside these six are not read.
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
    """Parse one checkpoint capacity-envelope block."""
    return CapacityEnvelope(
        min_series=_require_int(raw_payload, "min_series"),
        max_series=_require_int(raw_payload, "max_series"),
        t_input=_require_positive_float(raw_payload, "t_input"),
        t_output=_require_positive_float(raw_payload, "t_output"),
        n_input=_require_int(raw_payload, "n_input"),
        n_output=_require_int(raw_payload, "n_output"),
    )


def _resolve_capacity_envelope(
    checkpoint: Optional[Mapping[str, Any]],
) -> Optional[CapacityEnvelope]:
    """Return the capacity envelope a checkpoint records, or ``None`` without one.

    Raises when the block is present but malformed.
    """
    if checkpoint is None:
        return None
    raw_inference_metadata = checkpoint.get(CHECKPOINT_INFERENCE_METADATA_KEY)
    if not isinstance(raw_inference_metadata, Mapping):
        return None
    raw_envelope = raw_inference_metadata.get(CHECKPOINT_CAPACITY_ENVELOPE_KEY)
    if raw_envelope is None:
        return None
    if not isinstance(raw_envelope, Mapping):
        raise TypeError(
            "checkpoint inference metadata "
            f"{CHECKPOINT_CAPACITY_ENVELOPE_KEY!r} must be a mapping"
        )
    return _capacity_envelope_from_payload(raw_envelope)
