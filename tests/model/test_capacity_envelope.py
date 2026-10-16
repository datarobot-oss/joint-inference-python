"""Tests for reading the capacity envelope a checkpoint records."""

from __future__ import annotations

from typing import Any

import pytest

from jointfm_inference.model.checkpoint import (
    CHECKPOINT_CAPACITY_ENVELOPE_KEY,
    CHECKPOINT_INFERENCE_METADATA_KEY,
)
from jointfm_inference.model.inference import CapacityEnvelope
from jointfm_inference.model.inference.capacity_envelope import (
    _capacity_envelope_from_payload,
    capacity_envelope_from_checkpoint,
)


def _payload() -> dict[str, Any]:
    return {
        "min_series": 1,
        "max_series": 5,
        "t_input": 2.0,
        "t_output": 0.5,
        "n_input": 40,
        "n_output": 8,
    }


def _checkpoint(envelope: Any) -> dict[str, Any]:
    return {
        CHECKPOINT_INFERENCE_METADATA_KEY: {CHECKPOINT_CAPACITY_ENVELOPE_KEY: envelope}
    }


def test_capacity_envelope_round_trip() -> None:
    envelope = _capacity_envelope_from_payload(_payload())

    assert envelope == CapacityEnvelope(
        min_series=1, max_series=5, t_input=2.0, t_output=0.5, n_input=40, n_output=8
    )
    assert envelope.to_payload() == _payload()


def test_integer_time_spans_are_read_as_floats() -> None:
    envelope = _capacity_envelope_from_payload({**_payload(), "t_input": 3})

    assert isinstance(envelope.t_input, float)
    assert envelope.t_input == 3.0


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("min_series", "1", TypeError, "must be an int"),
        ("max_series", True, TypeError, "must be an int"),
        ("n_input", 4.0, TypeError, "must be an int"),
        ("max_series", -1, ValueError, "must be non-negative"),
        ("t_input", True, TypeError, "must be a number"),
        ("t_output", "0.5", TypeError, "must be a number"),
        ("t_output", 0.0, ValueError, "must be positive"),
        ("t_input", -1.0, ValueError, "must be positive"),
    ],
)
def test_capacity_envelope_rejects_invalid_fields(
    field: str, value: Any, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=f"{field!r} {message}"):
        _capacity_envelope_from_payload({**_payload(), field: value})


def test_a_missing_field_is_refused() -> None:
    payload = _payload()
    del payload["n_output"]

    with pytest.raises(ValueError, match=r"missing \['n_output'\]"):
        _capacity_envelope_from_payload(payload)


def test_a_key_outside_the_envelope_is_refused() -> None:
    """A key the block carries beyond the envelope is refused, never passed on."""
    with pytest.raises(ValueError, match=r"unknown \['notes'\]"):
        _capacity_envelope_from_payload({**_payload(), "notes": "unreported"})


@pytest.mark.parametrize(
    "checkpoint",
    [{}, {CHECKPOINT_INFERENCE_METADATA_KEY: "not-a-mapping"}],
)
def test_a_checkpoint_without_inference_metadata_is_refused(
    checkpoint: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="missing 'inference_metadata'"):
        capacity_envelope_from_checkpoint(checkpoint)


def test_a_checkpoint_without_an_envelope_block_is_refused() -> None:
    with pytest.raises(ValueError, match="missing 'data_generation'"):
        capacity_envelope_from_checkpoint({CHECKPOINT_INFERENCE_METADATA_KEY: {}})


def test_the_checkpoint_block_is_parsed() -> None:
    envelope = capacity_envelope_from_checkpoint(_checkpoint(_payload()))

    assert envelope.to_payload() == _payload()


@pytest.mark.parametrize("block", [None, ["nope"]])
def test_a_non_mapping_block_is_refused(block: Any) -> None:
    with pytest.raises(TypeError, match="must be a mapping"):
        capacity_envelope_from_checkpoint(_checkpoint(block))


def test_a_malformed_block_is_refused() -> None:
    with pytest.raises(ValueError, match="'max_series' must be non-negative"):
        capacity_envelope_from_checkpoint(_checkpoint({**_payload(), "max_series": -2}))
