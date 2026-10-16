"""Tests for rebuilding a JointFM 1 network from a checkpoint mapping."""

from dataclasses import asdict
from typing import Any

import pytest
import torch

from jointfm_inference.model.checkpoint import (
    CHECKPOINT_FORMAT_VERSION_KEY,
    CHECKPOINT_GENERATION_KEY,
    CHECKPOINT_MODEL_STATE_KEY,
    CHECKPOINT_RUNTIME_METADATA_KEY,
    CHECKPOINT_WEIGHTS_DTYPE_KEY,
)
from jointfm_inference.model.jointfm_1.loader import load_model_from_checkpoint
from tests.support.models import TINY_CONFIG, checkpoint_mapping, tiny_model


def _checkpoint() -> dict[str, Any]:
    """Return the checkpoint mapping of a fresh tiny network."""
    return checkpoint_mapping(tiny_model())


def test_a_checkpoint_rebuilds_the_network_that_wrote_it() -> None:
    saved = tiny_model(seed=3)

    model, cfg = load_model_from_checkpoint(
        checkpoint_mapping(saved), device=torch.device("cpu")
    )

    assert cfg == TINY_CONFIG
    assert not model.training
    past = torch.randn(2, 8, 3)
    horizons = torch.arange(4)
    with torch.no_grad():
        expected = saved(past, horizons)
        loaded = model(past, horizons)
    assert loaded.keys() == expected.keys()
    for key, value in expected.items():
        torch.testing.assert_close(loaded[key], value)


def test_weights_stored_in_bf16_load_into_float32_parameters() -> None:
    saved = tiny_model(seed=3)
    checkpoint = checkpoint_mapping(saved)
    checkpoint[CHECKPOINT_MODEL_STATE_KEY] = {
        name: tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
        for name, tensor in checkpoint[CHECKPOINT_MODEL_STATE_KEY].items()
    }
    checkpoint[CHECKPOINT_RUNTIME_METADATA_KEY][CHECKPOINT_WEIGHTS_DTYPE_KEY] = "bf16"

    model, _ = load_model_from_checkpoint(checkpoint)

    for name, parameter in model.state_dict().items():
        if parameter.is_floating_point():
            assert parameter.dtype == torch.float32
            stored = checkpoint[CHECKPOINT_MODEL_STATE_KEY][name]
            assert torch.equal(parameter, stored.to(torch.float32))


def test_a_checkpoint_outside_the_format_is_refused() -> None:
    checkpoint = _checkpoint()
    checkpoint[CHECKPOINT_RUNTIME_METADATA_KEY][CHECKPOINT_FORMAT_VERSION_KEY] = 2
    with pytest.raises(ValueError, match="unsupported checkpoint format version 2"):
        load_model_from_checkpoint(checkpoint)


def test_another_generation_is_refused() -> None:
    checkpoint = _checkpoint()
    checkpoint[CHECKPOINT_RUNTIME_METADATA_KEY][CHECKPOINT_GENERATION_KEY] = (
        "another-generation"
    )
    with pytest.raises(ValueError, match="requires generation 'jointfm-1'"):
        load_model_from_checkpoint(checkpoint)


def test_a_cfg_with_an_unknown_field_is_refused() -> None:
    checkpoint = _checkpoint()
    checkpoint["cfg"] = {**asdict(TINY_CONFIG), "unknown_field": 1}
    with pytest.raises(ValueError, match=r"unknown \['unknown_field'\]"):
        load_model_from_checkpoint(checkpoint)


def test_a_cfg_with_a_missing_field_is_refused() -> None:
    checkpoint = _checkpoint()
    raw_cfg = asdict(TINY_CONFIG)
    del raw_cfg["pooling_seeds"]
    checkpoint["cfg"] = raw_cfg
    with pytest.raises(ValueError, match=r"missing \['pooling_seeds'\]"):
        load_model_from_checkpoint(checkpoint)


def test_a_state_dict_missing_a_parameter_is_refused() -> None:
    checkpoint = _checkpoint()
    state = dict(checkpoint["model_state"])
    del state["head.out_logits.weight"]
    checkpoint["model_state"] = state
    with pytest.raises(RuntimeError, match="head.out_logits.weight"):
        load_model_from_checkpoint(checkpoint)


def test_a_state_dict_with_an_unexpected_parameter_is_refused() -> None:
    checkpoint = _checkpoint()
    checkpoint["model_state"] = {
        **checkpoint["model_state"],
        "extra.weight": torch.ones(1),
    }
    with pytest.raises(RuntimeError, match="extra.weight"):
        load_model_from_checkpoint(checkpoint)
