"""Tests for rebuilding a JointFM 1 network from a checkpoint mapping."""

from dataclasses import asdict
from typing import Any

import pytest
import torch

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


@pytest.mark.parametrize(
    "missing", ["cfg", "model_state", "runtime_metadata", "inference_metadata"]
)
def test_a_missing_top_level_entry_is_refused(missing: str) -> None:
    checkpoint = _checkpoint()
    del checkpoint[missing]
    with pytest.raises(
        ValueError, match=rf"not a JointFM 1 checkpoint: missing \['{missing}'\]"
    ):
        load_model_from_checkpoint(checkpoint)


def test_an_unknown_top_level_entry_is_refused() -> None:
    checkpoint = _checkpoint()
    checkpoint["notes"] = {}
    with pytest.raises(ValueError, match=r"unknown \['notes'\]"):
        load_model_from_checkpoint(checkpoint)


def test_another_generation_is_refused() -> None:
    checkpoint = _checkpoint()
    checkpoint["runtime_metadata"] = {"generation": "another-generation"}
    with pytest.raises(ValueError, match="requires generation 'jointfm-1'"):
        load_model_from_checkpoint(checkpoint)


@pytest.mark.parametrize("entry", ["cfg", "model_state"])
def test_a_non_mapping_cfg_or_state_is_refused(entry: str) -> None:
    checkpoint = _checkpoint()
    checkpoint[entry] = 5
    with pytest.raises(TypeError, match=f"'{entry}' must be a mapping"):
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
