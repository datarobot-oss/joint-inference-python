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

"""Tests for rebuilding a JointFM 1 network from a checkpoint mapping."""

from dataclasses import asdict
from typing import Any, Callable

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


def _with_slot_embedding(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Add the retired per-column slot embedding and the ``combine`` that read it."""
    d_model = TINY_CONFIG.d_model
    return {
        **state,
        "var_embed.weight": torch.zeros(TINY_CONFIG.column_capacity, d_model),
        "combine.weight": torch.zeros(d_model, TINY_CONFIG.d_value + d_model),
    }


def _with_slot_indexed_locations(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Replace the shared location bases by the retired slot-indexed output rows."""
    rows = TINY_CONFIG.n_mixtures * TINY_CONFIG.column_capacity
    retired = {
        name: value
        for name, value in state.items()
        if not name.startswith(("head.mean_base.", "head.raw_diag_base."))
    }
    for projection in ("out_means", "out_diag"):
        retired[f"head.{projection}.weight"] = torch.zeros(rows, TINY_CONFIG.d_model)
        retired[f"head.{projection}.bias"] = torch.zeros(rows)
    return retired


def _with_column_slot_layout(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Rewrite ``state`` into the whole retired layout."""
    return _with_slot_indexed_locations(_with_slot_embedding(state))


SLOT_EMBEDDING_ERRORS = ('"var_embed.weight"', "size mismatch for combine.weight")
SLOT_INDEXED_LOCATION_ERRORS = (
    '"head.out_means.weight"',
    '"head.out_diag.weight"',
    '"head.mean_base.weight"',
    '"head.raw_diag_base.weight"',
)


@pytest.mark.parametrize(
    ("retire", "named"),
    [
        (_with_slot_embedding, SLOT_EMBEDDING_ERRORS),
        (_with_slot_indexed_locations, SLOT_INDEXED_LOCATION_ERRORS),
        (
            _with_column_slot_layout,
            SLOT_EMBEDDING_ERRORS + SLOT_INDEXED_LOCATION_ERRORS,
        ),
    ],
    ids=["slot-embedding", "slot-indexed-locations", "whole-layout"],
)
def test_an_export_with_column_slot_parameters_is_refused_by_name(
    retire: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
    named: tuple[str, ...],
) -> None:
    """The network holds no parameter indexed by column position.

    An export that still carries the per-column slot embedding or the
    slot-indexed location rows fails the strict load, and the error names
    every retired, missing, or reshaped parameter.
    """
    checkpoint = _checkpoint()
    checkpoint["model_state"] = retire(dict(checkpoint["model_state"]))

    with pytest.raises(RuntimeError) as refusal:
        load_model_from_checkpoint(checkpoint)

    for fragment in named:
        assert fragment in str(refusal.value)
