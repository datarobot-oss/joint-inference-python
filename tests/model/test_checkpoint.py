"""Tests for reading checkpoint files and the checkpoint format contract."""

import pickle
from pathlib import Path
from typing import Any

import pytest
import torch

from jointfm_inference.model.checkpoint import (
    CHECKPOINT_CAPACITY_ENVELOPE_KEY,
    CHECKPOINT_CFG_KEY,
    CHECKPOINT_FORMAT_VERSION,
    CHECKPOINT_FORMAT_VERSION_KEY,
    CHECKPOINT_GENERATION_KEY,
    CHECKPOINT_INFERENCE_METADATA_KEY,
    CHECKPOINT_MODEL_STATE_KEY,
    CHECKPOINT_RUNTIME_METADATA_KEY,
    CHECKPOINT_TIME_INDEX_SUPPORT_KEY,
    CHECKPOINT_WEIGHTS_DTYPE_KEY,
    CHECKPOINT_WEIGHTS_DTYPES,
    JOINTFM_1_GENERATION,
    read_checkpoint_file,
    require_checkpoint_format,
)
from tests.support.models import checkpoint_mapping, tiny_model


class _Opaque:
    """A class a plain-data load must refuse to rebuild."""


def test_a_plain_data_checkpoint_reads_back_as_a_dict(tmp_path: Path) -> None:
    path = tmp_path / "plain.pt"
    weights = torch.arange(3.0)
    torch.save(
        {"model_state": {"w": weights}, "runtime_metadata": {"generation": "x"}}, path
    )

    checkpoint = read_checkpoint_file(str(path))

    assert type(checkpoint) is dict
    torch.testing.assert_close(checkpoint["model_state"]["w"], weights)
    assert checkpoint["runtime_metadata"] == {"generation": "x"}


def test_a_file_holding_a_non_mapping_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "list.pt"
    torch.save([1, 2, 3], path)
    with pytest.raises(TypeError, match="must hold a mapping"):
        read_checkpoint_file(str(path))


def test_a_file_holding_arbitrary_objects_is_not_unpickled(tmp_path: Path) -> None:
    path = tmp_path / "opaque.pt"
    torch.save({"payload": _Opaque()}, path)
    with pytest.raises(pickle.UnpicklingError):
        read_checkpoint_file(str(path))


def _checkpoint() -> dict[str, Any]:
    """Return a checkpoint mapping that follows the format."""
    return checkpoint_mapping(tiny_model())


def _runtime_metadata(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return checkpoint[CHECKPOINT_RUNTIME_METADATA_KEY]


def test_a_checkpoint_in_the_format_returns_its_generation() -> None:
    assert require_checkpoint_format(_checkpoint()) == JOINTFM_1_GENERATION


@pytest.mark.parametrize(
    "missing",
    [
        CHECKPOINT_CFG_KEY,
        CHECKPOINT_MODEL_STATE_KEY,
        CHECKPOINT_RUNTIME_METADATA_KEY,
        CHECKPOINT_INFERENCE_METADATA_KEY,
    ],
)
def test_a_missing_top_level_entry_is_refused(missing: str) -> None:
    checkpoint = _checkpoint()
    del checkpoint[missing]
    with pytest.raises(
        ValueError, match=rf"^checkpoint does not match .* missing \['{missing}'\]"
    ):
        require_checkpoint_format(checkpoint)


@pytest.mark.parametrize("unknown", ["optimizer_state", "checkpoint_dtype"])
def test_a_training_entry_outside_the_format_is_refused(unknown: str) -> None:
    checkpoint = _checkpoint()
    checkpoint[unknown] = {}
    with pytest.raises(ValueError, match=rf"unknown \['{unknown}'\]"):
        require_checkpoint_format(checkpoint)


def test_a_checkpoint_without_a_format_version_is_refused() -> None:
    checkpoint = _checkpoint()
    del _runtime_metadata(checkpoint)[CHECKPOINT_FORMAT_VERSION_KEY]
    with pytest.raises(ValueError, match="missing 'format_version'"):
        require_checkpoint_format(checkpoint)


@pytest.mark.parametrize("version", [0, CHECKPOINT_FORMAT_VERSION + 1])
def test_an_unknown_format_version_is_refused(version: int) -> None:
    checkpoint = _checkpoint()
    _runtime_metadata(checkpoint)[CHECKPOINT_FORMAT_VERSION_KEY] = version
    with pytest.raises(
        ValueError, match=f"unsupported checkpoint format version {version};"
    ):
        require_checkpoint_format(checkpoint)


def test_an_unknown_version_is_refused_for_its_version_not_its_layout() -> None:
    """A later layout fails on the version it declares, not deep in its entries."""
    checkpoint = _checkpoint()
    runtime_metadata = _runtime_metadata(checkpoint)
    runtime_metadata[CHECKPOINT_FORMAT_VERSION_KEY] = CHECKPOINT_FORMAT_VERSION + 1
    runtime_metadata["field_of_a_later_version"] = True
    del runtime_metadata[CHECKPOINT_WEIGHTS_DTYPE_KEY]
    with pytest.raises(ValueError, match="unsupported checkpoint format version"):
        require_checkpoint_format(checkpoint)


@pytest.mark.parametrize("version", [True, "1", 1.0])
def test_a_non_int_format_version_is_refused(version: object) -> None:
    checkpoint = _checkpoint()
    _runtime_metadata(checkpoint)[CHECKPOINT_FORMAT_VERSION_KEY] = version
    with pytest.raises(TypeError, match="format version must be an int"):
        require_checkpoint_format(checkpoint)


def test_runtime_metadata_that_is_not_a_mapping_is_refused() -> None:
    checkpoint = _checkpoint()
    checkpoint[CHECKPOINT_RUNTIME_METADATA_KEY] = [JOINTFM_1_GENERATION]
    with pytest.raises(TypeError, match="'runtime_metadata' must be a mapping"):
        require_checkpoint_format(checkpoint)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"world": "legacy"}, r"unknown \['world'\]"),
        ({CHECKPOINT_GENERATION_KEY: None}, r"missing \['generation'\]"),
        ({CHECKPOINT_WEIGHTS_DTYPE_KEY: None}, r"missing \['weights_dtype'\]"),
    ],
)
def test_runtime_metadata_holds_exactly_its_entries(
    change: dict[str, Any], match: str
) -> None:
    checkpoint = _checkpoint()
    runtime_metadata = _runtime_metadata(checkpoint)
    for key, value in change.items():
        if value is None:
            del runtime_metadata[key]
        else:
            runtime_metadata[key] = value
    with pytest.raises(ValueError, match=match):
        require_checkpoint_format(checkpoint)


@pytest.mark.parametrize(
    ("generation", "error", "match"),
    [("", ValueError, "must not be empty"), (1, TypeError, "must be a string")],
)
def test_a_malformed_generation_marker_is_refused(
    generation: object, error: type[Exception], match: str
) -> None:
    checkpoint = _checkpoint()
    _runtime_metadata(checkpoint)[CHECKPOINT_GENERATION_KEY] = generation
    with pytest.raises(error, match=match):
        require_checkpoint_format(checkpoint)


@pytest.mark.parametrize("entry", [CHECKPOINT_CFG_KEY, CHECKPOINT_MODEL_STATE_KEY])
def test_a_non_mapping_cfg_or_state_is_refused(entry: str) -> None:
    checkpoint = _checkpoint()
    checkpoint[entry] = 5
    with pytest.raises(TypeError, match=f"'{entry}' must be a mapping"):
        require_checkpoint_format(checkpoint)


@pytest.mark.parametrize("label", ["float32", "int8", 32])
def test_an_unknown_weights_dtype_label_is_refused(label: object) -> None:
    checkpoint = _checkpoint()
    _runtime_metadata(checkpoint)[CHECKPOINT_WEIGHTS_DTYPE_KEY] = label
    with pytest.raises(ValueError, match="weights dtype must be one of"):
        require_checkpoint_format(checkpoint)


def test_a_weight_stored_in_another_dtype_than_the_label_is_refused() -> None:
    checkpoint = _checkpoint()
    state = dict(checkpoint[CHECKPOINT_MODEL_STATE_KEY])
    state["head.out_logits.weight"] = state["head.out_logits.weight"].to(torch.bfloat16)
    checkpoint[CHECKPOINT_MODEL_STATE_KEY] = state
    with pytest.raises(
        ValueError,
        match=r"is 'fp32', but .* 'head.out_logits.weight' is torch.bfloat16",
    ):
        require_checkpoint_format(checkpoint)


@pytest.mark.parametrize("label", sorted(CHECKPOINT_WEIGHTS_DTYPES))
def test_every_weights_dtype_label_is_checked_against_the_stored_tensors(
    label: str,
) -> None:
    checkpoint = _checkpoint()
    dtype = CHECKPOINT_WEIGHTS_DTYPES[label]
    checkpoint[CHECKPOINT_MODEL_STATE_KEY] = {
        name: tensor.to(dtype) if tensor.is_floating_point() else tensor
        for name, tensor in checkpoint[CHECKPOINT_MODEL_STATE_KEY].items()
    }
    _runtime_metadata(checkpoint)[CHECKPOINT_WEIGHTS_DTYPE_KEY] = label

    assert require_checkpoint_format(checkpoint) == JOINTFM_1_GENERATION


def test_a_non_tensor_state_entry_is_refused() -> None:
    checkpoint = _checkpoint()
    checkpoint[CHECKPOINT_MODEL_STATE_KEY] = {
        **checkpoint[CHECKPOINT_MODEL_STATE_KEY],
        "step": 3,
    }
    with pytest.raises(TypeError, match="entry 'step' must be a tensor, got int"):
        require_checkpoint_format(checkpoint)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"training_config": {}}, r"unknown \['training_config'\]"),
        (
            {CHECKPOINT_TIME_INDEX_SUPPORT_KEY: None},
            r"missing \['time_index_support'\]",
        ),
        ({CHECKPOINT_CAPACITY_ENVELOPE_KEY: None}, r"missing \['data_generation'\]"),
    ],
)
def test_inference_metadata_holds_exactly_its_blocks(
    change: dict[str, Any], match: str
) -> None:
    checkpoint = _checkpoint()
    inference_metadata = checkpoint[CHECKPOINT_INFERENCE_METADATA_KEY]
    for key, value in change.items():
        if value is None:
            del inference_metadata[key]
        else:
            inference_metadata[key] = value
    with pytest.raises(ValueError, match=match):
        require_checkpoint_format(checkpoint)


def test_inference_metadata_that_is_not_a_mapping_is_refused() -> None:
    checkpoint = _checkpoint()
    checkpoint[CHECKPOINT_INFERENCE_METADATA_KEY] = []
    with pytest.raises(TypeError, match="'inference_metadata' must be a mapping"):
        require_checkpoint_format(checkpoint)
