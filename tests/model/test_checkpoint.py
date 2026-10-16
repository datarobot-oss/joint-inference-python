"""Tests for reading checkpoint files and their generation marker."""

import pickle
from pathlib import Path
from typing import Any

import pytest
import torch

from jointfm_inference.model.checkpoint import (
    JOINTFM_1_GENERATION,
    checkpoint_generation,
    read_checkpoint_file,
)


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


def test_the_generation_marker_is_read_from_the_runtime_metadata() -> None:
    checkpoint = {"runtime_metadata": {"generation": JOINTFM_1_GENERATION}}
    assert checkpoint_generation(checkpoint) == JOINTFM_1_GENERATION


@pytest.mark.parametrize(
    ("checkpoint", "error", "match"),
    [
        ({}, ValueError, "missing 'runtime_metadata'"),
        ({"runtime_metadata": ["jointfm-1"]}, TypeError, "must be a mapping"),
        ({"runtime_metadata": {}}, ValueError, "missing 'generation'"),
        ({"runtime_metadata": {"generation": ""}}, ValueError, "must not be empty"),
        ({"runtime_metadata": {"generation": 1}}, TypeError, "must be a string"),
    ],
)
def test_a_missing_or_malformed_generation_marker_is_refused(
    checkpoint: dict[str, Any], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        checkpoint_generation(checkpoint)
