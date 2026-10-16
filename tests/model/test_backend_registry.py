"""The inference backend registry resolves a generation to exactly one backend."""

from __future__ import annotations

from importlib.metadata import EntryPoint, EntryPoints
from pathlib import Path

import pytest
import torch

import jointfm_inference.model.inference.backend_registry as backend_registry
from jointfm_inference.model.checkpoint import (
    CHECKPOINT_GENERATION_KEY,
    CHECKPOINT_RUNTIME_METADATA_KEY,
    JOINTFM_1_GENERATION,
)
from jointfm_inference.model.inference import (
    INFERENCE_BACKEND_ENTRY_POINT_GROUP,
    JointFMInferenceSession,
    ServingBackend,
    UnregisteredGenerationError,
    registered_generations,
    resolve_serving_backend,
)
from jointfm_inference.model.inference.checkpoint import load_inference_components
from jointfm_inference.model.jointfm_1.model import JointFM

from tests.support.models import checkpoint_mapping, tiny_model

_JOINTFM_1_VALUE = (
    "jointfm_inference.model.inference.jointfm_1_backend:JOINTFM_1_SERVING_BACKEND"
)
_OTHER_GENERATION = "another-generation"


def _install_entry_points(
    monkeypatch: pytest.MonkeyPatch, registered: list[tuple[str, str]]
) -> None:
    """Make the registry see exactly ``registered`` as ``(name, value)`` entries."""
    installed = EntryPoints(
        EntryPoint(name, value, INFERENCE_BACKEND_ENTRY_POINT_GROUP)
        for name, value in registered
    )

    def fake_entry_points(*, group: str, name: str | None = None) -> EntryPoints:
        if name is None:
            return installed.select(group=group)
        return installed.select(group=group, name=name)

    monkeypatch.setattr(backend_registry, "entry_points", fake_entry_points)


def test_the_installed_distribution_registers_jointfm_1() -> None:
    assert registered_generations() == (JOINTFM_1_GENERATION,)
    for generation in registered_generations():
        backend = resolve_serving_backend(generation)
        assert isinstance(backend, ServingBackend)
        assert backend.generation == generation


def test_an_unregistered_generation_names_the_registered_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(monkeypatch, [(JOINTFM_1_GENERATION, _JOINTFM_1_VALUE)])

    with pytest.raises(UnregisteredGenerationError, match=r"\['jointfm-1'\]"):
        resolve_serving_backend(_OTHER_GENERATION)


def test_two_registrations_of_one_generation_are_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(
        monkeypatch,
        [
            (JOINTFM_1_GENERATION, _JOINTFM_1_VALUE),
            (JOINTFM_1_GENERATION, "other.backend:BACKEND"),
        ],
    )

    with pytest.raises(RuntimeError, match="several serving backends"):
        resolve_serving_backend(JOINTFM_1_GENERATION)


def test_a_registered_object_that_is_no_backend_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(monkeypatch, [(JOINTFM_1_GENERATION, "builtins:object")])

    with pytest.raises(TypeError, match="not a ServingBackend"):
        resolve_serving_backend(JOINTFM_1_GENERATION)


def test_a_backend_registered_under_another_generation_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(monkeypatch, [(_OTHER_GENERATION, _JOINTFM_1_VALUE)])

    with pytest.raises(ValueError, match="serves 'jointfm-1'"):
        resolve_serving_backend(_OTHER_GENERATION)


def _refuse_network_construction(*args: object, **kwargs: object) -> None:
    raise AssertionError("no network may be built for an unregistered generation")


def test_an_unregistered_generation_fails_at_load_before_any_network_is_built(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A checkpoint whose generation no backend serves stops before any network exists."""
    checkpoint = checkpoint_mapping(tiny_model())
    checkpoint[CHECKPOINT_RUNTIME_METADATA_KEY] = {
        CHECKPOINT_GENERATION_KEY: _OTHER_GENERATION
    }
    path = tmp_path / "other.pt"
    torch.save(checkpoint, path)
    monkeypatch.setattr(JointFM, "__init__", _refuse_network_construction)

    with pytest.raises(UnregisteredGenerationError, match="'another-generation'"):
        load_inference_components(str(path))
    with pytest.raises(UnregisteredGenerationError, match="'another-generation'"):
        JointFMInferenceSession.from_checkpoint(path)


def test_a_session_refuses_an_unregistered_generation() -> None:
    model = tiny_model()

    with pytest.raises(UnregisteredGenerationError, match="'another-generation'"):
        JointFMInferenceSession(
            model, model.cfg, checkpoint_mapping(model), generation=_OTHER_GENERATION
        )
