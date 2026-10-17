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

"""Tests for the inference session and the checkpoint metadata it mounts with."""

import gc
import weakref
from pathlib import Path
from typing import Any

import pytest
import torch

import jointfm_inference.model.inference.checkpoint as inference_checkpoint
from jointfm_inference.data.types import CanonicalSample
from jointfm_inference.model.checkpoint import (
    CHECKPOINT_CAPACITY_ENVELOPE_KEY,
    CHECKPOINT_INFERENCE_METADATA_KEY,
    CHECKPOINT_MODEL_STATE_KEY,
    CHECKPOINT_TIME_INDEX_SUPPORT_KEY,
    JOINTFM_1_GENERATION,
    read_checkpoint_file,
)
from jointfm_inference.model.inference import (
    DEFAULT_INFERENCE_PRECISION,
    DISCRETE_GRID_TIME_INDEX_ENCODING,
    INFERENCE_RETURN_MODES,
    CanonicalInferenceRequest,
    InferenceRequestError,
    JointFM1DistributionResult,
    JointFMInferenceSession,
    TimeIndexSupport,
)
from jointfm_inference.model.inference.jointfm_1_backend import (
    DECODING_STRATEGY,
    JointFM1InferenceBackend,
)
from jointfm_inference.model.inference.time_index import (
    time_index_support_from_checkpoint,
)
from jointfm_inference.model.jointfm_1.model import JointFM

from tests.support.models import (
    TINY_CAPACITY_ENVELOPE,
    TINY_CONFIG,
    checkpoint_mapping,
    tiny_model,
    write_checkpoint,
)

_SERVED_TIME_INDEX_MODES = ("ordinal", "continuous_float", "absolute_datetime")


class _RecordingForward:
    """Wraps a network's own forward and records what each call saw."""

    def __init__(self, model: JointFM) -> None:
        """Capture the class forward bound to ``model`` before it is shadowed."""
        self._forward = model.forward
        self.autocast: tuple[bool, torch.dtype] | None = None
        self.past_time_indices: torch.Tensor | None = None
        self.output_dtypes: set[torch.dtype] = set()

    def __call__(
        self,
        past: torch.Tensor,
        horizons: torch.Tensor,
        past_time_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Record the autocast state and inputs, then run the real forward."""
        self.autocast = (
            torch.is_autocast_enabled("cpu"),
            torch.get_autocast_dtype("cpu"),
        )
        self.past_time_indices = past_time_indices
        outputs = self._forward(past, horizons, past_time_indices=past_time_indices)
        self.output_dtypes = {value.dtype for value in outputs.values()}
        return outputs


def _recorded_model() -> tuple[JointFM, _RecordingForward]:
    model = tiny_model()
    recorder = _RecordingForward(model)
    model.forward = recorder  # ty: ignore[invalid-assignment]
    return model, recorder


def _session(
    model: JointFM | None = None, *, precision: str = DEFAULT_INFERENCE_PRECISION
) -> JointFMInferenceSession:
    model = tiny_model() if model is None else model
    return JointFMInferenceSession(
        model,
        model.cfg,
        checkpoint_mapping(model),
        generation=JOINTFM_1_GENERATION,
        precision=precision,
    )


def _jointfm_1_backend(session: JointFMInferenceSession) -> JointFM1InferenceBackend:
    """Return the session's JointFM 1 backend, the owner of the tensor entry."""
    backend = session.backend
    assert isinstance(backend, JointFM1InferenceBackend)
    return backend


def _canonical_request(
    *, history_rows: int, column_count: int = 2
) -> CanonicalInferenceRequest:
    """Build a request on the unit grid with two query rows after the history."""
    rows = history_rows + 2
    values = torch.linspace(0.0, 1.0, rows * column_count).reshape(rows, column_count)
    query_row_mask = torch.arange(rows) >= history_rows
    values[query_row_mask] = 0.0
    return CanonicalInferenceRequest(
        sample=CanonicalSample(
            values=values,
            time_positions=torch.arange(rows, dtype=torch.float64),
            time_index_mode="continuous_float",
            query_row_mask=query_row_mask,
            metadata={"column_names": [f"c{index}" for index in range(column_count)]},
        )
    )


def test_the_session_reports_what_the_mounted_checkpoint_serves() -> None:
    model = tiny_model().train()

    session = _session(model)

    assert isinstance(session.backend, JointFM1InferenceBackend)
    assert session.generation == JOINTFM_1_GENERATION
    assert session.precision == DEFAULT_INFERENCE_PRECISION == "bf16"
    assert session.decoding_strategy == DECODING_STRATEGY == "parallel_dense"
    assert session.metadata_traits == ()
    assert session.time_index_support == TimeIndexSupport(
        encoding=DISCRETE_GRID_TIME_INDEX_ENCODING,
        supported_request_time_index_modes=_SERVED_TIME_INDEX_MODES,
        max_time_ctx=TINY_CONFIG.max_time_ctx,
        max_time_fut=TINY_CONFIG.max_time_fut,
    )
    assert session.capacity_envelope is not None
    assert session.capacity_envelope.to_payload() == TINY_CAPACITY_ENVELOPE
    assert session.backend.head_name == "studentt"
    assert session.backend.supported_return_modes == INFERENCE_RETURN_MODES
    assert not model.training


def test_a_checkpoint_without_an_envelope_is_refused() -> None:
    """Every served model reports the request sizes it was built for."""
    model = tiny_model()
    checkpoint = checkpoint_mapping(model)
    del checkpoint[CHECKPOINT_INFERENCE_METADATA_KEY][CHECKPOINT_CAPACITY_ENVELOPE_KEY]

    with pytest.raises(ValueError, match="missing 'data_generation'"):
        JointFMInferenceSession(
            model, model.cfg, checkpoint, generation=JOINTFM_1_GENERATION
        )


def test_a_checkpoint_without_time_index_support_is_refused() -> None:
    """The time-index contract is read from the checkpoint, never derived elsewhere."""
    model = tiny_model()
    checkpoint = checkpoint_mapping(model)
    del checkpoint[CHECKPOINT_INFERENCE_METADATA_KEY][CHECKPOINT_TIME_INDEX_SUPPORT_KEY]

    with pytest.raises(ValueError, match="missing 'time_index_support'"):
        JointFMInferenceSession(
            model, model.cfg, checkpoint, generation=JOINTFM_1_GENERATION
        )


def test_the_session_keeps_no_reference_to_the_checkpoint() -> None:
    """A served process must not hold a second copy of the weights."""
    model = tiny_model()
    checkpoint = checkpoint_mapping(model)
    state_refs = [
        weakref.ref(tensor)
        for tensor in checkpoint[CHECKPOINT_MODEL_STATE_KEY].values()
    ]

    session = JointFMInferenceSession(
        model, model.cfg, checkpoint, generation=JOINTFM_1_GENERATION
    )
    del checkpoint
    gc.collect()

    assert state_refs
    assert all(ref() is None for ref in state_refs)
    assert session.capacity_envelope is not None


def test_a_session_loaded_from_a_file_keeps_no_reference_to_the_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "model.pt"
    write_checkpoint(path)
    state_refs: list[weakref.ref[torch.Tensor]] = []

    def read_and_track(path: str, *, map_location: Any) -> dict[str, Any]:
        checkpoint = read_checkpoint_file(path, map_location=map_location)
        state_refs.extend(
            weakref.ref(tensor)
            for tensor in checkpoint[CHECKPOINT_MODEL_STATE_KEY].values()
        )
        return checkpoint

    monkeypatch.setattr(inference_checkpoint, "read_checkpoint_file", read_and_track)

    session = JointFMInferenceSession.from_checkpoint(path)
    gc.collect()

    assert session.generation == JOINTFM_1_GENERATION
    assert state_refs
    assert all(ref() is None for ref in state_refs)


def test_from_checkpoint_predicts_like_the_model_that_wrote_it(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    model = write_checkpoint(path)
    past = torch.randn(1, 5, 2, generator=torch.Generator().manual_seed(0))
    horizons = torch.arange(3)

    loaded = JointFMInferenceSession.from_checkpoint(
        path, device="cpu", precision="fp32"
    )
    original = _session(model, precision="fp32")

    assert loaded.generation == JOINTFM_1_GENERATION
    assert loaded.precision == "fp32"
    assert loaded.time_index_support == original.time_index_support
    assert loaded.capacity_envelope == original.capacity_envelope
    loaded_params = (
        _jointfm_1_backend(loaded).predict_distribution(past, horizons).params
    )
    original_params = (
        _jointfm_1_backend(original).predict_distribution(past, horizons).params
    )
    assert loaded_params.keys() == original_params.keys()
    for name, value in original_params.items():
        assert torch.equal(loaded_params[name], value), name


def test_the_default_precision_runs_under_bf16_and_hands_float32_to_the_adapter() -> (
    None
):
    model, recorder = _recorded_model()

    adapter = _jointfm_1_backend(_session(model)).predict_distribution(
        torch.randn(1, 4, 2), torch.arange(3)
    )

    assert recorder.autocast == (True, torch.bfloat16)
    assert torch.bfloat16 in recorder.output_dtypes
    assert {value.dtype for value in adapter.params.values()} == {torch.float32}


def test_fp32_precision_disables_autocast() -> None:
    model, recorder = _recorded_model()

    _jointfm_1_backend(_session(model, precision="fp32")).predict_distribution(
        torch.randn(1, 4, 2), torch.arange(3)
    )

    assert recorder.autocast is not None
    assert recorder.autocast[0] is False


@pytest.mark.parametrize(
    ("precision", "match"),
    [("fp16", "'fp16' only on CUDA"), ("tf32", "precision to be one of")],
)
def test_the_session_rejects_a_precision_the_device_cannot_serve(
    precision: str, match: str
) -> None:
    """An unservable precision fails at construction, not on the first request."""
    with pytest.raises(ValueError, match=match):
        _session(precision=precision)


def test_the_session_rejects_a_model_without_parameters() -> None:
    model = torch.nn.Module()

    with pytest.raises(ValueError, match="no parameters"):
        JointFMInferenceSession(
            model,
            TINY_CONFIG,
            checkpoint_mapping(tiny_model()),
            generation=JOINTFM_1_GENERATION,
        )


def test_the_backend_refuses_a_model_that_is_not_a_jointfm_1_network() -> None:
    with pytest.raises(TypeError, match="JointFM 1 network, got Linear"):
        JointFMInferenceSession(
            torch.nn.Linear(2, 1),
            TINY_CONFIG,
            checkpoint_mapping(tiny_model()),
            generation=JOINTFM_1_GENERATION,
        )


def test_the_tensor_entry_forwards_grid_positions_to_the_network() -> None:
    model, recorder = _recorded_model()
    past_time_indices = torch.tensor([0, 1, 3, 4])

    _jointfm_1_backend(_session(model)).predict_distribution(
        torch.randn(1, 4, 2), torch.arange(3), past_time_indices=past_time_indices
    )

    assert recorder.past_time_indices is not None
    assert torch.equal(recorder.past_time_indices, past_time_indices)


def test_execute_answers_a_canonical_request() -> None:
    model, recorder = _recorded_model()
    session = _session(model)

    result = session.execute(_canonical_request(history_rows=4))

    assert isinstance(result, JointFM1DistributionResult)
    assert recorder.past_time_indices is not None
    assert torch.equal(recorder.past_time_indices, torch.arange(4))
    assert result.samples(n_samples=3).shape == (1, 3, 2, 2)
    assert result.mean(n_samples=4).shape == (1, 2, 2)


def test_execute_refuses_a_history_longer_than_the_time_capacity() -> None:
    session = _session()

    with pytest.raises(InferenceRequestError) as refusal:
        session.execute(_canonical_request(history_rows=TINY_CONFIG.max_time_ctx + 1))

    assert refusal.value.field == "history_rows"


def _time_index_block(**overrides: Any) -> dict[str, Any]:
    block = checkpoint_mapping(tiny_model())[CHECKPOINT_INFERENCE_METADATA_KEY][
        CHECKPOINT_TIME_INDEX_SUPPORT_KEY
    ]
    return {**block, **overrides}


def _with_time_index_block(block: Any) -> dict[str, Any]:
    return {
        CHECKPOINT_INFERENCE_METADATA_KEY: {CHECKPOINT_TIME_INDEX_SUPPORT_KEY: block}
    }


def test_time_index_support_is_read_from_the_checkpoint() -> None:
    support = time_index_support_from_checkpoint(
        _with_time_index_block(_time_index_block(max_time_ctx=8, max_time_fut=3))
    )

    assert support == TimeIndexSupport(
        encoding=DISCRETE_GRID_TIME_INDEX_ENCODING,
        supported_request_time_index_modes=_SERVED_TIME_INDEX_MODES,
        max_time_ctx=8,
        max_time_fut=3,
    )


@pytest.mark.parametrize(
    ("checkpoint", "error", "message"),
    [
        ({}, ValueError, "missing 'inference_metadata'"),
        (
            {CHECKPOINT_INFERENCE_METADATA_KEY: {}},
            ValueError,
            "missing 'time_index_support'",
        ),
        (
            _with_time_index_block(_time_index_block(encoding="continuous")),
            ValueError,
            "unsupported time-index encoding",
        ),
        (
            _with_time_index_block(
                _time_index_block(supported_request_time_index_modes="ordinal")
            ),
            TypeError,
            "must be a list of strings",
        ),
        (
            _with_time_index_block(_time_index_block(max_time_ctx="8")),
            TypeError,
            "'max_time_ctx' must be a positive int",
        ),
        (
            _with_time_index_block(_time_index_block(max_time_ctx=True)),
            TypeError,
            "'max_time_ctx' must be a positive int",
        ),
        (
            _with_time_index_block(_time_index_block(max_time_fut=0)),
            TypeError,
            "'max_time_fut' must be a positive int",
        ),
        (
            _with_time_index_block(_time_index_block(max_time_fut=None)),
            TypeError,
            "'max_time_fut' must be a positive int",
        ),
    ],
    ids=[
        "no_metadata",
        "no_block",
        "other_encoding",
        "modes_string",
        "ctx_string",
        "ctx_bool",
        "fut_zero",
        "fut_missing",
    ],
)
def test_time_index_support_refuses_a_malformed_block(
    checkpoint: dict[str, Any], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        time_index_support_from_checkpoint(checkpoint)
