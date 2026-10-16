"""Inference session over one loaded JointFM checkpoint.

The session mounts the model on the serving backend the inference backend
registry holds for the checkpoint's generation and forwards every capability
query and every request to it, so it names no generation itself:
:meth:`JointFMInferenceSession.execute` answers one canonical request at the
session's precision. A generation-specific entry point, such as the JointFM 1
tensor entry ``predict_distribution``, lives on :attr:`JointFMInferenceSession.backend`.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Union

import torch

from jointfm_inference.model.precision import resolve_autocast_dtype

from .backend_registry import resolve_serving_backend
from .capacity_envelope import CapacityEnvelope, capacity_envelope_from_checkpoint
from .checkpoint import load_inference_components
from .contract import (
    CanonicalInferenceRequest,
    InferenceBackend,
    InferenceDistributionResult,
)
from .time_index import TimeIndexSupport

# Runs matmuls and attention on tensor cores at half the activation memory.
DEFAULT_INFERENCE_PRECISION = "bf16"


class JointFMInferenceSession:
    """A loaded model mounted on the serving backend of its generation."""

    def __init__(
        self,
        model: torch.nn.Module,
        model_cfg: Any,
        checkpoint: dict[str, Any],
        *,
        generation: str,
        precision: str = DEFAULT_INFERENCE_PRECISION,
    ) -> None:
        """Mount ``model`` on the backend registered for ``generation``.

        ``checkpoint`` supplies the metadata the backend and the health
        payload read; the session keeps no reference to it, so its tensors
        are freed once the caller drops it. ``precision`` (``fp32`` | ``bf16``
        | ``fp16``) selects the autocast dtype of every forward and is
        validated against the device of the model's parameters.

        Raises:
            UnregisteredGenerationError: when no serving backend is
                registered for ``generation``.
            TypeError: when the backend refuses the model.
            ValueError: when the model has no parameters or ``precision``
                cannot run on their device.
        """
        serving_backend = resolve_serving_backend(generation)
        first_parameter = next(model.parameters(), None)
        if first_parameter is None:
            raise ValueError(
                "model has no parameters, so the session cannot resolve the "
                "device its precision must run on"
            )
        autocast_dtype = resolve_autocast_dtype(
            precision, first_parameter.device, setting="precision"
        )
        self._backend = serving_backend.mount(
            model,
            model_cfg,
            checkpoint,
            device=first_parameter.device,
            autocast_dtype=autocast_dtype,
        )
        model.eval()
        self._precision = precision
        self._capacity_envelope = capacity_envelope_from_checkpoint(checkpoint)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | os.PathLike[str],
        *,
        map_location: Union[str, torch.device] = "cpu",
        device: Optional[Union[str, torch.device]] = None,
        precision: str = DEFAULT_INFERENCE_PRECISION,
    ) -> "JointFMInferenceSession":
        """Load a checkpoint file through the backend registered for its generation."""
        generation, model, model_cfg, checkpoint = load_inference_components(
            os.fspath(checkpoint_path), map_location=map_location, device=device
        )
        return cls(
            model=model,
            model_cfg=model_cfg,
            checkpoint=checkpoint,
            generation=generation,
            precision=precision,
        )

    @property
    def backend(self) -> InferenceBackend:
        """Return the serving backend the model is mounted on."""
        return self._backend

    @property
    def generation(self) -> str:
        """Return the served checkpoint generation marker."""
        return self._backend.generation

    @property
    def precision(self) -> str:
        """Return the model precision every forward runs at."""
        return self._precision

    @property
    def metadata_traits(self) -> tuple[str, ...]:
        """Return the metadata traits the mounted model reads as expert hints."""
        return self._backend.metadata_traits

    @property
    def time_index_support(self) -> TimeIndexSupport:
        """Return the time-index contract the mounted model reads request times under."""
        return self._backend.time_index_support

    @property
    def decoding_strategy(self) -> str:
        """Return how the mounted model decodes forecast horizons."""
        return self._backend.decoding_strategy

    @property
    def capacity_envelope(self) -> CapacityEnvelope:
        """Return the capacity envelope the checkpoint records."""
        return self._capacity_envelope

    def execute(
        self, request: CanonicalInferenceRequest
    ) -> InferenceDistributionResult:
        """Answer one stateless canonical request on the mounted backend.

        Raises:
            InferenceRequestError: when the request asks for something the
                mounted model cannot represent.
        """
        return self._backend.execute(request)


__all__ = ["DEFAULT_INFERENCE_PRECISION", "JointFMInferenceSession"]
