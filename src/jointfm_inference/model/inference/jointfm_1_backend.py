"""The serving backend of JointFM 1 checkpoints (generation ``jointfm-1``).

Registered in the inference backend registry as :data:`JOINTFM_1_SERVING_BACKEND`.
A JointFM 1 network reads a batch of history values on a discrete time grid
and emits the parameters of its Student-t mixture head, so the mounted backend
maps each canonical request onto that grid (``jointfm_1_grid``) and wraps the
emitted parameters in a :class:`PredictorAdapter`. The tensor entry point
:meth:`JointFM1InferenceBackend.predict_distribution` serves callers that
already hold grid tensors.
"""

from __future__ import annotations

from typing import Any, Final

import torch

from jointfm_inference.model.checkpoint import JOINTFM_1_GENERATION
from jointfm_inference.model.jointfm_1.config import JointFMConfig
from jointfm_inference.model.jointfm_1.head import StudentTHead
from jointfm_inference.model.jointfm_1.model import JointFM
from jointfm_inference.model.precision import autocast_context, upcast_reduced_precision

from .adapter import PredictorAdapter
from .contract import (
    INFERENCE_RETURN_MODES,
    CanonicalInferenceRequest,
    InferenceDistributionResult,
    JointFM1DistributionResult,
    normalize_head_name,
)
from .jointfm_1_grid import grid_model_inputs
from .time_index import TimeIndexSupport, time_index_support_from_checkpoint

# Every queried horizon is decoded in one forward pass.
DECODING_STRATEGY: Final = "parallel_dense"


class JointFM1InferenceBackend:
    """A JointFM 1 network mounted for serving.

    It serves every return mode and conditioning, and reads no metadata
    traits.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        checkpoint: dict[str, Any],
        *,
        device: torch.device,
        autocast_dtype: torch.dtype | None,
    ) -> None:
        """Mount ``model`` with the time-index support its checkpoint records.

        Raises:
            TypeError: when ``model`` is not a JointFM 1 network.
        """
        if not isinstance(model, JointFM):
            raise TypeError(
                f"model must be a JointFM 1 network, got {type(model).__name__}"
            )
        self._model = model
        self._head: StudentTHead = model.head
        self._device = device
        self._autocast_dtype = autocast_dtype
        self._time_index_support = time_index_support_from_checkpoint(checkpoint)

    @property
    def generation(self) -> str:
        """Return ``'jointfm-1'``."""
        return JOINTFM_1_GENERATION

    @property
    def head_name(self) -> str:
        """Return the head class name normalized by :func:`normalize_head_name`."""
        return normalize_head_name(type(self._head).__name__)

    @property
    def time_index_support(self) -> TimeIndexSupport:
        """Return the discrete-grid time-index contract of the checkpoint."""
        return self._time_index_support

    @property
    def decoding_strategy(self) -> str:
        """Return ``'parallel_dense'``."""
        return DECODING_STRATEGY

    @property
    def metadata_traits(self) -> tuple[str, ...]:
        """Return no traits: JointFM 1 reads no expert hints."""
        return ()

    @property
    def supported_return_modes(self) -> frozenset[str]:
        """Return every return mode."""
        return INFERENCE_RETURN_MODES

    @property
    def supports_conditioning(self) -> bool:
        """Return ``True``: the Student-t mixture has a closed-form conditional."""
        return True

    def execute(
        self, request: CanonicalInferenceRequest
    ) -> InferenceDistributionResult:
        """Map the request onto the time grid and run one forward."""
        past, past_time_indices, horizons = grid_model_inputs(
            request.sample,
            time_index_support=self._time_index_support,
            device=self._device,
        )
        predictor = self.predict_distribution(
            past, horizons, past_time_indices=past_time_indices
        )
        return JointFM1DistributionResult(
            predictor, active_column_count=request.column_count
        )

    def predict_distribution(
        self,
        past: torch.Tensor,
        horizons: torch.Tensor,
        past_time_indices: torch.Tensor | None = None,
    ) -> PredictorAdapter:
        """Run the network on grid tensors and return its predictive distribution.

        ``past`` is a ``(B, T, D)`` batch of history values, ``horizons`` the
        ``(H,)`` horizon indices to emit, and ``past_time_indices`` the
        optional ``(T,)`` grid positions of the history rows. The forward runs
        at the session's precision; the adapter receives float32 parameters.
        """
        with torch.no_grad(), autocast_context(self._device.type, self._autocast_dtype):
            raw_params = self._model(
                past, horizons, past_time_indices=past_time_indices
            )
        return PredictorAdapter(self._head, upcast_reduced_precision(raw_params))


class JointFM1ServingBackend:
    """Registry entry of generation ``jointfm-1``."""

    @property
    def generation(self) -> str:
        """Return ``'jointfm-1'``."""
        return JOINTFM_1_GENERATION

    def load(
        self, checkpoint: dict[str, Any], *, device: torch.device | None
    ) -> tuple[torch.nn.Module, JointFMConfig]:
        """Rebuild the network from the checkpoint."""
        from jointfm_inference.model.jointfm_1.loader import load_model_from_checkpoint

        return load_model_from_checkpoint(checkpoint, device=device)

    def mount(
        self,
        model: torch.nn.Module,
        model_cfg: Any,
        checkpoint: dict[str, Any],
        *,
        device: torch.device,
        autocast_dtype: torch.dtype | None,
    ) -> JointFM1InferenceBackend:
        """Mount one loaded JointFM 1 network; ``model_cfg`` is not read."""
        del model_cfg
        return JointFM1InferenceBackend(
            model, checkpoint, device=device, autocast_dtype=autocast_dtype
        )


JOINTFM_1_SERVING_BACKEND: Final = JointFM1ServingBackend()
"""The object the ``jointfm-1`` registry entry point names."""


__all__ = [
    "DECODING_STRATEGY",
    "JOINTFM_1_SERVING_BACKEND",
    "JointFM1InferenceBackend",
    "JointFM1ServingBackend",
]
