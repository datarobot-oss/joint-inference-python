"""Backend-neutral inference contract.

A :class:`ServingBackend` is what the inference backend registry
(``jointfm_inference.model.inference.backend_registry``) holds for one
checkpoint generation: it loads that generation's checkpoints and mounts a
loaded model. The mounted :class:`InferenceBackend` reports what the served
model can answer and runs one request, producing a distribution result with
the four return-mode operations ``mean``, ``samples``, ``quantiles``, and
``log_prob`` (:class:`InferenceDistributionResult`). Session and transport code
read only this contract, so neither names a generation.

- Transport concerns (field validation, version checks, JSON serialization)
  stay in ``jointfm_inference.model.service``.
- The served generation and checkpoint are fixed when the backend is mounted,
  never per request, and there is no fallback between backends.
- Inference is stateless: a backend encodes a request's complete row set once
  and produces every requested row in one model forward.

Result shapes (shared with ``jointfm_inference.model.service.serialization``):
``B`` is the request batch axis (``1`` for the HTTP transport), ``H`` the
number of requested rows, ``S`` the sample count, ``Q`` the quantile count,
and ``D`` the column axis, one position per request column in request order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol, runtime_checkable

import torch
from torch import Tensor

from jointfm_inference.data.types import CanonicalSample

from ._bounds import _SampleBound
from .adapter import PredictorAdapter
from .conditional_query import (
    DEFAULT_INTERVAL_POINTS,
    ConditionRequest,
    ConditionalReadOut,
    condition_at_position,
)
from .time_index import TimeIndexSupport

INFERENCE_RETURN_MODES: Final[frozenset[str]] = frozenset(
    {"mean", "samples", "quantiles", "log_prob"}
)
"""The return modes named by the :class:`InferenceDistributionResult` operations."""
HEAD_CLASS_SUFFIX: Final[str] = "Head"


def normalize_head_name(head_class_name: str) -> str:
    """Return the head name a service reports for a head class name.

    Drops a trailing ``Head`` and every underscore and lower-cases the rest
    (``StudentTHead`` becomes ``studentt``).
    """
    return head_class_name.removesuffix(HEAD_CLASS_SUFFIX).replace("_", "").lower()


class InferenceRequestError(ValueError):
    """A request the mounted backend refuses because of what the request asks.

    Raised by :meth:`InferenceBackend.execute` for a request the served model
    cannot represent (a history longer than its time capacity, query times off
    its time grid), as opposed to a failure of the backend itself. ``field``
    names the request field responsible when one is, so a transport can point
    the caller at it.
    """

    def __init__(self, message: str, *, field: str | None = None) -> None:
        """Record the refusal message and the responsible request field."""
        super().__init__(message)
        self.field = field


@dataclass(frozen=True)
class CanonicalInferenceRequest:
    """One stateless inference request: history rows, then the rows to forecast."""

    sample: CanonicalSample

    def __post_init__(self) -> None:
        """Fail fast when the sample has no row to forecast."""
        if not isinstance(self.sample, CanonicalSample):
            raise TypeError(
                f"sample must be a CanonicalSample, got {type(self.sample).__name__}"
            )
        if not bool(self.sample.query_row_mask.any()):
            raise ValueError("request sample must carry at least one query row")

    @property
    def query_row_mask(self) -> Tensor:
        """``(rows,)`` bool mask of the rows to forecast."""
        return self.sample.query_row_mask

    @property
    def column_count(self) -> int:
        """Number of canonical request columns (the aligned ``D`` prefix)."""
        return int(self.sample.values.shape[1])


@runtime_checkable
class InferenceDistributionResult(Protocol):
    """Distribution-result operations every serving backend provides.

    One result is bound to one executed request; the operation names mirror
    the public return-mode vocabulary. All tensors follow the result-shape
    contract in the module docstring.
    """

    def mean(
        self,
        *,
        n_samples: int | None = None,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        """Return the ``(B, H, D)`` predictive mean.

        ``mean`` may be a Monte Carlo estimate; ``n_samples`` then controls
        the precision/latency trade-off (``None`` uses the backend default)
        and ``rng`` makes it reproducible.
        """
        ...

    def samples(
        self,
        *,
        n_samples: int,
        rng: torch.Generator | None = None,
        min_value: _SampleBound | None = None,
        max_value: _SampleBound | None = None,
    ) -> Tensor:
        """Draw ``(B, S, H, D)`` predictive samples.

        ``min_value`` / ``max_value`` are optional rejection bounds (scalars
        apply globally; tensors may provide broadcastable per-column bounds).
        A draw outside them, or a non-finite draw, is redrawn.
        """
        ...

    def quantiles(
        self,
        q: Sequence[float],
        *,
        n_samples: int,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        """Return ``(Q, B, H, D)`` marginal quantiles for probabilities ``q``."""
        ...

    def log_prob(self, values: Tensor) -> Tensor:
        """Score one observed continuation ``values`` of shape ``(B, H, D)``.

        Returns per-row log-probability scores of shape ``(B, H)``.
        """
        ...


@runtime_checkable
class ConditionalDistributionResult(InferenceDistributionResult, Protocol):
    """A distribution result that can also answer a conditioning request.

    Conditioning is a strictly narrower capability than the four return-mode
    operations: it needs the head's emitted mixture parameters, not just draws
    from them. A backend whose result does not satisfy this protocol serves no
    ``condition`` query, and the service refuses the mode before execution
    rather than discovering the gap afterwards.
    """

    def condition(
        self,
        request: ConditionRequest,
        *,
        n_samples: int,
        interval_points: int = DEFAULT_INTERVAL_POINTS,
        rng: torch.Generator | None = None,
        min_value: _SampleBound | None = None,
        max_value: _SampleBound | None = None,
    ) -> ConditionalReadOut:
        """Answer one conditioning request at one emission position.

        The answer covers the request's columns and no others: its read-out
        set, every tensor it returns, and the per-column bounds it accepts are
        all ``column_count`` wide.
        """
        ...


@runtime_checkable
class InferenceBackend(Protocol):
    """One serving backend mounted on one loaded model.

    Built once per mounted checkpoint by :meth:`ServingBackend.mount`, with
    its generation, head, device, and autocast dtype fixed; nothing about it
    changes per request. The properties report what the mounted model
    can serve, so a transport advertises and gates requests from them without
    knowing the generation. ``execute`` encodes the request's complete row
    set in one stateless model forward and returns the bound distribution
    result.
    """

    @property
    def generation(self) -> str:
        """The checkpoint generation marker this backend serves."""
        ...

    @property
    def head_name(self) -> str:
        """The mounted head's kind as the service reports it.

        Built with :func:`normalize_head_name`, so one head kind has one name
        whatever generation serves it.
        """
        ...

    @property
    def time_index_support(self) -> TimeIndexSupport:
        """The time-index contract the mounted model reads request times under."""
        ...

    @property
    def decoding_strategy(self) -> str:
        """How the mounted model decodes forecast horizons, as ``/healthz`` reports it."""
        ...

    @property
    def metadata_traits(self) -> tuple[str, ...]:
        """The metadata traits the mounted model reads as expert hints; may be empty."""
        ...

    @property
    def supported_return_modes(self) -> frozenset[str]:
        """The return modes the mounted head serves; a subset of :data:`INFERENCE_RETURN_MODES`."""
        ...

    @property
    def supports_conditioning(self) -> bool:
        """Whether every result of :meth:`execute` answers conditioning requests.

        ``True`` promises a :class:`ConditionalDistributionResult`.
        """
        ...

    def execute(
        self,
        request: CanonicalInferenceRequest,
    ) -> InferenceDistributionResult:
        """Run one stateless full-request model execution.

        Raises:
            InferenceRequestError: when the request asks for something the
                mounted model cannot represent.
        """
        ...


@runtime_checkable
class ServingBackend(Protocol):
    """The registry entry for one checkpoint generation.

    Stateless: it loads a checkpoint of its generation into a model and mounts
    a loaded model as an :class:`InferenceBackend`. A generation's backend is
    found through ``jointfm_inference.model.inference.backend_registry``, never imported
    by session or transport code.
    """

    @property
    def generation(self) -> str:
        """The checkpoint generation marker this backend is registered for."""
        ...

    def load(
        self,
        checkpoint: dict[str, Any],
        *,
        device: torch.device | None,
    ) -> tuple[torch.nn.Module, Any]:
        """Rebuild the model behind one loaded checkpoint mapping.

        Returns ``(model, model_cfg)``: the model in evaluation mode on
        ``device`` (left where it is for ``None``) and the configuration
        object this generation's backend reads.
        """
        ...

    def mount(
        self,
        model: torch.nn.Module,
        model_cfg: Any,
        checkpoint: dict[str, Any],
        *,
        device: torch.device,
        autocast_dtype: torch.dtype | None,
    ) -> InferenceBackend:
        """Mount one loaded model for serving; ``checkpoint`` supplies its metadata.

        ``device`` is where the model's parameters live and ``autocast_dtype``
        the autocast dtype of every forward (``None`` for full precision).

        Raises:
            TypeError: when the model, its head, or ``model_cfg`` is not of
                this generation.
        """
        ...


class JointFM1DistributionResult:
    """The distribution-result operations of a JointFM 1 forward.

    Wraps the :class:`PredictorAdapter` of one forward and the request's
    column count. Every operation answers over the request's columns, and
    ``condition`` narrows to them before any algebra, so a conditional's
    read-out set, its tensors, and the caller's bounds all have the request's
    width.
    """

    def __init__(
        self,
        adapter: PredictorAdapter,
        *,
        active_column_count: int,
    ) -> None:
        """Bind one predictor adapter and the request's column count."""
        if not isinstance(adapter, PredictorAdapter):
            raise TypeError(
                f"adapter must be a PredictorAdapter, got {type(adapter).__name__}"
            )
        if active_column_count <= 0:
            raise ValueError("active_column_count must be a positive int")
        self._adapter = adapter
        self._active_column_count = active_column_count

    def mean(
        self,
        *,
        n_samples: int | None = None,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        """Return the ``(B, H, D)`` Monte Carlo predictive mean."""
        if n_samples is None:
            return self._adapter.mean(rng=rng)
        return self._adapter.mean(n_samples=n_samples, rng=rng)

    def samples(
        self,
        *,
        n_samples: int,
        rng: torch.Generator | None = None,
        min_value: _SampleBound | None = None,
        max_value: _SampleBound | None = None,
    ) -> Tensor:
        """Draw ``(B, S, H, D)`` samples over the request's columns."""
        return self._adapter.sample(
            n_samples=n_samples,
            rng=rng,
            min_value=min_value,
            max_value=max_value,
        )

    def quantiles(
        self,
        q: Sequence[float],
        *,
        n_samples: int,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        """Return ``(Q, B, H, D)`` sampling-based marginal quantiles."""
        return self._adapter.quantiles(q=q, n_samples=n_samples, rng=rng)

    def log_prob(self, values: Tensor) -> Tensor:
        """Score ``(B, H, D)`` observed values, returning ``(B, H)``."""
        return self._adapter.log_prob(values)

    def condition(
        self,
        request: ConditionRequest,
        *,
        n_samples: int,
        interval_points: int = DEFAULT_INTERVAL_POINTS,
        rng: torch.Generator | None = None,
        min_value: _SampleBound | None = None,
        max_value: _SampleBound | None = None,
    ) -> ConditionalReadOut:
        """Answer one conditioning request over the executed request's columns."""
        return condition_at_position(
            self._adapter,
            request,
            n_samples=n_samples,
            interval_points=interval_points,
            rng=rng,
            min_value=min_value,
            max_value=max_value,
            column_count=self._active_column_count,
        )


__all__ = [
    "INFERENCE_RETURN_MODES",
    "CanonicalInferenceRequest",
    "ConditionalDistributionResult",
    "InferenceBackend",
    "InferenceDistributionResult",
    "InferenceRequestError",
    "JointFM1DistributionResult",
    "ServingBackend",
    "normalize_head_name",
]
