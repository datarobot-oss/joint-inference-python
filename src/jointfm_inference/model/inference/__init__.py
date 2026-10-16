"""Inference over loaded JointFM checkpoints.

- ``session``: ``JointFMInferenceSession``, the entry point.
- ``contract``: the backend-neutral contract (``CanonicalInferenceRequest``,
  ``ServingBackend``, ``InferenceBackend``, ``InferenceDistributionResult``).
- ``backend_registry``: resolves a checkpoint generation marker to its
  registered ``ServingBackend``.
- ``jointfm_1_backend`` and ``jointfm_1_grid``: the ``jointfm-1`` serving
  backend, reached through the registry and not re-exported here, so importing
  this package builds no network.
- ``adapter``: ``PredictorAdapter`` over one forward's parameters.
- ``conditioning``, ``conditional_query``, ``interval_conditioning``: exact
  conditioning on equality and interval conditions.
- ``time_index`` and ``capacity_envelope``: the checkpoint metadata the health
  payload reports.
"""

from .adapter import PredictorAdapter
from .backend_registry import (
    INFERENCE_BACKEND_ENTRY_POINT_GROUP,
    UnregisteredGenerationError,
    registered_generations,
    resolve_serving_backend,
)
from .capacity_envelope import CapacityEnvelope
from .conditional_query import (
    ConditionalReadOut,
    ConditionRequest,
    condition_at_position,
)
from .conditioning import (
    MixtureCovariance,
    condition_student_t_mixture,
    conditioned_log_density,
)
from .contract import (
    INFERENCE_RETURN_MODES,
    CanonicalInferenceRequest,
    ConditionalDistributionResult,
    InferenceBackend,
    InferenceDistributionResult,
    InferenceRequestError,
    JointFM1DistributionResult,
    ServingBackend,
    normalize_head_name,
)
from .session import DEFAULT_INFERENCE_PRECISION, JointFMInferenceSession
from .time_index import DISCRETE_GRID_TIME_INDEX_ENCODING, TimeIndexSupport

__all__ = [
    "DEFAULT_INFERENCE_PRECISION",
    "DISCRETE_GRID_TIME_INDEX_ENCODING",
    "INFERENCE_BACKEND_ENTRY_POINT_GROUP",
    "INFERENCE_RETURN_MODES",
    "CanonicalInferenceRequest",
    "CapacityEnvelope",
    "ConditionRequest",
    "ConditionalDistributionResult",
    "ConditionalReadOut",
    "InferenceBackend",
    "InferenceDistributionResult",
    "InferenceRequestError",
    "JointFM1DistributionResult",
    "JointFMInferenceSession",
    "MixtureCovariance",
    "PredictorAdapter",
    "ServingBackend",
    "TimeIndexSupport",
    "UnregisteredGenerationError",
    "condition_at_position",
    "condition_student_t_mixture",
    "conditioned_log_density",
    "normalize_head_name",
    "registered_generations",
    "resolve_serving_backend",
]
