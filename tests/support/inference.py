"""Student-t head and adapter helpers for test fixtures."""

import math

import torch

from jointfm_inference.model.inference import PredictorAdapter
from jointfm_inference.model.jointfm_1.head import StudentTHead

TINY_HEAD_WIDTH = 4
HEAD_SEED = 0


def student_t_head(
    *,
    D: int,
    K: int,
    d_model: int = TINY_HEAD_WIDTH,
    covariance_rank: int = 1,
    min_df: float = 2.0,
    min_scale: float = 1e-4,
) -> StudentTHead:
    """Return a small Student-t head over ``D`` columns and ``K`` components.

    The weights are drawn from a fixed seed without touching the global random
    state, so a test that reads the head's emitted parameters is reproducible
    and a test that draws from the global generator is not perturbed.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(HEAD_SEED)
        return StudentTHead(
            d_model=d_model,
            D=D,
            K=K,
            trunk_layers=1,
            trunk_hidden=d_model,
            covariance_rank=covariance_rank,
            min_df=min_df,
            min_scale=min_scale,
        )


def create_student_t_adapter(params: dict[str, torch.Tensor]) -> PredictorAdapter:
    """Return an adapter over a Student-t parameter mapping ``(B, H, K, ...)``."""
    _batch, _positions, mixtures = params["logits"].shape
    dimension = params["means"].shape[-1]
    return PredictorAdapter(student_t_head(D=dimension, K=mixtures), params)


def student_t_pair_mixture(dtype=torch.float64):
    """Build a two-component Student-t mixture over one emission position.

    The components disagree on everything conditioning reads: the observed
    column's scale (so the posterior weights move), the sign of the
    cross-column covariance (so the two location shifts point opposite ways),
    and therefore the observed block's Mahalanobis distance (so the two scale
    inflations differ). Both carry the same degrees of freedom, which keeps the
    conditional's raised ``df`` a single number to reason about, and that value
    stays above four so the retained draws of a Monte Carlo oracle have a
    convergent sample variance.

    Returns:
        The parameter mapping of a ``(B=1, H=1, K=2, D=2)`` mixture:
        ``logits``, ``means``, ``chol``, and ``df``.
    """
    covariance = torch.stack(
        [
            torch.tensor([[1.0, 0.8], [0.8, 1.0]], dtype=dtype),
            torch.tensor([[4.0, -1.6], [-1.6, 1.6]], dtype=dtype),
        ]
    )
    return {
        "logits": torch.zeros(1, 1, 2, dtype=dtype),
        "means": torch.tensor([[[[0.0, 0.0], [0.0, 1.0]]]], dtype=dtype),
        "chol": torch.linalg.cholesky(covariance).reshape(1, 1, 2, 2, 2),
        "df": torch.full((1, 1, 2), 6.0, dtype=dtype),
    }


def student_t_log_density(
    value: float | torch.Tensor,
    location: float | torch.Tensor,
    scale: float | torch.Tensor,
    df: float | torch.Tensor,
) -> torch.Tensor:
    """Evaluate a univariate Student-t log density from its defining formula.

    ``scale`` is the squared scale, the one-column counterpart of a scale
    matrix. Plain numbers are read in float64 so a closed-form oracle keeps
    its precision.
    """
    value, location, scale, df = (
        term
        if isinstance(term, torch.Tensor)
        else torch.tensor(term, dtype=torch.float64)
        for term in (value, location, scale, df)
    )
    return (
        torch.lgamma((df + 1.0) / 2.0)
        - torch.lgamma(df / 2.0)
        - 0.5 * torch.log(df * math.pi * scale)
        - 0.5 * (df + 1.0) * torch.log1p((value - location) ** 2 / (scale * df))
    )
