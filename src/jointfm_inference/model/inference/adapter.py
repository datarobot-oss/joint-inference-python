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

"""Predictor adapter over one Student-t mixture parameter mapping.

Provides Monte Carlo means and quantiles, sampling with rejection bounds,
log-densities, and exact conditioning on known columns.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch

from jointfm_inference.model.jointfm_1.head import StudentTHead

from ._bounds import (
    _SampleBound,
    _bound_error_fragment,
    _normalize_sample_bound,
    _sample_constraint_violations,
)
from .conditioning import MixtureCovariance, condition_student_t_mixture

DEFAULT_MEAN_SAMPLES = 500
DEFAULT_QUANTILE_SAMPLES = 2000
DEFAULT_SAMPLE_COUNT = 100
DEFAULT_MAX_REDRAWS = 100


class PredictorAdapter:
    """A head and the batched parameters ``(B, H, ...)`` of one forward."""

    def __init__(self, head: StudentTHead, params: Dict[str, torch.Tensor]) -> None:
        """Bind the head and its parameters.

        Raises:
            ValueError: when ``logits`` is not ``(B, H, K)`` or a parameter is
                not finite.
        """
        if params["logits"].ndim != 3:
            raise ValueError(
                f"logits must have shape (B, H, K), got {tuple(params['logits'].shape)}"
            )
        for name, value in params.items():
            if not torch.all(torch.isfinite(value)):
                raise ValueError(f"Parameter '{name}' contains NaNs or Infs")
        self.head = head
        self.params = params

    def _draw(self, n_samples: int, rng: Optional[torch.Generator]) -> torch.Tensor:
        """Draw ``(B, n_samples, H, D)`` samples from the head."""
        samples = self.head.sample(self.params, num_samples=n_samples, rng=rng)
        batch_times_samples, horizon_count, dim = samples.shape
        return samples.view(
            batch_times_samples // n_samples, n_samples, horizon_count, dim
        )

    def mean(
        self,
        *,
        n_samples: int = DEFAULT_MEAN_SAMPLES,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return the ``(B, H, D)`` Monte Carlo mean.

        The mean shares one sampler with ``quantiles`` and ``sample``;
        ``n_samples`` trades precision for latency.
        """
        return self.head.mean(self.params, num_samples=n_samples, rng=rng)

    def log_prob(self, values: torch.Tensor) -> torch.Tensor:
        """Return the ``(B, H)`` log-density of one ``(B, H, D)`` continuation."""
        if not torch.is_floating_point(values):
            raise TypeError("values must use a floating-point dtype")
        if not torch.all(torch.isfinite(values)):
            raise ValueError("values contains NaNs or Infs")
        reference = self.params["logits"]
        if values.ndim != 3 or tuple(values.shape[:2]) != tuple(reference.shape[:2]):
            raise ValueError(
                "log_prob values must have shape (B, H, D) matching the parameters' "
                f"{tuple(reference.shape[:2])}, got {tuple(values.shape)}"
            )
        return self.head.log_prob(
            self.params, values.to(device=reference.device, dtype=reference.dtype)
        )

    def sample(
        self,
        n_samples: int = DEFAULT_SAMPLE_COUNT,
        rng: Optional[torch.Generator] = None,
        min_value: _SampleBound | None = None,
        max_value: _SampleBound | None = None,
        max_retries: int = DEFAULT_MAX_REDRAWS,
    ) -> torch.Tensor:
        """Draw ``(B, n_samples, H, D)`` samples, redrawing out-of-bound ones.

        Args:
            n_samples: Draws per batch item and horizon.
            rng: Optional generator making the draws reproducible.
            min_value: Lower rejection bound; scalars apply to every column,
                tensors broadcast per column.
            max_value: Upper rejection bound, likewise.
            max_retries: Redraw rounds before a draw outside the bounds raises.

        Raises:
            RuntimeError: when some draw still violates the bounds after
                ``max_retries`` redraws.
        """
        samples = self._draw(n_samples, rng)
        lower = _normalize_sample_bound(min_value, samples=samples, field="min_value")
        upper = _normalize_sample_bound(max_value, samples=samples, field="max_value")
        if (
            lower is not None
            and upper is not None
            and bool((lower > upper).any().item())
        ):
            raise ValueError("min_value must not exceed max_value")

        is_bad = _sample_constraint_violations(
            samples, min_value=lower, max_value=upper
        )
        for _ in range(max_retries):
            if not is_bad.any():
                break
            replacements = self._draw(n_samples, rng)
            samples = torch.where(
                is_bad.unsqueeze(-1).expand_as(samples), replacements, samples
            )
            is_bad = _sample_constraint_violations(
                samples, min_value=lower, max_value=upper
            )
        if is_bad.any():
            error_msg = f"Failed to satisfy constraints after {max_retries} retries."
            if lower is not None and (samples < lower).any():
                error_msg += _bound_error_fragment("min_value", lower)
            if upper is not None and (samples > upper).any():
                error_msg += _bound_error_fragment("max_value", upper)
            if torch.isnan(samples).any():
                error_msg += " NaN violation."
            raise RuntimeError(error_msg)
        return samples

    def quantiles(
        self,
        q: Sequence[float],
        n_samples: int = DEFAULT_QUANTILE_SAMPLES,
        rng: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Return Monte Carlo marginal quantiles ``(len(q), B, H, D)``."""
        return self.head.quantiles(self.params, q, num_samples=n_samples, rng=rng)

    def condition(self, values: torch.Tensor, mask: torch.Tensor) -> "PredictorAdapter":
        """Return the exact conditional over the columns ``mask`` leaves unknown.

        ``values`` is ``(B, H, D)`` and ``mask`` a ``(D,)`` boolean mask of the
        known columns. The returned adapter carries the reduced-width
        conditional in the head's own parameter mapping (with a dense
        Cholesky factor), so its sampling and scoring apply unchanged.
        """
        means = self.params["means"]
        full_dim = means.shape[-1]
        if values.ndim != 3 or values.shape[-1] != full_dim:
            raise ValueError("values must have shape (B, H, D) with the full width D")
        if mask.dtype is not torch.bool:
            raise TypeError("mask must be a bool tensor")
        if mask.shape != (full_dim,):
            raise ValueError(f"mask must have shape ({full_dim},)")
        new_logits, new_means, new_chol, new_df = condition_student_t_mixture(
            self.params["logits"],
            means,
            MixtureCovariance.from_params(self.params),
            self.params["df"],
            values,
            mask.to(device=means.device, dtype=torch.bool),
        )
        return PredictorAdapter(
            self.head,
            {"logits": new_logits, "means": new_means, "chol": new_chol, "df": new_df},
        )


__all__ = ["PredictorAdapter"]
