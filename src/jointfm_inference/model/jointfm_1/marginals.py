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

"""Exact means and marginal quantiles of a Student-t mixture.

Both summaries are deterministic: the same parameters give the same answer for
every seed, because neither one draws. The mean is the mixture-weighted sum of
the component locations. A quantile is the root of one column's marginal
distribution function, found by bisection to a stated tolerance.

A column's marginal of a multivariate Student-t component is the univariate
Student-t with the component's location in that column, the square root of the
scale matrix's diagonal entry as its scale, and the component's degrees of
freedom, so a column's marginal of the mixture is the mixture of those
univariate laws under the same weights.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import stdtr, stdtrit
from torch import Tensor

# A bisected quantile lies within this fraction of its column's smallest
# component scale of the exact root, below what a float32 answer resolves.
QUANTILE_TOLERANCE = 1e-6
# Halvings after which a float64 bracket stops shrinking: a bracket wider than
# 2**64 tolerances spans values whose spacing in float64 exceeds the tolerance.
MAX_BISECTION_STEPS = 64
# A Student-t component has a mean only above one degree of freedom.
MIN_DF_FOR_MEAN = 1.0


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """Return the dtype a mean is summed in: float64 stays, anything else is float32."""
    return torch.promote_types(dtype, torch.float32)


def component_marginal_scales(params: Dict[str, Tensor]) -> Tensor:
    """Return the ``(B, H, K, D)`` scale of every component in every column.

    The scale is the square root of the scale matrix's diagonal: ``lowrank_diag``
    plus the squared factor rows for emitted parameters, the squared row norms of
    ``chol`` for a conditioned mixture, which carries a dense Cholesky factor.
    The scales are float64, so a finite scale whose square overflows the
    parameters' own dtype stays finite.

    Raises:
        ValueError: when a scale is not finite and positive.
    """
    if "chol" in params:
        variance = params["chol"].double().pow(2).sum(dim=-1)
    else:
        variance = params["lowrank_diag"].double() + params["lowrank_U"].double().pow(
            2
        ).sum(dim=-1)
    scales = variance.sqrt()
    if not bool(torch.isfinite(scales).all()) or not bool((scales > 0).all()):
        raise ValueError("every component scale must be finite and positive")
    return scales


def mixture_mean(params: Dict[str, Tensor]) -> Tensor:
    """Return the ``(B, H, D)`` mean of the mixture: weighted component locations.

    ``logits`` need not be normalized; the weights are their softmax, so a
    pooled conditional mixture is accepted too.

    Raises:
        ValueError: when some component has at most one degree of freedom, so
            the mixture has no mean.
    """
    df = params["df"]
    if not bool((df > MIN_DF_FOR_MEAN).all()):
        raise ValueError(
            "a Student-t mixture has a mean only when every component has more "
            f"than {MIN_DF_FOR_MEAN:g} degree of freedom; the smallest is "
            f"{float(df.min()):g}"
        )
    means = params["means"]
    compute_dtype = _compute_dtype(means.dtype)
    weights = F.softmax(params["logits"].to(dtype=compute_dtype), dim=-1)
    mean = (weights.unsqueeze(-1) * means.to(dtype=compute_dtype)).sum(dim=-2)
    return mean.to(dtype=means.dtype)


def _require_probabilities(q: Sequence[float]) -> list[float]:
    """Return ``q`` as floats, refusing an empty list or a level outside ``(0, 1)``."""
    levels = [float(level) for level in q]
    if not levels:
        raise ValueError("q must contain at least one probability")
    for level in levels:
        if not 0.0 < level < 1.0:
            raise ValueError(f"every quantile level must lie in (0, 1), got {level}")
    return levels


def _host(value: Tensor, cells: int) -> Any:
    """Return a ``(B, H, ...)`` tensor as a float64 ``(B * H, ...)`` NumPy array."""
    flat = value.detach().to(device="cpu", dtype=torch.float64)
    return flat.reshape(cells, *flat.shape[2:]).numpy()


@dataclass(frozen=True)
class _MarginalMixtures:
    """Every column's marginal mixture at ``N`` emission positions, on the host.

    Attributes:
        weights: ``(N, K)`` component weights.
        df: ``(N, K)`` degrees of freedom, shared by a component's columns.
        locations: ``(N, K, D)`` component locations.
        scales: ``(N, K, D)`` component scales.
    """

    weights: Any
    df: Any
    locations: Any
    scales: Any

    def cdf(self, points: Any, cells: slice) -> Any:
        """Return the ``(n, Q, D)`` distribution function at ``(n, Q, D)`` points.

        ``points`` covers the positions ``cells`` selects.
        """
        locations = self.locations[cells, None]
        scales = self.scales[cells, None]
        standardized = (points[:, :, None, :] - locations) / scales
        probability = stdtr(self.df[cells, None, :, None], standardized)
        return (self.weights[cells, None, :, None] * probability).sum(axis=2)


def mixture_quantiles(params: Dict[str, Tensor], q: Sequence[float]) -> Tensor:
    """Return the ``(Q, B, H, D)`` marginal quantiles of the mixture at levels ``q``.

    Each quantile is the root of its column's marginal mixture distribution
    function. Every component's own quantile at a level bounds the mixture's:
    at the smallest of them every component's distribution function is at most
    the level, at the largest at least, so the bracket between the two holds the
    root. Bisection halves every bracket for as many steps as the widest one
    needs to shrink below :data:`QUANTILE_TOLERANCE` times its column's smallest
    component scale, capped at :data:`MAX_BISECTION_STEPS`, and returns the
    midpoint. A one-component mixture's bracket is already a point, the
    component's quantile.

    The Student-t distribution function and its inverse come from SciPy,
    because torch implements neither, so the work runs in float64 on the host:
    one transfer of the parameters, and one of the answer back to their device
    and dtype. Each step evaluates ``Q * B * H * K * D`` distribution functions,
    split by emission position over as many threads as torch's intra-op
    setting allows; SciPy releases the interpreter lock while it evaluates, and
    every position's answer is the same however the positions are split.

    Raises:
        ValueError: when ``q`` is empty or holds a level outside ``(0, 1)``, or
            when a component scale is not finite and positive.
    """
    levels = _require_probabilities(q)
    means = params["means"]
    batch, horizons, _components, columns = means.shape
    cells = batch * horizons
    mixtures = _MarginalMixtures(
        weights=_host(F.softmax(params["logits"].double(), dim=-1), cells),
        df=_host(params["df"], cells),
        locations=_host(means, cells),
        scales=_host(component_marginal_scales(params), cells),
    )

    level_axis = np.asarray(levels, dtype=np.float64).reshape(1, -1, 1)
    # (N, Q, K): standardized component quantiles; df is shared by columns.
    standardized = stdtrit(mixtures.df[:, None, :], level_axis)
    component_quantiles = (
        mixtures.locations[:, None] + mixtures.scales[:, None] * standardized[..., None]
    )
    lower = component_quantiles.min(axis=2)
    upper = component_quantiles.max(axis=2)

    tolerance = QUANTILE_TOLERANCE * mixtures.scales.min(axis=1)
    widest = float(((upper - lower) / tolerance[:, None, :]).max())
    steps = (
        0 if widest <= 1.0 else min(math.ceil(math.log2(widest)), MAX_BISECTION_STEPS)
    )
    chunks = [
        slice(int(part[0]), int(part[-1]) + 1)
        for part in np.array_split(
            np.arange(cells), min(torch.get_num_threads(), cells)
        )
    ]
    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        for _ in range(steps):
            midpoint = 0.5 * (lower + upper)
            parts = pool.map(
                mixtures.cdf, [midpoint[chunk] for chunk in chunks], chunks
            )
            below = np.concatenate(list(parts), axis=0) < level_axis
            lower = np.where(below, midpoint, lower)
            upper = np.where(below, upper, midpoint)
    quantiles = torch.from_numpy(0.5 * (lower + upper))
    quantiles = quantiles.reshape(batch, horizons, len(levels), columns)
    return quantiles.permute(2, 0, 1, 3).to(device=means.device, dtype=means.dtype)


__all__ = [
    "MAX_BISECTION_STEPS",
    "QUANTILE_TOLERANCE",
    "component_marginal_scales",
    "mixture_mean",
    "mixture_quantiles",
]
