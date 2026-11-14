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

Under a per-column tail map (:mod:`jointfm_inference.model.jointfm_1.tail_map`)
the mixture is the latent law and every column's data value is one increasing
function of its latent coordinate. A quantile is then the root of the latent
distribution function at the value's latent image, bisected in data units.
The mean is no longer the weighted locations: the map bends each component's
law, and the anchors it standardizes by are not the component's center, so
the mean is a one-dimensional integral per component and column, evaluated by
quadrature to a stated tolerance (:func:`mapped_mixture_mean`).
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import stdtr, stdtrit
from torch import Tensor

from jointfm_inference.model.jointfm_1.tail_map import (
    ANCHOR_LOCATION_KEY,
    ANCHOR_SCALE_KEY,
    ColumnTailMap,
    map_standardized,
)

# A bisected quantile lies within this fraction of its column's smallest
# component scale of the exact root, below what a float32 answer resolves.
QUANTILE_TOLERANCE = 1e-6
# Halvings after which a float64 bracket stops shrinking: a bracket wider than
# 2**64 tolerances spans values whose spacing in float64 exceeds the tolerance.
MAX_BISECTION_STEPS = 64
# A Student-t component has a mean only above one degree of freedom.
MIN_DF_FOR_MEAN = 1.0
# Step of the double-exponential quadrature behind a mapped mean, in the
# transformed variable. Against the same rule at a quarter of the step it
# differs by under 1e-7 anchor scales for any single component, however far
# from the anchors, so a mixture's mean stays within MAPPED_MEAN_TOLERANCE
# whatever its weights; the tests hold it against an adaptive float64
# reference.
MAPPED_MEAN_STEP = 1.0 / 16.0
# A mapped mean lies within this fraction of its column's anchor scale of the
# exact integral, below what a float32 answer resolves.
MAPPED_MEAN_TOLERANCE = 1e-7
# Half-line nodes reach from exp(-HALF_LINE_LOG_NEAR) to HALF_LINE_FAR past
# their endpoint, in units of the component's own standardized coordinate on
# an asinh scale. A component's density falls faster than exp(-|t|) there
# (the configured bounds keep every tail parameter below half the degrees of
# freedom), so the integral beyond is below float64 resolution; nearer than
# the first node, the map's closest complex singularity (at distance 1e-4 or
# more for any component weight above 1e-8) is still resolved.
HALF_LINE_LOG_NEAR = 25.0
HALF_LINE_FAR = 60.0
# Rate of the half-line map exp(HALF_LINE_RATE * sinh(t)). Below the usual
# pi / 2 it crowds the nodes less where the Student-t density's complex poles
# sit, a quarter turn off the real axis, at the price of a few more nodes.
HALF_LINE_RATE = 1.0
# The finite pieces' tanh-sinh nodes stop where their distance to an end
# underflows float64.
FINITE_PIECE_EXTENT = 3.2
# Half-width of the finite piece that holds a component's density peak when
# the map's kink lies farther out, so the peak is not left to the sparse
# middle of a long piece.
PEAK_EXTENT = 2.0
# Elements (positions times components times columns) integrated per chunk,
# which bounds the quadrature's working set.
MEAN_CHUNK_ELEMENTS = 1 << 15


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


def column_anchors(params: Dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    """Return the tail map's anchors: each column's mixture center and total spread.

    A per-column tail map serves every component of a column, so it
    standardizes by anchors of the column, ``(B, H, D)`` each:

        a_d = sum_k pi_k mu_kd,
        s_d = sqrt(sum_k pi_k (sigma_kd^2 + (mu_kd - a_d)^2)),

    with ``pi`` the softmax of ``logits`` and ``sigma_kd`` the component
    scales of :func:`component_marginal_scales`. They exist for every
    ``nu``, equal one component's location and scale when the mixture has
    one, and commute with a positive per-column rescaling. They are computed
    in float64 and returned in the dtype of ``means``.

    Raises:
        ValueError: when a component scale is not finite and positive.
    """
    scales = component_marginal_scales(params)
    means = params["means"].double()
    weights = F.softmax(params["logits"].double(), dim=-1).unsqueeze(-1)
    location = (weights * means).sum(dim=-2)
    spread = scales.pow(2) + (means - location.unsqueeze(-2)).pow(2)
    scale = (weights * spread).sum(dim=-2).sqrt()
    dtype = params["means"].dtype
    return location.to(dtype=dtype), scale.to(dtype=dtype)


def attach_column_anchors(params: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Return ``params`` with its emitted mixture's :func:`column_anchors` attached.

    The anchors go under ``ANCHOR_LOCATION_KEY`` and ``ANCHOR_SCALE_KEY``.
    They belong to the emitted joint mixture, so they are attached once, before
    any conditioning, and a conditioned mapping keeps them.
    """
    location, scale = column_anchors(params)
    return {**params, ANCHOR_LOCATION_KEY: location, ANCHOR_SCALE_KEY: scale}


def mixture_mean(params: Dict[str, Tensor]) -> Tensor:
    """Return the ``(B, H, D)`` mean of the mixture.

    Without a tail map it is the weighted component locations. With one
    (``params`` carries :data:`~jointfm_inference.model.jointfm_1.tail_map.TAIL_MAP_KEYS`)
    it is :func:`mapped_mixture_mean`. ``logits`` need not be normalized; the
    weights are their softmax, so a pooled conditional mixture is accepted
    too.

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
    tail_map = ColumnTailMap.from_params(params)
    if tail_map is not None:
        return mapped_mixture_mean(params, tail_map)
    means = params["means"]
    compute_dtype = _compute_dtype(means.dtype)
    weights = F.softmax(params["logits"].to(dtype=compute_dtype), dim=-1)
    mean = (weights.unsqueeze(-1) * means.to(dtype=compute_dtype)).sum(dim=-2)
    return mean.to(dtype=means.dtype)


@dataclass(frozen=True)
class _QuadratureRule:
    """Double-exponential nodes behind a mapped mean, on a standard frame.

    Attributes:
        half_line_distance: ``(n_h,)`` distances of a half-line's nodes from
            its endpoint, ``exp(HALF_LINE_RATE * sinh(t))`` (the exp-sinh
            rule).
        half_line_weight: ``(n_h,)`` their weights.
        finite_position: ``(n_f,)`` nodes of a finite piece on ``(-1, 1)``,
            ``tanh(pi/2 * sinh(t))`` (the tanh-sinh rule).
        finite_weight: ``(n_f,)`` their weights on ``(-1, 1)``.
    """

    half_line_distance: Tensor
    half_line_weight: Tensor
    finite_position: Tensor
    finite_weight: Tensor


def _quadrature_rule(device: torch.device) -> _QuadratureRule:
    """Return the double-exponential rule at :data:`MAPPED_MEAN_STEP` in float64.

    Both rules space their nodes evenly in ``t`` and cluster them doubly
    exponentially at a piece's ends, where the kink of the two-sided map and
    the component's density peak sit; on a half-line they also spread out
    doubly exponentially into the tail (Takahasi & Mori, 1974, "Double
    exponential formulas for numerical integration").
    """
    step = MAPPED_MEAN_STEP
    half_pi = 0.5 * math.pi
    near = math.asinh(-HALF_LINE_LOG_NEAR / HALF_LINE_RATE)
    far = math.asinh(math.log(HALF_LINE_FAR) / HALF_LINE_RATE)
    half_t = torch.arange(near, far + step, step, dtype=torch.float64, device=device)
    distance = torch.exp(HALF_LINE_RATE * torch.sinh(half_t))
    finite_t = torch.arange(
        -FINITE_PIECE_EXTENT,
        FINITE_PIECE_EXTENT + step,
        step,
        dtype=torch.float64,
        device=device,
    )
    inner = half_pi * torch.sinh(finite_t)
    return _QuadratureRule(
        half_line_distance=distance,
        half_line_weight=step * HALF_LINE_RATE * torch.cosh(half_t) * distance,
        finite_position=torch.tanh(inner),
        finite_weight=step * half_pi * torch.cosh(finite_t) / torch.cosh(inner).pow(2),
    )


def _mapped_coordinate_integrand(
    angle: Tensor,
    *,
    offset: Tensor,
    ratio: Tensor,
    df: Tensor,
    tail: Tensor,
) -> Tensor:
    """Return ``g(delta, c + r * sinh(t)) * f_nu(sinh(t)) * cosh(t)`` at nodes ``t``.

    ``f_nu`` is the standard Student-t density, so integrating over ``t``
    gives ``E[g(delta, c + r * U)]`` for ``U ~ t_nu``. The per-element
    arguments are ``(n, 1)`` and broadcast against the ``(n, nodes)`` nodes.
    """
    standard = torch.sinh(angle)
    mapped = map_standardized(offset + ratio * standard, tail)
    log_norm = (
        torch.lgamma((df + 1.0) / 2.0)
        - torch.lgamma(df / 2.0)
        - 0.5 * torch.log(df * math.pi)
    )
    magnitude = angle.abs()
    log_cosh = magnitude + torch.log1p(torch.exp(-2.0 * magnitude)) - math.log(2.0)
    log_density = (
        log_norm - 0.5 * (df + 1.0) * torch.log1p(standard.pow(2) / df) + log_cosh
    )
    return mapped * torch.exp(log_density)


def _expected_mapped_coordinate(
    offset: Tensor, ratio: Tensor, df: Tensor, upper: Tensor, lower: Tensor
) -> Tensor:
    """Return ``E[g(c + r * U)]`` per element for ``U ~ t_nu`` and the two-sided map ``g``.

    Every argument is a float64 vector over the same elements. With
    ``U = sinh(t)`` the component's density is a peak of unit width at
    ``t = 0`` whatever ``nu`` is, and the map switches from its lower to its
    upper branch at ``t0 = asinh(-c / r)``. Splitting at ``0`` and ``t0``
    leaves no kink inside a piece: two half-lines, each from one of those
    points outward, which the exp-sinh rule integrates, and the stretch
    between them, which the tanh-sinh rule integrates in two pieces split
    :data:`PEAK_EXTENT` from the peak toward the kink, so the peak stays near
    a piece's end, where the rule's nodes are dense.
    """
    rule = _quadrature_rule(offset.device)
    expected = torch.empty_like(offset)
    for start in range(0, offset.numel(), MEAN_CHUNK_ELEMENTS):
        part = slice(start, start + MEAN_CHUNK_ELEMENTS)
        c, r, nu = offset[part, None], ratio[part, None], df[part, None]
        upper_tail, lower_tail = upper[part, None], lower[part, None]
        kink = torch.asinh(-c / r)
        peak = torch.zeros_like(kink)
        arguments = {"offset": c, "ratio": r, "df": nu}
        left = _mapped_coordinate_integrand(
            torch.minimum(kink, peak) - rule.half_line_distance,
            tail=lower_tail,
            **arguments,
        )
        right = _mapped_coordinate_integrand(
            torch.maximum(kink, peak) + rule.half_line_distance,
            tail=upper_tail,
            **arguments,
        )
        # Between the peak and the kink the map keeps one branch: the upper
        # one when the kink sits below the peak, the lower one otherwise.
        between_tail = torch.where(kink < 0, upper_tail, lower_tail)
        split = kink.clamp(min=-PEAK_EXTENT, max=PEAK_EXTENT)
        total = (left + right) @ rule.half_line_weight
        for first, second in ((peak, split), (split, kink)):
            center, half_width = 0.5 * (first + second), 0.5 * (second - first).abs()
            finite = _mapped_coordinate_integrand(
                center + half_width * rule.finite_position,
                tail=between_tail,
                **arguments,
            )
            total = total + (finite * half_width) @ rule.finite_weight
        expected[part] = total
    return expected


def mapped_mixture_mean(params: Dict[str, Tensor], tail_map: ColumnTailMap) -> Tensor:
    """Return the ``(B, H, D)`` mean of a Student-t mixture mapped per column.

    Column ``d`` is ``a + s * g(delta, (X - a) / s)`` for its latent
    coordinate ``X``, which under component ``k`` is ``mu_kd + sigma_kd * U``
    with ``U ~ t_nu_k``, so its mean is

        a + s * sum_k pi_k E[g(delta, (mu_kd - a) / s + (sigma_kd / s) * U)],

    one integral per component and column
    (:func:`_expected_mapped_coordinate`), evaluated in float64 to within
    :data:`MAPPED_MEAN_TOLERANCE` times the anchor scale ``s``. The integral
    exists while both tail parameters stay below the component's degrees of
    freedom, which a checkpoint's bounds guarantee.

    Raises:
        ValueError: when a component's degrees of freedom do not exceed a
            column's larger tail parameter, so that column has no mean, or
            when a component scale is not finite and positive.
    """
    means = params["means"]
    df = params["df"].double()
    heaviest = torch.maximum(tail_map.upper, tail_map.lower).double().unsqueeze(-2)
    if not bool((df.unsqueeze(-1) > heaviest).all()):
        raise ValueError(
            "a mapped Student-t mixture has a mean only when every component's "
            "degrees of freedom exceed both tail parameters of every column"
        )
    location = tail_map.anchor_location.double().unsqueeze(-2)
    anchor_scale = tail_map.anchor_scale.double().unsqueeze(-2)
    offset = (means.double() - location) / anchor_scale
    ratio = component_marginal_scales(params) / anchor_scale
    shape = offset.shape
    expected = _expected_mapped_coordinate(
        offset.reshape(-1),
        ratio.reshape(-1),
        df.unsqueeze(-1).expand(shape).reshape(-1),
        tail_map.upper.double().unsqueeze(-2).expand(shape).reshape(-1),
        tail_map.lower.double().unsqueeze(-2).expand(shape).reshape(-1),
    ).reshape(shape)
    weights = F.softmax(params["logits"].double(), dim=-1).unsqueeze(-1)
    mean = location.squeeze(-2) + anchor_scale.squeeze(-2) * (weights * expected).sum(
        dim=-2
    )
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
        tail_map: The per-column tail map over ``(N, 1, D)`` in float64 on
            the host, or ``None`` when the mixture is the data law itself.
    """

    weights: Any
    df: Any
    locations: Any
    scales: Any
    tail_map: Optional[ColumnTailMap]

    def latent(self, points: Any, cells: slice) -> Any:
        """Return the latent images of ``(n, Q, D)`` data points of the positions ``cells`` selects."""
        if self.tail_map is None:
            return points
        rows = ColumnTailMap(
            upper=self.tail_map.upper[cells],
            lower=self.tail_map.lower[cells],
            anchor_location=self.tail_map.anchor_location[cells],
            anchor_scale=self.tail_map.anchor_scale[cells],
        )
        return rows.to_latent(torch.from_numpy(points)).numpy()

    def data(self, latent: Any) -> Any:
        """Return the data values of ``(N, Q, D)`` latent coordinates at every position."""
        if self.tail_map is None:
            return latent
        return self.tail_map.to_data(torch.from_numpy(latent)).numpy()

    def cdf(self, points: Any, cells: slice) -> Any:
        """Return the ``(n, Q, D)`` distribution function at ``(n, Q, D)`` data points.

        ``points`` covers the positions ``cells`` selects; under a tail map
        each is evaluated at its latent image, since the map is increasing.
        """
        points = self.latent(points, cells)
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

    Under a per-column tail map (``params`` carries
    :data:`~jointfm_inference.model.jointfm_1.tail_map.TAIL_MAP_KEYS`) the
    bracket's ends are the mapped component quantiles and every bisection
    step evaluates the latent distribution function at the midpoint's latent
    image. The map is increasing, so the root is the mapped latent quantile,
    and the bisection runs in data units to the same tolerance.

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
    tail_map = ColumnTailMap.from_params(params)
    mixtures = _MarginalMixtures(
        weights=_host(F.softmax(params["logits"].double(), dim=-1), cells),
        df=_host(params["df"], cells),
        locations=_host(means, cells),
        scales=_host(component_marginal_scales(params), cells),
        tail_map=(
            None
            if tail_map is None
            else ColumnTailMap(
                **{
                    name: torch.from_numpy(_host(value, cells)).unsqueeze(1)
                    for name, value in (
                        ("upper", tail_map.upper),
                        ("lower", tail_map.lower),
                        ("anchor_location", tail_map.anchor_location),
                        ("anchor_scale", tail_map.anchor_scale),
                    )
                }
            )
        ),
    )

    level_axis = np.asarray(levels, dtype=np.float64).reshape(1, -1, 1)
    # (N, Q, K): standardized component quantiles; df is shared by columns.
    standardized = stdtrit(mixtures.df[:, None, :], level_axis)
    component_quantiles = (
        mixtures.locations[:, None] + mixtures.scales[:, None] * standardized[..., None]
    )
    lower = mixtures.data(np.ascontiguousarray(component_quantiles.min(axis=2)))
    upper = mixtures.data(np.ascontiguousarray(component_quantiles.max(axis=2)))

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
    "MAPPED_MEAN_TOLERANCE",
    "MAX_BISECTION_STEPS",
    "QUANTILE_TOLERANCE",
    "attach_column_anchors",
    "column_anchors",
    "component_marginal_scales",
    "mapped_mixture_mean",
    "mixture_mean",
    "mixture_quantiles",
]
