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

"""Confining mixture components to a box, by distribution functions.

The served ``condition`` query mode answers an interval condition — a column
confined to a range — on the reduced mixture the equality step leaves behind.
A component truncated to a box is no longer Student-t, so the box
carries no closed-form density, and the box *probability* of a correlated
component has no elementary form either. What does survive is the univariate
structure: one Cholesky factorization turns the box event into a sequence of
univariate truncations, each one a difference of univariate distribution
functions, and each draw an inverse CDF inside that difference. This is the
separation-of-variables construction of Genz (1992), "Numerical computation of
multivariate normal probabilities", Journal of Computational and Graphical
Statistics 1, 141-149.

Two properties make it the right primitive for a served stress query. Every
draw lands inside the box by construction, so a far-tail region costs exactly
what a central one costs and the response never starves; and with a single
interval column the walk is exact — one CDF difference for the probability, one
inverse CDF for the draw, with every weight equal.

A Student-t component factorizes through the chain rule of its own
conditionals: given the columns already drawn, one column's conditional is a
Student-t whose degrees of freedom rise by one per drawn column and whose scale
is inflated by how far the drawn columns landed (Ding, 2016, "On the
Conditional Distribution of the Multivariate t Distribution", arXiv:1604.00561)
— the same two scalars the equality step applies, threaded one column at a
time. That keeps the walk one dimension wide per interval column, rather than
integrating over the Student-t's scaling variate as an extra dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from scipy.special import stdtr, stdtrit
from torch import Tensor

# The linear-space CDF difference an inverse-CDF draw needs underflows for a
# region past roughly 37 standard deviations, where the log-space probability
# this module also returns is still exact. Clamping the quantile argument keeps
# such a draw finite instead of poisoning the read-out with an infinity; the
# reported probability is what tells the caller the region is absurd.
_QUANTILE_FLOOR = 1e-300


@dataclass(frozen=True)
class BoxWalk:
    """One separation-of-variables pass over an interval block.

    Attributes:
        log_box_probability: ``(..., K)`` log probability each component
            assigns to the box. Exact when the block has one column, and the
            log of a quasi-Monte-Carlo mean otherwise.
        draws: ``(..., K, points, m)`` draws of the interval columns, all
            inside the box, in the original (unwhitened) coordinates.
        log_weights: ``(..., K, points)`` log importance weights of those
            draws against the truncated target. They are constant within a
            component when the block has one column, which is what makes that
            case exact; otherwise their spread is the estimator's only failure
            mode and is reported to the caller as an effective sample size.
    """

    log_box_probability: Tensor
    draws: Tensor
    log_weights: Tensor

    @property
    def is_exact(self) -> bool:
        """Whether this walk needed no estimation (a single interval column)."""
        return int(self.draws.shape[-1]) == 1


def effective_sample_size(log_weights: Tensor) -> Tensor:
    """Return Kish's effective sample size of one flat set of log weights.

    ``(sum w)^2 / sum w^2``, computed in log space so that weights spanning
    hundreds of orders of magnitude — the normal case for a tail region — do
    not underflow before the ratio is taken. The result says how many equally
    weighted draws the weighted set is worth, so it is at most the number of
    weights and falls toward one as a single draw takes over.
    """
    if log_weights.ndim != 1:
        raise ValueError(
            f"effective_sample_size expects a flat weight vector, got shape "
            f"{tuple(log_weights.shape)}"
        )
    log_sum = torch.logsumexp(log_weights, dim=0)
    log_sum_squares = torch.logsumexp(2.0 * log_weights, dim=0)
    return torch.exp(2.0 * log_sum - log_sum_squares)


def _sobol_points(
    dimension: int,
    count: int,
    *,
    rng: torch.Generator | None,
    device: torch.device,
) -> Tensor:
    """Draw a scrambled Sobol point set in the open unit cube.

    The walk integrates a smooth function of these points, which is where the
    quasi-random sequence earns its place over independent uniforms: the
    integrand is a product of CDF differences with no discontinuity inside the
    cube. Scrambling is seeded from ``rng`` so one request's answer is
    reproducible from its own seed.
    """
    seed = None
    if rng is not None:
        seed = int(
            torch.randint(0, 2**31 - 1, (1,), generator=rng, device=rng.device).item()
        )
    engine = torch.quasirandom.SobolEngine(
        dimension=dimension, scramble=True, seed=seed
    )
    points = engine.draw(count).to(device=device, dtype=torch.float64)
    # A scrambled Sobol point can land exactly on a boundary, where the inverse
    # CDF is infinite; the walk needs an interior point, not a uniform one.
    return points.clamp(_QUANTILE_FLOOR, 1.0 - 1e-16)


@dataclass(frozen=True)
class _ColumnInterval:
    """One column's admissible range, as the walk consumes it.

    Attributes:
        log_span: Log probability the column's conditional assigns to the
            range, which multiplies into the walk's importance weight.
        cdf_low: Distribution function at the lower end, in linear space,
            which the inverse-CDF draw interpolates from.
        cdf_high: Distribution function at the upper end.
    """

    log_span: Tensor
    cdf_low: Tensor
    cdf_high: Tensor


def _log_span(log_cdf_low: Tensor, log_cdf_high: Tensor) -> Tensor:
    """Return the log of a CDF difference from the two logs, without cancelling.

    Subtracting two distribution function values that agree to most of their
    digits — a narrow range deep in a tail, which is the case this mode exists
    for — loses those digits in linear space. Factoring the larger one out
    leaves the difference to ``log1p``, which keeps them.

    Far enough out, the larger of the two is itself zero to the last
    representable digit: past roughly ``1e100`` standardized units a Student-t
    distribution function underflows to zero. The difference of the two logs
    is then minus infinity minus minus infinity, and the NaN that produces would
    ride into the region probability the response reports. The span's own log
    at that depth is minus infinity, which is what the caller is told.
    """
    ratio = torch.log1p(-torch.exp(log_cdf_low - log_cdf_high))
    return torch.where(torch.isneginf(log_cdf_high), log_cdf_high, log_cdf_high + ratio)


def _to_numpy(value: Tensor) -> Any:
    """Hand one double-precision tensor to SciPy, whatever device it sits on."""
    return value.cpu().numpy()


@dataclass(frozen=True)
class _StudentTColumns:
    """The column law of a Student-t component: a conditional t per column.

    Conditioning a multivariate Student-t on the columns already drawn raises
    the degrees of freedom by one per column and inflates the scale by
    ``(nu + d) / (nu + j)``, where ``d`` is the squared Mahalanobis distance
    those columns landed at (Ding, 2016). Both scalars are the ones
    ``condition_student_t_mixture`` applies to a whole known block; walking one
    column at a time applies them ``j`` times instead, which is the same
    conditional by the chain rule.

    The distribution function and its inverse come from SciPy, because torch's
    ``StudentT`` implements neither ``cdf`` nor ``icdf``. The round trip to the
    host costs one synchronization per interval column on an accelerator, which
    is immaterial against the one emission position a conditioning request
    covers. The linear-space distribution function is safe to take the log of:
    a Student-t tail falls polynomially, so reaching the underflow floor would
    take a bound no finite request carries.

    Attributes:
        df: ``(..., K, 1)`` degrees of freedom of each component, carrying the
            trailing axis the per-point quantities broadcast against.
    """

    df: Tensor

    def column_df(self, column: int) -> Tensor:
        """Return the degrees of freedom of the conditional at one column."""
        return self.df + column

    def scale_multiplier(self, column: int, squared_distance: Tensor) -> Tensor:
        """Return the conditional scale inflation the drawn columns imply."""
        return ((self.df + squared_distance) / self.column_df(column)).sqrt()

    def interval(self, low: Tensor, high: Tensor, column: int) -> _ColumnInterval:
        """Return the admissible range of one standardized column."""
        df = self.column_df(column)
        cdf_low = self._cdf(df, low)
        cdf_high = self._cdf(df, high)
        return _ColumnInterval(
            log_span=_log_span(cdf_low.log(), cdf_high.log()),
            cdf_low=cdf_low,
            cdf_high=cdf_high,
        )

    def quantile(self, probability: Tensor, column: int) -> Tensor:
        """Return the standardized draw at one probability."""
        df = self.column_df(column).expand_as(probability)
        clamped = probability.clamp(_QUANTILE_FLOOR, 1.0 - 1e-16)
        return torch.as_tensor(
            stdtrit(_to_numpy(df), _to_numpy(clamped)),
            dtype=probability.dtype,
            device=probability.device,
        )

    def _cdf(self, df: Tensor, value: Tensor) -> Tensor:
        """Evaluate the central Student-t distribution function through SciPy."""
        return torch.as_tensor(
            stdtr(_to_numpy(df.expand_as(value)), _to_numpy(value)),
            dtype=value.dtype,
            device=value.device,
        )


def walk_box(
    means: Tensor,
    chol: Tensor,
    lower: Tensor,
    upper: Tensor,
    *,
    points: int,
    df: Tensor,
    rng: torch.Generator | None = None,
) -> BoxWalk:
    """Walk each component's interval block, column by column.

    Whitening by the component's Cholesky factor makes the box event a
    sequence of univariate constraints: given the columns already drawn, the
    next whitened coordinate is confined to one interval whose ends are the
    remaining slack divided by the diagonal entry. Its probability is a
    difference of univariate distribution functions, and drawing inside it
    is an inverse CDF, so the product of those differences is an unbiased
    estimate of the component's box probability and every draw satisfies the
    constraint.

    Args:
        means: ``(..., K, m)`` component means over the interval columns.
        chol: ``(..., K, m, m)`` lower-triangular Cholesky factors of those
            columns' covariance — the scale matrix for a Student-t mixture,
            which is the covariance only up to the ``nu / (nu - 2)`` factor.
        lower: ``(m,)`` lower bounds, ``-inf`` for an open side.
        upper: ``(m,)`` upper bounds, ``+inf`` for an open side.
        points: Number of quasi-random points per component.
        df: ``(..., K)`` degrees of freedom of each Student-t component.
        rng: Optional generator seeding the Sobol scramble.

    Returns:
        A :class:`BoxWalk` over the same leading axes as ``means``.

    Raises:
        ValueError: when the shapes disagree, when a bound pair is empty or
            open on both sides, when ``df`` is not strictly positive on the
            components' own axes, or when ``points`` is not positive.

    The walk runs in double precision regardless of the input dtype. Its
    working set is one emission position wide — components times points times
    interval columns — so the cost is immaterial, while the quantities it
    computes span the full range of a tail probability and the inverse-CDF step
    differences two nearly equal distribution function values.
    """
    if points <= 0:
        raise ValueError(f"points must be positive, got {points}")
    if means.ndim < 2 or chol.ndim != means.ndim + 1:
        raise ValueError(
            "walk_box expects means (..., K, m) and chol (..., K, m, m); got "
            f"{tuple(means.shape)} and {tuple(chol.shape)}"
        )
    block_width = int(means.shape[-1])
    if tuple(chol.shape[-2:]) != (block_width, block_width):
        raise ValueError(
            f"chol must be square at the block width {block_width}; got "
            f"{tuple(chol.shape[-2:])}"
        )
    if lower.shape != (block_width,) or upper.shape != (block_width,):
        raise ValueError(
            f"lower and upper must both have shape ({block_width},); got "
            f"{tuple(lower.shape)} and {tuple(upper.shape)}"
        )
    if not bool((lower < upper).all().item()):
        raise ValueError("every interval must have lower < upper")
    if bool((torch.isinf(lower) & torch.isinf(upper)).any().item()):
        raise ValueError("an interval open on both sides is not a condition")

    device = means.device
    means_64 = means.to(torch.float64)
    chol_64 = chol.to(torch.float64)
    lower_64 = lower.to(device=device, dtype=torch.float64)
    upper_64 = upper.to(device=device, dtype=torch.float64)

    if df.shape != means.shape[:-1]:
        raise ValueError(
            "df must carry one value per component, shaped like means without "
            f"its column axis {tuple(means.shape[:-1])}; got {tuple(df.shape)}"
        )
    if not bool((df > 0).all().item()):
        raise ValueError("df must be strictly positive")
    law = _StudentTColumns(df=df.to(device=device, dtype=torch.float64).unsqueeze(-1))

    quasi_points = _sobol_points(block_width, points, rng=rng, device=device)

    batch_shape = means_64.shape[:-1]
    whitened = torch.zeros(
        *batch_shape, points, block_width, dtype=torch.float64, device=device
    )
    log_weights = torch.zeros(*batch_shape, points, dtype=torch.float64, device=device)
    # Squared Mahalanobis distance of the columns drawn so far, which is what a
    # Student-t conditional scale depends on.
    squared_distance = torch.zeros(
        *batch_shape, points, dtype=torch.float64, device=device
    )

    for column in range(block_width):
        # Slack left by the columns already drawn, in the whitened frame.
        offset = means_64[..., column].unsqueeze(-1)
        if column:
            factor_row = chol_64[..., column, :column].unsqueeze(-2)
            offset = offset + (factor_row * whitened[..., :column]).sum(dim=-1)
        multiplier = law.scale_multiplier(column, squared_distance)
        scale = chol_64[..., column, column].unsqueeze(-1) * multiplier
        low = (lower_64[column] - offset) / scale
        high = (upper_64[column] - offset) / scale

        # Orienting the interval toward the left tail keeps both distribution
        # function values small, where their difference keeps its significant
        # digits; on the right tail both would sit against one.
        flipped = (low + high) > 0
        low_oriented = torch.where(flipped, -high, low)
        high_oriented = torch.where(flipped, -low, high)

        admissible = law.interval(low_oriented, high_oriented, column)
        log_weights = log_weights + admissible.log_span

        span = admissible.cdf_high - admissible.cdf_low
        quantile = admissible.cdf_low + quasi_points[:, column] * span
        # The draw is inside its range by construction, so clamping only undoes
        # the rounding of a probability that reached the floor of its own
        # inverse: a region deep enough for that is one the reported
        # probability already describes as absurd.
        drawn = law.quantile(quantile, column).clamp(
            min=low_oriented, max=high_oriented
        )
        standardized = torch.where(flipped, -drawn, drawn)
        whitened[..., column] = standardized * multiplier
        squared_distance = squared_distance + whitened[..., column] ** 2

    log_box_probability = torch.logsumexp(log_weights, dim=-1) - torch.log(
        torch.tensor(float(points), dtype=torch.float64, device=device)
    )
    draws = means_64.unsqueeze(-2) + torch.einsum(
        "...ij,...nj->...ni", chol_64, whitened
    )
    return BoxWalk(
        log_box_probability=log_box_probability,
        draws=draws,
        log_weights=log_weights,
    )
