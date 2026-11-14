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

"""Per-column tail map of the Student-t mixture head.

A head with a per-column tail map keeps its Student-t mixture as a latent law
and maps every column's latent coordinate ``x_d`` to its data value through
one increasing function per column:

    y_d = a_d + s_d * g(delta_d, (x_d - a_d) / s_d),
    g(delta, u) = sinh(delta * asinh(u)) / delta,

the sinh-arcsinh transformation of Jones & Pewsey (2009, "Sinh-arcsinh
distributions", Biometrika), scaled to unit slope at the origin. ``delta = 1``
is the identity. Each column has two tail parameters: ``delta+`` applies
where the standardized coordinate is non-negative and ``delta-`` below the
anchor, so one tail can be heavier than the other. On a latent component with
``nu`` degrees of freedom, column ``d``'s upper tail falls like a power law of
index ``nu / delta+`` and its lower one ``nu / delta-``. The branches meet at
the anchor with equal value, slope, and second derivative.

The map is shared by every mixture component of a column, so it standardizes
by anchors of the column rather than of one component: the column's mixture
center ``a_d`` and total spread ``s_d``
(:func:`jointfm_inference.model.jointfm_1.marginals.column_anchors`). They are
computed from the emitted joint mixture and kept unchanged through
conditioning. A mapping carries the map under four keys of shape
``(..., D)``, with no component axis: :data:`TAIL_UPPER_KEY`,
:data:`TAIL_LOWER_KEY`, :data:`ANCHOR_LOCATION_KEY`, and
:data:`ANCHOR_SCALE_KEY`.

Because one increasing function maps each column, every read-out follows from
the latent law: a data value's density is the latent density of its latent
image divided by the map's derivative there, a latent draw maps to a data
draw, a quantile maps to a quantile, and an equality or interval condition
on data values is the same condition on their latent images.

Every function here works in float64, whatever the input dtype, and returns
the input dtype. The inverse map grows like ``|v|**(1 / delta)``, so it is the
step that would overflow float32 first. Values outside the declared envelope
raise instead of being clipped: tail parameters outside
``[TAIL_PARAMETER_FLOOR, TAIL_PARAMETER_CEILING]`` and standardized
residuals beyond :data:`STANDARDIZED_RESIDUAL_LIMIT`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

TAIL_UPPER_KEY = "tail_upper"
TAIL_LOWER_KEY = "tail_lower"
ANCHOR_LOCATION_KEY = "tail_anchor_location"
ANCHOR_SCALE_KEY = "tail_anchor_scale"
TAIL_MAP_KEYS = (TAIL_UPPER_KEY, TAIL_LOWER_KEY, ANCHOR_LOCATION_KEY, ANCHOR_SCALE_KEY)
"""Keys under which a parameter mapping carries the map, each ``(..., D)``."""

SINH_ARCSINH_COLUMN_TAIL_MAP = "sinh_arcsinh_column"
"""Name ``/healthz`` reports for this module's two-sided map, one per column."""

# The envelope a checkpoint's tail parameter bounds must lie in, inside which
# the float64 evaluation below is verified.
TAIL_PARAMETER_FLOOR = 0.2
TAIL_PARAMETER_CEILING = 1.9
# Beyond this standardized residual delta * v approaches float32's range.
STANDARDIZED_RESIDUAL_LIMIT = 1e30

_LOG_TWO = math.log(2.0)
_COMPUTE_DTYPE = torch.float64


def tail_parameter(raw: Tensor, *, minimum: float, maximum: float) -> Tensor:
    """Squash raw readout outputs into tail parameters inside ``(minimum, maximum)``.

    ``log delta = c * tanh(r / c)`` with ``c = log(maximum)`` for ``r >= 0``
    and ``c = log(1 / minimum)`` for ``r < 0``, so ``delta = 1`` with unit
    slope at ``r = 0``: a zero readout is the identity map. Once ``tanh``
    rounds to one, ``exp`` can round one unit in the last place past a bound,
    which the final clamp absorbs.

    Raises:
        RuntimeError: when ``raw`` carries a non-finite value.
    """
    if not bool(torch.isfinite(raw).all()):
        raise RuntimeError("tail_parameter received non-finite raw values")
    upper = math.log(maximum)
    lower = math.log(1.0 / minimum)
    log_tail = torch.where(
        raw >= 0, upper * torch.tanh(raw / upper), lower * torch.tanh(raw / lower)
    )
    return torch.clamp(torch.exp(log_tail), min=minimum, max=maximum)


def _log_cosh(x: Tensor) -> Tensor:
    """Return ``log cosh(x)`` without overflow."""
    magnitude = x.abs()
    return magnitude + F.softplus(-2.0 * magnitude) - _LOG_TWO


def side_tail(standardized: Tensor, upper: Tensor, lower: Tensor) -> Tensor:
    """Return each cell's tail parameter: ``upper`` at or above the anchor, ``lower`` below.

    The map keeps signs, so a standardized latent coordinate and its data
    residual pick the same side. At the anchor every function of this module
    gives the same value whichever side is picked.
    """
    return torch.where(standardized >= 0, upper, lower)


def map_standardized(latent: Tensor, tail: Tensor) -> Tensor:
    """Return ``g(delta, u)`` for standardized latent coordinates ``u``."""
    return torch.sinh(tail * torch.asinh(latent)) / tail


def inverse_standardized(standardized: Tensor, tail: Tensor) -> Tensor:
    """Return ``g^-1(delta, v) = sinh(asinh(delta * v) / delta)`` for standardized residuals ``v``."""
    return torch.sinh(torch.asinh(tail * standardized) / tail)


def log_derivative_at_residual(standardized: Tensor, tail: Tensor) -> Tensor:
    """Return ``log g'(delta, g^-1(delta, v))`` without forming the latent coordinate.

    With ``A = asinh(delta * v) / delta``, which equals ``asinh(g^-1(v))``,
    the log-derivative is ``log cosh(delta * A) - log cosh(A)``: zero at the
    identity and finite wherever ``v`` is.
    """
    latent_asinh = torch.asinh(tail * standardized) / tail
    return _log_cosh(tail * latent_asinh) - _log_cosh(latent_asinh)


def _require_finite(result: Tensor, what: str) -> None:
    """Fail when a mapped value overflowed its dtype.

    Raises:
        RuntimeError: when ``result`` holds a non-finite value.
    """
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError(f"the tail map's {what} overflows {result.dtype}")


@dataclass(frozen=True)
class ColumnTailMap:
    """One increasing map per column: tail parameters and the anchors it standardizes by.

    Attributes:
        upper: ``(..., D)`` upper tail parameters ``delta+``.
        lower: ``(..., D)`` lower tail parameters ``delta-``.
        anchor_location: ``(..., D)`` anchor locations ``a``.
        anchor_scale: ``(..., D)`` strictly positive anchor scales ``s``.

    Raises:
        ValueError: when the four tensors differ in shape, lack a column axis,
            hold a tail parameter outside ``[TAIL_PARAMETER_FLOOR,
            TAIL_PARAMETER_CEILING]``, or hold a non-finite anchor or a
            non-positive anchor scale.
    """

    upper: Tensor
    lower: Tensor
    anchor_location: Tensor
    anchor_scale: Tensor

    def __post_init__(self) -> None:
        """Refuse inconsistent shapes and values outside the envelope."""
        shape = self.upper.shape
        if len(shape) == 0:
            raise ValueError("a tail map needs a trailing column axis")
        for name in ("lower", "anchor_location", "anchor_scale"):
            if getattr(self, name).shape != shape:
                raise ValueError(
                    f"tail map {name} {tuple(getattr(self, name).shape)} must share "
                    f"the upper tail parameters' shape {tuple(shape)}"
                )
        for name in ("upper", "lower"):
            tail = getattr(self, name)
            inside = (tail >= TAIL_PARAMETER_FLOOR) & (tail <= TAIL_PARAMETER_CEILING)
            if not bool(inside.all()):
                raise ValueError(
                    f"{name} tail parameters must lie in [{TAIL_PARAMETER_FLOOR}, "
                    f"{TAIL_PARAMETER_CEILING}]; got values from "
                    f"{float(tail.min())} to {float(tail.max())}"
                )
        if not bool(torch.isfinite(self.anchor_location).all()):
            raise ValueError("tail map anchor locations must be finite")
        scale = self.anchor_scale
        if not bool((torch.isfinite(scale) & (scale > 0)).all()):
            raise ValueError("tail map anchor scales must be finite and positive")

    @classmethod
    def from_params(cls, params: Mapping[str, Tensor]) -> Optional["ColumnTailMap"]:
        """Return the map a parameter mapping carries, or ``None`` when it carries none.

        Raises:
            ValueError: when the mapping carries some of :data:`TAIL_MAP_KEYS`
                but not all of them.
        """
        carried = [key for key in TAIL_MAP_KEYS if key in params]
        if not carried:
            return None
        if len(carried) != len(TAIL_MAP_KEYS):
            missing = sorted(set(TAIL_MAP_KEYS) - set(carried))
            raise ValueError(
                f"a tail map needs all of {TAIL_MAP_KEYS}; missing {missing}"
            )
        return cls(
            upper=params[TAIL_UPPER_KEY],
            lower=params[TAIL_LOWER_KEY],
            anchor_location=params[ANCHOR_LOCATION_KEY],
            anchor_scale=params[ANCHOR_SCALE_KEY],
        )

    def to_params(self) -> dict[str, Tensor]:
        """Return the map under :data:`TAIL_MAP_KEYS`."""
        return {
            TAIL_UPPER_KEY: self.upper,
            TAIL_LOWER_KEY: self.lower,
            ANCHOR_LOCATION_KEY: self.anchor_location,
            ANCHOR_SCALE_KEY: self.anchor_scale,
        }

    def reshape(self, *shape: int) -> "ColumnTailMap":
        """Return the map with every tensor reshaped to ``shape``."""
        return ColumnTailMap(
            upper=self.upper.reshape(*shape),
            lower=self.lower.reshape(*shape),
            anchor_location=self.anchor_location.reshape(*shape),
            anchor_scale=self.anchor_scale.reshape(*shape),
        )

    def columns(self, index: Sequence[int] | Tensor) -> "ColumnTailMap":
        """Return the map of the columns ``index`` selects, in that order."""
        positions = torch.as_tensor(
            index, dtype=torch.long, device=self.upper.device
        ).reshape(-1)
        return ColumnTailMap(
            upper=self.upper.index_select(-1, positions),
            lower=self.lower.index_select(-1, positions),
            anchor_location=self.anchor_location.index_select(-1, positions),
            anchor_scale=self.anchor_scale.index_select(-1, positions),
        )

    def _parts(self, like: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return the four tensors in the compute dtype on ``like``'s device."""
        device = like.device
        return (
            self.upper.to(device=device, dtype=_COMPUTE_DTYPE),
            self.lower.to(device=device, dtype=_COMPUTE_DTYPE),
            self.anchor_location.to(device=device, dtype=_COMPUTE_DTYPE),
            self.anchor_scale.to(device=device, dtype=_COMPUTE_DTYPE),
        )

    def to_latent(self, values: Tensor) -> Tensor:
        """Map data values ``(..., D)`` to their latent images, in ``values``' dtype.

        Raises:
            ValueError: when a value is not finite or its standardized
                residual exceeds :data:`STANDARDIZED_RESIDUAL_LIMIT`.
            RuntimeError: when a latent image overflows ``values``' dtype.
        """
        upper, lower, location, scale = self._parts(values)
        standardized = (values.to(_COMPUTE_DTYPE) - location) / scale
        _require_residual_in_envelope(standardized)
        tail = side_tail(standardized, upper, lower)
        latent = (location + scale * inverse_standardized(standardized, tail)).to(
            dtype=values.dtype
        )
        _require_finite(latent, "inverse")
        return latent

    def to_data(self, latent: Tensor) -> Tensor:
        """Map latent coordinates ``(..., D)`` to data values, in ``latent``'s dtype.

        Raises:
            ValueError: when a latent coordinate is not finite.
            RuntimeError: when a data value overflows ``latent``'s dtype.
        """
        if not bool(torch.isfinite(latent).all()):
            raise ValueError("the tail map received a non-finite latent coordinate")
        upper, lower, location, scale = self._parts(latent)
        standardized = (latent.to(_COMPUTE_DTYPE) - location) / scale
        tail = side_tail(standardized, upper, lower)
        data = (location + scale * map_standardized(standardized, tail)).to(
            dtype=latent.dtype
        )
        _require_finite(data, "map")
        return data

    def log_derivative(self, values: Tensor) -> Tensor:
        """Return ``log g'`` at every data value ``(..., D)``, in float64.

        ``dy_d / dx_d = g'(delta_d, u_d)``, since the anchor scale cancels, so
        a data value's log-density is its latent image's log-density minus
        the sum of these over its columns.

        Raises:
            ValueError: when a value is not finite or its standardized
                residual exceeds :data:`STANDARDIZED_RESIDUAL_LIMIT`.
        """
        upper, lower, location, scale = self._parts(values)
        standardized = (values.to(_COMPUTE_DTYPE) - location) / scale
        _require_residual_in_envelope(standardized)
        return log_derivative_at_residual(
            standardized, side_tail(standardized, upper, lower)
        )


def _require_residual_in_envelope(standardized: Tensor) -> None:
    """Fail on a non-finite standardized residual or one beyond the envelope.

    Raises:
        ValueError: when a residual is not finite or exceeds
            :data:`STANDARDIZED_RESIDUAL_LIMIT` in magnitude.
    """
    if not bool((standardized.abs() <= STANDARDIZED_RESIDUAL_LIMIT).all()):
        raise ValueError(
            "tail map standardized residuals must be finite with magnitude at most "
            f"{STANDARDIZED_RESIDUAL_LIMIT:.0e}"
        )


def mapped_mixture_log_prob(
    *,
    logits: Tensor,
    means: Tensor,
    chol: Tensor,
    df: Tensor,
    tail_map: ColumnTailMap,
    values: Tensor,
) -> Tensor:
    """Return the log-density of data values under a Student-t mixture mapped per column.

    Every value maps to its latent image ``x``, which every component scores
    with its multivariate Student-t density, and the map's log-derivative is
    subtracted once, since one map serves every component:

        log p(y) = logsumexp_k(log pi_k + log t(x; mu_k, Sigma_k, nu_k))
                   - sum_d log g'(delta_d, u_d).

    The latent residual ``x - mu_k`` is ``(a - mu_k) + s * sinh(A)`` with
    ``A = asinh(g^-1(v))``, divided by its largest entry before the
    triangular solve, so the squared Mahalanobis distance is reassembled in
    log space and stays finite wherever the density is.

    Args:
        logits: ``(N, K)`` mixture logits; they need not be normalized.
        means: ``(N, K, D)`` latent component locations.
        chol: ``(N, K, D, D)`` lower Cholesky factors of the latent scales.
        df: ``(N, K)`` degrees of freedom.
        tail_map: The map of the ``(N, D)`` columns.
        values: ``(N, D)`` data values.

    Returns:
        ``(N,)`` log-densities in float64.

    Raises:
        ValueError: when a value's standardized residual leaves the envelope.
    """
    upper, lower, location, scale = tail_map._parts(values)
    standardized = (values.to(_COMPUTE_DTYPE) - location) / scale
    _require_residual_in_envelope(standardized)
    tail = side_tail(standardized, upper, lower)
    latent_asinh = torch.asinh(tail * standardized) / tail
    log_jacobian = (_log_cosh(tail * latent_asinh) - _log_cosh(latent_asinh)).sum(
        dim=-1
    )

    means_64 = means.to(_COMPUTE_DTYPE)
    chol_64 = chol.to(_COMPUTE_DTYPE)
    df_64 = df.to(_COMPUTE_DTYPE)
    columns = values.shape[-1]
    residual = (location.unsqueeze(-2) - means_64) + (
        scale * torch.sinh(latent_asinh)
    ).unsqueeze(-2)
    largest = residual.abs().amax(dim=-1, keepdim=True)
    row_scale = torch.where(largest > 0, largest, torch.ones_like(largest))
    whitened = torch.linalg.solve_triangular(
        chol_64, (residual / row_scale).unsqueeze(-1), upper=False
    ).squeeze(-1)
    scaled_squared = whitened.pow(2).sum(dim=-1)
    nonzero = scaled_squared > 0
    log_ratio = (
        2.0 * torch.log(row_scale.squeeze(-1))
        + torch.log(
            torch.where(nonzero, scaled_squared, torch.ones_like(scaled_squared))
        )
        - torch.log(df_64)
    )
    log_kernel = torch.where(
        nonzero, F.softplus(log_ratio), torch.zeros_like(log_ratio)
    )
    log_det = 2.0 * torch.log(torch.diagonal(chol_64, dim1=-2, dim2=-1)).sum(dim=-1)
    component_log_prob = (
        torch.lgamma((df_64 + columns) / 2.0)
        - torch.lgamma(df_64 / 2.0)
        - (columns / 2.0) * torch.log(df_64 * math.pi)
        - 0.5 * log_det
        - ((df_64 + columns) / 2.0) * log_kernel
    )
    log_weights = F.log_softmax(logits.to(_COMPUTE_DTYPE), dim=-1)
    return torch.logsumexp(log_weights + component_log_prob, dim=-1) - log_jacobian


__all__ = [
    "ANCHOR_LOCATION_KEY",
    "ANCHOR_SCALE_KEY",
    "SINH_ARCSINH_COLUMN_TAIL_MAP",
    "STANDARDIZED_RESIDUAL_LIMIT",
    "TAIL_LOWER_KEY",
    "TAIL_MAP_KEYS",
    "TAIL_PARAMETER_CEILING",
    "TAIL_PARAMETER_FLOOR",
    "TAIL_UPPER_KEY",
    "ColumnTailMap",
    "inverse_standardized",
    "log_derivative_at_residual",
    "map_standardized",
    "mapped_mixture_log_prob",
    "side_tail",
    "tail_parameter",
]
