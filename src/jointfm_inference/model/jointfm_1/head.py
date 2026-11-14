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

"""Student-t mixture head of the JointFM 1 network."""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from jointfm_inference.core.timing import (
    SAMPLE_ASSEMBLE_STAGE,
    SAMPLE_CHOLESKY_STAGE,
    SAMPLE_GAMMA_STAGE,
    SAMPLE_GAUSSIAN_STAGE,
    CudaTimer,
)
from jointfm_inference.model.jointfm_1.marginals import (
    mixture_mean,
    mixture_quantiles,
)
from jointfm_inference.model.jointfm_1.readouts import (
    MEAN_OUTPUT,
    RAW_DIAG_OUTPUT,
    ColumnFactorReadout,
    ColumnLocationReadout,
    ColumnTailReadout,
    emit_lowrank_covariance,
)
from jointfm_inference.model.jointfm_1.tail_map import (
    TAIL_LOWER_KEY,
    TAIL_UPPER_KEY,
    ColumnTailMap,
    mapped_mixture_log_prob,
    tail_parameter,
)
from jointfm_inference.model.linalg import (
    apply_selected_cholesky,
    cholesky_from_lowrank,
    compute_whitened_residuals,
)
from jointfm_inference.model.precision import upcast_reduced_precision

# Keys under which the network attaches the standardization statistics of the
# history window to its emitted parameters. They describe the window, not a
# column of the emitted joint, so code that reads emitted parameters per column
# or per emission position must leave them out.
AUTO_STANDARDIZATION_CENTER_KEY = "_internal_center"
AUTO_STANDARDIZATION_SCALE_KEY = "_internal_scale"
AUTO_STANDARDIZATION_METADATA_KEYS = (
    AUTO_STANDARDIZATION_CENTER_KEY,
    AUTO_STANDARDIZATION_SCALE_KEY,
)

MIN_POSITIVE_PROB = 1e-12
MIN_CHOLESKY_DIAGONAL = 1e-6
MAX_DF = 1e4
# Capping the exponent of the df readout keeps it finite in reduced precision.
MAX_LOG_DF_EXCESS = math.log(MAX_DF)
# Added to the degrees-of-freedom floor so it stays strictly above ``min_df``.
DF_EPS = 1e-4
# Floor of the chi-square draw, keeping the Student-t scale sqrt(df / w) finite.
MIN_CHI2_DRAW = 1e-6


def _gumbel_argmax_one_hot(
    logits: Tensor, *, num_samples: int, rng: Optional[torch.Generator]
) -> Tensor:
    """Draw ``num_samples`` one-hot component choices per row of ``logits`` ``(N, K)``.

    Returns ``(N, num_samples, K)`` one-hot draws in ``logits.dtype``; the
    argmax of Gumbel-perturbed logits is an exact categorical draw.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be >= 1")
    batch_size, num_components = logits.shape
    uniform = torch.rand(
        batch_size,
        num_samples,
        num_components,
        generator=rng,
        device=logits.device,
        dtype=torch.float32,
    )
    eps = torch.finfo(uniform.dtype).eps
    gumbels = -torch.log(-torch.log(uniform.clamp(min=eps, max=1.0 - eps)))
    perturbed = logits.float().unsqueeze(1).expand(-1, num_samples, -1) + gumbels
    indices = torch.softmax(perturbed, dim=-1).argmax(dim=-1)
    one_hot = F.one_hot(indices, num_classes=num_components)
    return one_hot.to(dtype=logits.dtype)


def _component_cholesky(params: Dict[str, Tensor], B_H: int, K: int) -> Tensor:
    """Return the ``(B_H, K, D, D)`` Cholesky factor of every component's scale.

    The network emits ``lowrank_diag`` and ``lowrank_U``; a conditioned
    mixture carries a dense ``chol`` instead.
    """
    if "chol" in params:
        chol = params["chol"]
        return chol.reshape(B_H, K, chol.shape[-2], chol.shape[-1])
    lowrank_U = params["lowrank_U"]
    return cholesky_from_lowrank(
        params["lowrank_diag"].reshape(B_H, K, -1),
        lowrank_U.reshape(B_H, K, lowrank_U.shape[-2], lowrank_U.shape[-1]),
    )


class StudentTHead(nn.Module):
    """Mixture of ``K`` multivariate Student-t distributions per horizon.

    Every component covariance is ``diag(d) + U U^T`` with a rank
    ``covariance_rank`` factor ``U``. The mixture logits and degrees of
    freedom describe the whole sample and are decoded from the pooled
    per-horizon context alone. Every column's mean and raw covariance diagonal
    is a base shared by all columns, decoded from the pooled context
    (``mean_base`` and ``raw_diag_base``, one value per component), plus that
    column's term decoded from its own horizon token
    (:class:`ColumnLocationReadout`); each column's factor rows come from its
    own token too (:class:`ColumnFactorReadout`). No output depends on a
    column's position, so permuting the columns permutes every per-column
    parameter. A component's degrees of freedom are
    ``min_df + exp(out_log_df(h))``, so the projection emits the log of their
    excess over the floor. Emitted keys:
    ``logits`` ``(B, H, K)``, ``means`` ``(B, H, K, D)``, ``df`` ``(B, H, K)``,
    ``lowrank_diag`` ``(B, H, K, D)`` and ``lowrank_U`` ``(B, H, K, D, rank)``,
    with ``D`` the column capacity, which sizes no parameter.

    A head built with ``tail_parameter_bounds`` also has a per-column tail map
    (:mod:`jointfm_inference.model.jointfm_1.tail_map`): its mixture is the
    latent law, and :class:`ColumnTailReadout` decodes each column's upper and
    lower tail parameters from the location readout's features, emitted in
    float32 under ``TAIL_UPPER_KEY`` and ``TAIL_LOWER_KEY`` ``(B, H, D)``.
    The network attaches the map's anchors after it rescales the mixture to
    data units, so every distribution method takes a mapping that carries
    the whole map, and refuses one that does not match the head.
    """

    def __init__(
        self,
        *,
        d_model: int,
        D: int,
        K: int,
        trunk_layers: int,
        trunk_hidden: int,
        covariance_rank: int,
        min_df: float,
        min_scale: float,
        tail_parameter_bounds: Optional[tuple[float, float]] = None,
    ):
        """Build the trunk, the pooled projections, and the per-column readouts.

        ``tail_parameter_bounds`` is the ``(minimum, maximum)`` of every tail
        parameter of a head with a per-column tail map, ``None`` for a head
        without one.
        """
        super().__init__()
        self.D = D
        self.K = K
        self.min_df = min_df
        self.min_scale = min_scale
        self.covariance_rank = int(covariance_rank)
        self.tail_parameter_bounds = tail_parameter_bounds

        layers: list[nn.Module] = []
        in_dim = d_model
        for _ in range(max(1, trunk_layers)):
            layers.extend([nn.Linear(in_dim, trunk_hidden), nn.GELU()])
            in_dim = trunk_hidden
        self.trunk = nn.Sequential(*layers)

        self.out_logits = nn.Linear(trunk_hidden, K)
        self.mean_base = nn.Linear(trunk_hidden, K)
        self.out_log_df = nn.Linear(trunk_hidden, K)
        self.raw_diag_base = nn.Linear(trunk_hidden, K)
        self.column_location = ColumnLocationReadout(
            column_dim=d_model,
            context_dim=trunk_hidden,
            hidden=trunk_hidden,
            n_components=K,
        )
        self.column_factor = ColumnFactorReadout(
            column_dim=d_model,
            context_dim=trunk_hidden,
            hidden=trunk_hidden,
            n_components=K,
            rank=self.covariance_rank,
        )
        self.column_tail = (
            None
            if tail_parameter_bounds is None
            else ColumnTailReadout(hidden=trunk_hidden)
        )

    def forward(self, x: Tensor, column_tokens: Tensor) -> Dict[str, Tensor]:
        """Map the pooled context and the per-column tokens to mixture parameters.

        Args:
            x: ``(B, H, d_model)`` pooled per-horizon context.
            column_tokens: ``(B, H, D_active, d_model)`` per-column horizon
                tokens; absent columns up to the capacity ``D`` get zero
                factor rows and sit at the shared base, and a head with a
                tail map gives them the identity map.
        """
        h = self.trunk(x)
        column_features = self.column_location.features(column_tokens, h)
        column_terms = self.column_location.terms(column_features, capacity=self.D)
        logits = self.out_logits(h)
        means = self.mean_base(h).unsqueeze(-1) + column_terms[..., MEAN_OUTPUT]
        log_df_excess = self.out_log_df(h)
        df = (
            torch.exp(log_df_excess.clamp(max=MAX_LOG_DF_EXCESS)) + self.min_df + DF_EPS
        )
        df = torch.clamp(df, max=MAX_DF)
        # Autocast may compute the exponential in float32.
        if df.dtype != logits.dtype:
            df = df.to(dtype=logits.dtype)
        out = {"logits": logits, "means": means, "df": df}
        out.update(
            emit_lowrank_covariance(
                raw_diag=self.raw_diag_base(h).unsqueeze(-1)
                + column_terms[..., RAW_DIAG_OUTPUT],
                column_factor=self.column_factor,
                hidden=h,
                column_tokens=column_tokens,
                capacity=self.D,
                min_scale=self.min_scale,
                dtype=logits.dtype,
            )
        )
        if self.column_tail is not None and self.tail_parameter_bounds is not None:
            minimum, maximum = self.tail_parameter_bounds
            # Squashed in float32: a reduced precision would round the tail
            # parameters to about three digits.
            raw = upcast_reduced_precision(
                {"raw": self.column_tail(column_features, self.D)}
            )["raw"]
            out[TAIL_UPPER_KEY], out[TAIL_LOWER_KEY] = tail_parameter(
                raw, minimum=minimum, maximum=maximum
            ).unbind(-1)
        return out

    def _tail_map(self, params: Dict[str, Tensor]) -> Optional[ColumnTailMap]:
        """Return the tail map ``params`` carries, refusing one that does not match the head.

        Raises:
            ValueError: when a head with a tail map gets parameters without
                one, or a head without a map gets parameters with one, which
                would score or draw a law other than the forecast.
        """
        tail_map = ColumnTailMap.from_params(params)
        if (tail_map is None) != (self.column_tail is None):
            raise ValueError(
                "StudentTHead has "
                f"{'no' if self.column_tail is None else 'a'} per-column tail map, "
                f"but the parameters {'carry' if tail_map is not None else 'lack'} one"
            )
        return tail_map

    def log_prob(self, params: Dict[str, Tensor], y: Tensor) -> Tensor:
        """Return the ``(B, H)`` mixture log-density of ``y`` ``(B, H, D)``.

        ``params`` may be the emitted parameters or a conditioned mapping with
        a dense ``chol``; the component count is read from ``logits``, so a
        pooled conditional mixture is accepted too. Under a tail map the
        density is :func:`~jointfm_inference.model.jointfm_1.tail_map.mapped_mixture_log_prob`,
        evaluated in float64.
        """
        tail_map = self._tail_map(params)
        if tail_map is not None:
            return self._mapped_log_prob(params, tail_map, y)
        logits = params["logits"]
        orig_dtype = logits.dtype
        compute_dtype = torch.float32
        B, H, D = y.shape
        B_H = B * H
        K = logits.shape[-1]
        logits_flat = logits.reshape(B_H, K).to(dtype=compute_dtype)
        means_flat = params["means"].reshape(B_H, K, -1).to(dtype=compute_dtype)
        df_flat = params["df"].reshape(B_H, K).to(dtype=compute_dtype)
        if not torch.isfinite(df_flat).all():
            raise RuntimeError(
                "StudentTHead.log_prob received non-finite degrees of freedom"
            )
        L = _component_cholesky(
            {name: value.to(dtype=compute_dtype) for name, value in params.items()},
            B_H,
            K,
        )
        diags = torch.clamp(
            torch.diagonal(L, dim1=-2, dim2=-1), min=MIN_CHOLESKY_DIAGONAL
        )
        log_det = 2 * torch.sum(torch.log(diags), dim=-1)
        const_term = (
            torch.lgamma((df_flat + D) / 2.0)
            - torch.lgamma(df_flat / 2.0)
            - (D / 2.0) * torch.log(df_flat * math.pi)
        )
        z = compute_whitened_residuals(
            y.reshape(B_H, 1, 1, D).to(dtype=compute_dtype),
            means_flat.unsqueeze(1),
            L.unsqueeze(1),
        )
        dist_sq = z.pow(2).sum(dim=-1).squeeze(1)
        log_pdf = (
            const_term
            - 0.5 * log_det
            - ((df_flat + D) / 2.0) * torch.log1p(dist_sq / df_flat)
        )
        log_like = torch.logsumexp(logits_flat + log_pdf, dim=-1) - torch.logsumexp(
            logits_flat, dim=-1
        )
        return log_like.reshape(B, H).to(dtype=orig_dtype)

    def _mapped_log_prob(
        self, params: Dict[str, Tensor], tail_map: ColumnTailMap, y: Tensor
    ) -> Tensor:
        """Return the ``(B, H)`` log-density of ``y`` under the mapped mixture."""
        logits = params["logits"]
        B, H, D = y.shape
        K = logits.shape[-1]
        double = {name: value.double() for name, value in params.items()}
        log_like = mapped_mixture_log_prob(
            logits=double["logits"].reshape(B * H, K),
            means=double["means"].reshape(B * H, K, D),
            chol=_component_cholesky(double, B * H, K),
            df=double["df"].reshape(B * H, K),
            tail_map=tail_map.reshape(B * H, D),
            values=y.reshape(B * H, D),
        )
        return log_like.reshape(B, H).to(dtype=logits.dtype)

    def sample(
        self,
        params: Dict[str, Tensor],
        num_samples: int = 1,
        rng: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Draw ``(B * num_samples, H, D)`` samples, batch-major.

        Under a tail map every latent draw maps through it, with the random
        stream of the latent draw.
        """
        tail_map = self._tail_map(params)
        logits = params["logits"]
        means = params["means"]
        df = params["df"]
        B, H, K = logits.shape
        D = means.shape[-1]
        probs = F.softmax(logits.float(), dim=-1).reshape(-1, K)
        row_sums = probs.sum(dim=-1)
        if not torch.isfinite(row_sums).all() or (row_sums <= MIN_POSITIVE_PROB).any():
            raise RuntimeError(
                "StudentTHead.sample produced invalid mixture probabilities"
            )

        with CudaTimer(SAMPLE_CHOLESKY_STAGE, logits.device):
            L_flat = _component_cholesky(params, B * H, K)
        with CudaTimer(SAMPLE_GAUSSIAN_STAGE, logits.device):
            means_flat = means.reshape(B * H, K, D)
            df_flat = df.reshape(B * H, K)
            selector = _gumbel_argmax_one_hot(
                logits.reshape(B * H, K), num_samples=num_samples, rng=rng
            )
            means_selected = torch.einsum("bnk,bkd->bnd", selector, means_flat)
            df_selected = torch.einsum("bnk,bk->bn", selector, df_flat)
            z = torch.randn(
                B * H,
                num_samples,
                D,
                device=logits.device,
                generator=rng,
                dtype=means.dtype,
            )
            z_mvn = apply_selected_cholesky(selector, L_flat, z)
        # A Student-t draw is a Gaussian draw scaled by sqrt(df / w), w ~ Chi2(df).
        with CudaTimer(SAMPLE_GAMMA_STAGE, logits.device):
            gamma_samples = torch._standard_gamma(df_selected / 2.0, generator=rng)
            w = torch.clamp(2.0 * gamma_samples, min=MIN_CHI2_DRAW)
        with CudaTimer(SAMPLE_ASSEMBLE_STAGE, logits.device):
            samples = means_selected + z_mvn * torch.sqrt(df_selected / w).unsqueeze(-1)
            if tail_map is not None:
                samples = tail_map.reshape(B * H, 1, D).to_data(samples)
            return (
                samples.reshape(B, H, num_samples, D)
                .permute(0, 2, 1, 3)
                .reshape(B * num_samples, H, D)
            )

    def mean(self, params: Dict[str, Tensor]) -> Tensor:
        """Return the exact ``(B, H, D)`` mean.

        Without a tail map it is the mixture-weighted locations; with one, a
        quadrature to a stated tolerance. It draws nothing, so it is the same
        for every seed
        (:func:`~jointfm_inference.model.jointfm_1.marginals.mixture_mean`).
        """
        self._tail_map(params)
        return mixture_mean(params)

    def quantiles(self, params: Dict[str, Tensor], q: Sequence[float]) -> Tensor:
        """Return exact marginal quantiles ``(len(q), B, H, D)``.

        Each is the root of its column's marginal distribution function,
        bisected to a stated tolerance without drawing
        (:func:`~jointfm_inference.model.jointfm_1.marginals.mixture_quantiles`).
        """
        self._tail_map(params)
        return mixture_quantiles(params, q)


__all__ = [
    "AUTO_STANDARDIZATION_CENTER_KEY",
    "AUTO_STANDARDIZATION_METADATA_KEYS",
    "AUTO_STANDARDIZATION_SCALE_KEY",
    "StudentTHead",
]
