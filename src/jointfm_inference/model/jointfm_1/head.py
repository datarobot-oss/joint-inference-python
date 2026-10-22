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

from jointfm_inference.model.jointfm_1.readouts import (
    ColumnFactorReadout,
    ColumnLocationReadout,
    emit_lowrank_covariance,
)
from jointfm_inference.model.linalg import (
    apply_selected_cholesky,
    cholesky_from_lowrank,
    compute_whitened_residuals,
)

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
    ``covariance_rank`` factor ``U``. The mixture logits, degrees of freedom,
    pooled means and pooled covariance diagonal are decoded from the pooled
    per-horizon context; each column's factor rows and its mean and diagonal
    offsets are decoded from that column's own horizon token. A component's
    degrees of freedom are ``min_df + exp(out_log_df(h))``, so the projection
    emits the log of their excess over the floor. Emitted keys:
    ``logits`` ``(B, H, K)``, ``means`` ``(B, H, K, D)``, ``df`` ``(B, H, K)``,
    ``lowrank_diag`` ``(B, H, K, D)`` and ``lowrank_U`` ``(B, H, K, D, rank)``,
    with ``D`` the column capacity.
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
    ):
        """Build the trunk, the pooled projections, and the per-column readouts."""
        super().__init__()
        self.D = D
        self.K = K
        self.min_df = min_df
        self.min_scale = min_scale
        self.covariance_rank = int(covariance_rank)

        layers: list[nn.Module] = []
        in_dim = d_model
        for _ in range(max(1, trunk_layers)):
            layers.extend([nn.Linear(in_dim, trunk_hidden), nn.GELU()])
            in_dim = trunk_hidden
        self.trunk = nn.Sequential(*layers)

        self.out_logits = nn.Linear(trunk_hidden, K)
        self.out_means = nn.Linear(trunk_hidden, K * D)
        self.out_log_df = nn.Linear(trunk_hidden, K)
        self.out_diag = nn.Linear(trunk_hidden, K * D)
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

    def forward(self, x: Tensor, column_tokens: Tensor) -> Dict[str, Tensor]:
        """Map the pooled context and the per-column tokens to mixture parameters.

        Args:
            x: ``(B, H, d_model)`` pooled per-horizon context.
            column_tokens: ``(B, H, D_active, d_model)`` per-column horizon
                tokens; absent columns up to the capacity ``D`` get zero
                factor rows and offsets.
        """
        B, H, _ = x.shape
        h = self.trunk(x)
        mean_offset, raw_diag_offset = self.column_location(
            column_tokens, h, capacity=self.D
        )
        logits = self.out_logits(h)
        means = self.out_means(h).reshape(B, H, self.K, self.D) + mean_offset
        log_df_excess = self.out_log_df(h).reshape(B, H, self.K)
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
                raw_diag=self.out_diag(h).reshape(B, H, self.K, self.D)
                + raw_diag_offset,
                column_factor=self.column_factor,
                hidden=h,
                column_tokens=column_tokens,
                capacity=self.D,
                min_scale=self.min_scale,
                dtype=logits.dtype,
            )
        )
        return out

    def log_prob(self, params: Dict[str, Tensor], y: Tensor) -> Tensor:
        """Return the ``(B, H)`` mixture log-density of ``y`` ``(B, H, D)``.

        ``params`` may be the emitted parameters or a conditioned mapping with
        a dense ``chol``; the component count is read from ``logits``, so a
        pooled conditional mixture is accepted too.
        """
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

    def sample(
        self,
        params: Dict[str, Tensor],
        num_samples: int = 1,
        rng: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Draw ``(B * num_samples, H, D)`` samples, batch-major."""
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

        L_flat = _component_cholesky(params, B * H, K)
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
        gamma_samples = torch._standard_gamma(df_selected / 2.0, generator=rng)
        w = torch.clamp(2.0 * gamma_samples, min=MIN_CHI2_DRAW)
        samples = means_selected + z_mvn * torch.sqrt(df_selected / w).unsqueeze(-1)
        return (
            samples.reshape(B, H, num_samples, D)
            .permute(0, 2, 1, 3)
            .reshape(B * num_samples, H, D)
        )

    def _samples_by_batch(self, samples: Tensor, num_samples: int) -> Tensor:
        """Reshape ``(B * num_samples, H, D)`` draws to ``(B, num_samples, H, D)``."""
        total, horizon_count, dim = samples.shape
        return samples.view(total // num_samples, num_samples, horizon_count, dim)

    def mean(
        self,
        params: Dict[str, Tensor],
        num_samples: int = 500,
        rng: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Return the Monte Carlo mean ``(B, H, D)`` of ``num_samples`` draws.

        Sampling keeps the mean consistent with ``quantiles`` and ``sample``.
        """
        samples = self.sample(params, num_samples=num_samples, rng=rng)
        return self._samples_by_batch(samples, num_samples).mean(dim=1)

    def quantiles(
        self,
        params: Dict[str, Tensor],
        q: Sequence[float],
        num_samples: int = 1000,
        rng: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Return Monte Carlo marginal quantiles ``(len(q), B, H, D)``."""
        samples = self._samples_by_batch(
            self.sample(params, num_samples=num_samples, rng=rng), num_samples
        )
        q_t = torch.as_tensor(q, device=samples.device, dtype=samples.dtype)
        return torch.quantile(samples, q_t, dim=1)


__all__ = [
    "AUTO_STANDARDIZATION_CENTER_KEY",
    "AUTO_STANDARDIZATION_METADATA_KEYS",
    "AUTO_STANDARDIZATION_SCALE_KEY",
    "StudentTHead",
]
