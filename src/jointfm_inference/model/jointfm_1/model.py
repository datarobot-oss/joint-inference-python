"""The JointFM 1 network: tokens, factored encoder, pooling, Student-t head."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from jointfm_inference.model.jointfm_1.config import JointFMConfig
from jointfm_inference.model.jointfm_1.encoder import FactoredEncoder
from jointfm_inference.model.jointfm_1.head import (
    AUTO_STANDARDIZATION_CENTER_KEY,
    AUTO_STANDARDIZATION_SCALE_KEY,
    StudentTHead,
)
from jointfm_inference.model.jointfm_1.pooling import _PMA

# Floor of the history standard deviation used to standardize the input.
MIN_STANDARDIZATION_SCALE = 1e-6
PAST_TOKEN_TYPE = 0
QUERY_TOKEN_TYPE = 1
TOKEN_TYPES = 2


class SinusoidalPosition(nn.Module):
    """Fixed sinusoidal position table (Vaswani et al., 2017, arXiv:1706.03762)."""

    def __init__(self, dim: int, max_len: int, base_freq: float):
        """Build the ``(max_len, dim)`` table."""
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"SinusoidalPosition requires an even dim, got {dim}")
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(base_freq) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Deterministic, so it stays out of checkpoints.
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, idx: Tensor) -> Tensor:
        """Gather the rows at ``idx``."""
        return self.pe[idx]  # ty: ignore[not-subscriptable]


class JointFM(nn.Module):
    """Joint forecaster over a ``(time, column)`` token grid.

    The history ``(B, T, D)`` is standardized per column by its last value and
    its standard deviation, and every cell becomes one token. For each queried
    horizon and column a query token is appended on the time axis, at position
    ``origin + 1 + horizon`` of the history's own time table. The factored
    encoder reads history and query rows together; each horizon's column
    tokens are pooled, and the Student-t head decodes the joint distribution
    of all columns at every queried horizon in one pass. Emitted locations and
    scales are mapped back to data space.
    """

    def __init__(self, cfg: JointFMConfig):
        """Build the network for ``cfg``."""
        super().__init__()
        self.cfg = cfg
        self.D = cfg.column_capacity
        self.value_embed = nn.Linear(1, cfg.d_value)
        self.var_embed = nn.Embedding(self.D, cfg.d_model)
        self.token_type_embed = nn.Embedding(TOKEN_TYPES, cfg.d_model)
        self.time_embed = SinusoidalPosition(
            cfg.d_model,
            cfg.max_time_ctx + cfg.max_time_fut,
            base_freq=cfg.time_freq_base,
        )
        self.horizon_embed = SinusoidalPosition(
            cfg.d_model, cfg.max_time_fut, base_freq=cfg.horizon_freq_base
        )
        self.combine = nn.Linear(cfg.d_value + cfg.d_model, cfg.d_model)
        self.encoder = FactoredEncoder(
            num_layers=cfg.num_layers,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            layer_norm_eps=cfg.layer_norm_eps,
            time_blocks=cfg.attention_time_blocks,
            row_blocks=cfg.attention_row_blocks,
            cnn_kernel_size=cfg.attention_cnn_kernel_size,
            cnn_first_layers=cfg.attention_cnn_first_layers,
        )
        self.horizon_pool_set = _PMA(
            d_model=cfg.d_model,
            nhead=cfg.pooling_nhead,
            num_seeds=cfg.pooling_seeds,
            num_layers=cfg.pooling_layers,
            dim_feedforward=None,
            layer_norm_eps=cfg.layer_norm_eps,
        )
        self.head = StudentTHead(
            d_model=cfg.d_model,
            D=self.D,
            K=cfg.n_mixtures,
            trunk_layers=cfg.head_trunk_layers,
            trunk_hidden=(
                cfg.head_trunk_hidden
                if cfg.head_trunk_hidden is not None
                else cfg.d_model
            ),
            covariance_rank=cfg.covariance_rank,
            min_df=cfg.head_min_df,
            min_scale=cfg.head_min_scale,
        )

    def _history_tokens(self, past: Tensor, time_indices: Tensor) -> Tensor:
        """Return ``(B, T * D, d_model)`` time-major tokens of the standardized history."""
        batch_size, T_ctx, active_dim = past.shape
        device = past.device
        val_emb = self.value_embed(past.unsqueeze(-1))
        var_ids = torch.arange(active_dim, device=device).view(1, 1, active_dim)
        var_emb = self.var_embed(var_ids.expand(batch_size, T_ctx, active_dim))
        time_idx = time_indices.view(1, T_ctx, 1).expand(batch_size, T_ctx, active_dim)
        token_type = self.token_type_embed(
            torch.full((1,), PAST_TOKEN_TYPE, device=device, dtype=torch.long)
        ).view(1, 1, 1, -1)
        tokens = (
            self.combine(torch.cat([val_emb, var_emb], dim=-1))
            + self.time_embed(time_idx)
            + token_type
        )
        return tokens.view(batch_size, T_ctx * active_dim, self.cfg.d_model)

    def _query_tokens(
        self, horizons: Tensor, origin_index: Tensor, active_dim: int
    ) -> Tensor:
        """Return ``(1, H * D_active, d_model)`` horizon-major query tokens.

        A query token sums the column, horizon, and query-type embeddings and
        the time embedding at ``origin_index + 1 + horizon``.
        """
        horizon_count = horizons.shape[0]
        device = horizons.device
        hz_emb = self.horizon_embed(horizons)
        if self.cfg.horizon_embed_scale != 1.0:
            hz_emb = hz_emb * self.cfg.horizon_embed_scale
        hz_emb = hz_emb.view(1, horizon_count, 1, -1)
        var_ids = torch.arange(active_dim, device=device).view(1, 1, active_dim)
        var_emb = self.var_embed(var_ids).view(1, 1, active_dim, -1)
        token_type = self.token_type_embed(
            torch.full((1,), QUERY_TOKEN_TYPE, device=device, dtype=torch.long)
        ).view(1, 1, 1, -1)
        tokens = (var_emb + hz_emb + token_type).expand(
            1, horizon_count, active_dim, -1
        )
        time_emb = self.time_embed(horizons + (origin_index + 1)).view(
            1, horizon_count, 1, -1
        )
        tokens = tokens.reshape(1, horizon_count * active_dim, self.cfg.d_model)
        return tokens + time_emb.expand(1, horizon_count, active_dim, -1).reshape(
            1, horizon_count * active_dim, self.cfg.d_model
        )

    def _pool_horizons(self, column_tokens: Tensor) -> Tensor:
        """Pool ``(B, H, D_active, d_model)`` column tokens to ``(B, H, d_model)``."""
        batch_size, horizon_count, active_dim, hidden = column_tokens.shape
        pooled = self.horizon_pool_set(
            column_tokens.reshape(batch_size * horizon_count, active_dim, hidden)
        ).mean(dim=1)
        return pooled.reshape(batch_size, horizon_count, hidden)

    def forward(
        self,
        past: Tensor,
        horizons: Tensor,
        past_time_indices: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Return the head's parameters for every queried horizon.

        Args:
            past: ``(B, T, D_active)`` history, ``1 <= D_active <= capacity``.
            horizons: ``(H,)`` zero-based forecast step indices.
            past_time_indices: ``(T,)`` grid positions of the history rows;
                ``None`` means ``0 .. T-1``.

        Returns:
            The head's parameters (means and covariances in data space) plus
            the standardization center and scale under
            ``AUTO_STANDARDIZATION_CENTER_KEY`` and ``..._SCALE_KEY``.
        """
        batch_size, T_ctx, active_dim = past.shape
        if not 1 <= active_dim <= self.D:
            raise ValueError(
                f"the history carries {active_dim} columns; the capacity is {self.D}"
            )
        if past_time_indices is None:
            past_time_indices = torch.arange(T_ctx, device=past.device)
        if past_time_indices.dim() != 1 or past_time_indices.shape[0] != T_ctx:
            raise ValueError("past_time_indices must have shape (T,)")
        past_time_indices = past_time_indices.to(past.device)

        center = past[:, -1:, :]
        scale = torch.clamp(
            past.std(dim=1, keepdim=True, unbiased=False), min=MIN_STANDARDIZATION_SCALE
        )
        history = self._history_tokens((past - center) / scale, past_time_indices)
        horizon_count = horizons.shape[0]
        queries = self._query_tokens(horizons, past_time_indices[-1], active_dim)
        encoded = self.encoder(
            torch.cat([history, queries.expand(batch_size, -1, -1)], dim=1),
            T_ctx=T_ctx,
            D=active_dim,
            query_rows=horizon_count,
        )
        column_tokens = encoded[:, history.shape[1] :, :].reshape(
            batch_size, horizon_count, active_dim, self.cfg.d_model
        )
        params = self.head(
            self._pool_horizons(column_tokens), column_tokens=column_tokens
        )

        out = dict(params)
        out["means"] = (
            center.unsqueeze(2) + scale.unsqueeze(2) * params["means"][..., :active_dim]
        )
        out["lowrank_diag"] = (
            params["lowrank_diag"][..., :active_dim] * scale.unsqueeze(2) ** 2
        )
        out["lowrank_U"] = params["lowrank_U"][..., :active_dim, :] * scale.unsqueeze(
            2
        ).unsqueeze(-1)
        out[AUTO_STANDARDIZATION_CENTER_KEY] = center
        out[AUTO_STANDARDIZATION_SCALE_KEY] = scale
        return out


__all__ = ["JointFM", "SinusoidalPosition"]
