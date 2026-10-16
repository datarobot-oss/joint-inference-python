"""Factored time/row attention encoder of the JointFM 1 network.

The encoder reads a time-major token grid ``(B, (T_ctx + query_rows) * D, C)``:
``T_ctx`` history rows of ``D`` column tokens each, followed by ``query_rows``
forecast query rows. Each layer runs time-attention blocks per column, then
row-attention blocks across the columns of each row, then one feed-forward
block. A history row never attends to a query row and a query row attends to
no other query row, so each horizon's forecast is the same whichever other
horizons are queried with it.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _query_row_isolation_mask(
    T_ctx: int, query_rows: int, device: torch.device
) -> Optional[Tensor]:
    """Return the ``(rows, rows)`` time-attention mask; ``True`` blocks a pair.

    History rows may not read query rows, and a query row reads every history
    row and itself but no other query row. ``query_rows=0`` needs no mask.
    """
    if query_rows < 0:
        raise ValueError(f"query_rows must be >= 0, got {query_rows}")
    if query_rows == 0:
        return None
    rows = T_ctx + query_rows
    mask = torch.zeros(rows, rows, dtype=torch.bool, device=device)
    mask[:T_ctx, T_ctx:] = True
    mask[T_ctx:, T_ctx:] = ~torch.eye(query_rows, dtype=torch.bool, device=device)
    return mask


class _TimeAttentionBlock(nn.Module):
    """Self-attention along the time axis of each column.

    Expects the column-major grid ``(B, D, T, C)``, so ``view(B * D, T, C)``
    yields one sequence per column.
    """

    def __init__(self, d_model: int, nhead: int, layer_norm_eps: float):
        """Build the attention and its residual norm."""
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor]) -> Tensor:
        """Attend over time per column; ``attn_mask`` is ``(T, T)`` or ``None``."""
        B, D, T, C = x.shape
        x_reshaped = x.view(B * D, T, C)
        attn_out, _ = self.attn(
            x_reshaped, x_reshaped, x_reshaped, attn_mask=attn_mask, need_weights=False
        )
        x_reshaped = self.norm(x_reshaped + attn_out)
        return x_reshaped.view(B, D, T, C)


class _RowAttentionBlock(nn.Module):
    """Self-attention across the columns of each row of the ``(B, T, D, C)`` grid."""

    def __init__(self, d_model: int, nhead: int, layer_norm_eps: float):
        """Build the attention and its residual norm."""
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, x: Tensor) -> Tensor:
        """Attend over columns per row."""
        B, T, D, C = x.shape
        x_reshaped = x.view(B * T, D, C)
        attn_out, _ = self.attn(x_reshaped, x_reshaped, x_reshaped, need_weights=False)
        x_reshaped = self.norm(x_reshaped + attn_out)
        return x_reshaped.view(B, T, D, C)


class _PositionwiseFFN(nn.Module):
    """Token-wise feed-forward block over ``(B, N, C)``."""

    def __init__(self, d_model: int, dim_feedforward: int, layer_norm_eps: float):
        """Build the two-layer MLP and its residual norm."""
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the MLP with a residual connection."""
        return self.norm(x + self.ffn(x))


class _PositionwiseConvFFN(nn.Module):
    """Feed-forward block whose first layer convolves along each column's time axis.

    The kernel convolves the history rows only, zero-padded at both ends; query
    rows get the kernel's center tap, so the filter never reads across the
    forecast origin and a query row's neighbors never enter its value.
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        kernel_size: int,
        layer_norm_eps: float,
    ):
        """Build the convolutions and the residual norm."""
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.conv1 = nn.Conv1d(
            d_model, dim_feedforward, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(dim_feedforward, d_model, kernel_size=1, padding=0)
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, x: Tensor, T_ctx: int, D: int, query_rows: int) -> Tensor:
        """Apply the convolutional MLP to ``(B, (T_ctx + query_rows) * D, C)``."""
        B, N, C = x.shape
        rows = T_ctx + query_rows
        x_seq = (
            x.view(B, rows, D, C).permute(0, 2, 3, 1).contiguous().view(B * D, C, rows)
        )
        h = self.conv1(x_seq[..., :T_ctx])
        if query_rows > 0:
            center = self.conv1.kernel_size[0] // 2
            h_query = F.conv1d(
                x_seq[..., T_ctx:],
                self.conv1.weight[:, :, center : center + 1],
                self.conv1.bias,
            )
            h = torch.cat([h, h_query], dim=-1)
        h = self.conv2(self.act(h))
        h = h.view(B, D, C, rows).permute(0, 3, 1, 2).contiguous()
        return self.norm(x + h.view(B, N, C))


class FactoredEncoderLayer(nn.Module):
    """Time blocks, then row blocks, then one feed-forward block."""

    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        layer_norm_eps: float,
        time_blocks: int,
        row_blocks: int,
        cnn_kernel_size: Optional[int],
    ):
        """Build the blocks; ``cnn_kernel_size=None`` selects the MLP feed-forward."""
        super().__init__()
        self.time_blocks = nn.ModuleList(
            _TimeAttentionBlock(d_model, nhead, layer_norm_eps)
            for _ in range(time_blocks)
        )
        self.row_blocks = nn.ModuleList(
            _RowAttentionBlock(d_model, nhead, layer_norm_eps)
            for _ in range(row_blocks)
        )
        # Exactly one feed-forward block exists per layer; the two attribute
        # names keep the parameter names of each kind distinct.
        self.ffn_mlp: _PositionwiseFFN | None = None
        self.ffn_cnn: _PositionwiseConvFFN | None = None
        if cnn_kernel_size is not None:
            self.ffn_cnn = _PositionwiseConvFFN(
                d_model, dim_feedforward, cnn_kernel_size, layer_norm_eps
            )
        else:
            self.ffn_mlp = _PositionwiseFFN(d_model, dim_feedforward, layer_norm_eps)

    def forward(self, x: Tensor, T_ctx: int, D: int, query_rows: int) -> Tensor:
        """Encode one ``(B, (T_ctx + query_rows) * D, C)`` time-major grid."""
        B, N, C = x.shape
        rows = T_ctx + query_rows
        if N != rows * D:
            raise ValueError("N must equal (T_ctx + query_rows) * D")
        time_mask = _query_row_isolation_mask(T_ctx, query_rows, x.device)
        columns = x.view(B, rows, D, C).transpose(1, 2).contiguous()
        for block in self.time_blocks:
            columns = block(columns, time_mask)
        x_grid = columns.transpose(1, 2).contiguous()
        for block in self.row_blocks:
            x_grid = block(x_grid)
        x_flat = x_grid.view(B, N, C)
        if self.ffn_cnn is not None:
            return self.ffn_cnn(x_flat, T_ctx=T_ctx, D=D, query_rows=query_rows)
        assert self.ffn_mlp is not None
        return self.ffn_mlp(x_flat)


class FactoredEncoder(nn.Module):
    """Stack of factored encoder layers.

    ``cnn_first_layers`` leading layers use the convolutional feed-forward
    block; ``0`` means every layer.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        layer_norm_eps: float,
        time_blocks: int,
        row_blocks: int,
        cnn_kernel_size: int,
        cnn_first_layers: int,
    ):
        """Build ``num_layers`` layers."""
        super().__init__()
        self.layers = nn.ModuleList(
            FactoredEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                layer_norm_eps=layer_norm_eps,
                time_blocks=time_blocks,
                row_blocks=row_blocks,
                cnn_kernel_size=(
                    cnn_kernel_size
                    if cnn_first_layers == 0 or index < cnn_first_layers
                    else None
                ),
            )
            for index in range(num_layers)
        )

    def forward(self, x: Tensor, T_ctx: int, D: int, query_rows: int) -> Tensor:
        """Encode history and query rows through every layer."""
        for layer in self.layers:
            x = layer(x, T_ctx=T_ctx, D=D, query_rows=query_rows)
        return x


__all__ = ["FactoredEncoder", "FactoredEncoderLayer"]
