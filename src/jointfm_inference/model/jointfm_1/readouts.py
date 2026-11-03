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

"""Per-column readouts of the Student-t head's locations and covariance factor."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Softplus(x) overflows float32 at x ~ 89; capping keeps covariance diagonals,
# degrees of freedom, and downstream samples finite.
MAX_SOFTPLUS = 1e4

# Positions of the per-column location terms along the readout's output axis.
MEAN_OUTPUT = 0
RAW_DIAG_OUTPUT = 1
# One mean term and one raw covariance-diagonal term per column and component.
LOCATION_OUTPUTS = 2


class ColumnFactorReadout(nn.Module):
    """Emit each column's rows of the covariance factor from its own token.

    ``forward(column_tokens, context)`` takes the per-column horizon tokens
    ``(B, H, D_active, column_dim)`` and the pooled per-horizon context
    ``(B, H, context_dim)`` and returns ``(B, H, K, D_active, rank)``. The
    readout is shared across columns, so it is equivariant under column
    permutation and works for any ``D_active``: identical columns get identical
    factor rows, and the context concatenated to every token gives the columns
    of one sample a shared factor basis.
    """

    def __init__(
        self,
        *,
        column_dim: int,
        context_dim: int,
        hidden: int,
        n_components: int,
        rank: int,
    ) -> None:
        """Build the shared per-column MLP."""
        super().__init__()
        self.column_dim = int(column_dim)
        self.context_dim = int(context_dim)
        self.n_components = int(n_components)
        self.rank = int(rank)
        self.input = nn.Linear(self.column_dim + self.context_dim, int(hidden))
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(int(hidden))
        self.output = nn.Linear(int(hidden), self.n_components * self.rank, bias=False)

    def forward(self, column_tokens: Tensor, context: Tensor) -> Tensor:
        """Return the factor rows as ``(B, H, K, D_active, rank)``.

        Raises:
            ValueError: when the token or context shapes do not match the
                construction-time widths or each other.
        """
        if column_tokens.dim() != 4 or column_tokens.shape[-1] != self.column_dim:
            raise ValueError(
                f"column_tokens must have shape (B, H, D_active, {self.column_dim}), "
                f"got {tuple(column_tokens.shape)}"
            )
        batch_size, horizon_count, active_dim, _ = column_tokens.shape
        if tuple(context.shape) != (batch_size, horizon_count, self.context_dim):
            raise ValueError(
                f"context must have shape {(batch_size, horizon_count, self.context_dim)}, "
                f"got {tuple(context.shape)}"
            )
        context_per_column = context.unsqueeze(2).expand(-1, -1, active_dim, -1)
        features = torch.cat(
            [column_tokens, context_per_column.to(dtype=column_tokens.dtype)], dim=-1
        )
        hidden = self.norm(self.activation(self.input(features)))
        rows = self.output(hidden).reshape(
            batch_size, horizon_count, active_dim, self.n_components, self.rank
        )
        return rows.permute(0, 1, 3, 2, 4)


def _pad_columns(rows: Tensor, capacity: int) -> Tensor:
    """Zero-pad the column axis (``-2``) of ``rows`` to ``capacity``.

    Raises:
        ValueError: when ``rows`` already carries more columns than ``capacity``.
    """
    active_dim = rows.shape[-2]
    if active_dim > capacity:
        raise ValueError(
            f"column_tokens carry {active_dim} columns, exceeding the head capacity "
            f"D={capacity}"
        )
    if active_dim == capacity:
        return rows
    padding = rows.new_zeros(*rows.shape[:-2], capacity - active_dim, rows.shape[-1])
    return torch.cat([rows, padding], dim=-2)


class ColumnLocationReadout(nn.Module):
    """Emit the column-specific part of each column's location from its own token.

    ``forward(column_tokens, context, capacity)`` takes the per-column horizon
    tokens ``(B, H, D_active, column_dim)`` and the pooled per-horizon context
    ``(B, H, context_dim)`` and returns ``(B, H, K, capacity,
    LOCATION_OUTPUTS)``: per column and component, the term added to the mean
    base shared by all columns (:data:`MEAN_OUTPUT`) and the term added to the
    shared raw covariance-diagonal base before its softplus
    (:data:`RAW_DIAG_OUTPUT`); absent columns get zero terms. The one MLP is
    applied to every column's own token, so permuting the columns permutes the
    terms, as :class:`ColumnFactorReadout` does for the factor rows.
    """

    def __init__(
        self, *, column_dim: int, context_dim: int, hidden: int, n_components: int
    ) -> None:
        """Build the shared per-column MLP."""
        super().__init__()
        self.readout = ColumnFactorReadout(
            column_dim=column_dim,
            context_dim=context_dim,
            hidden=hidden,
            n_components=n_components,
            rank=LOCATION_OUTPUTS,
        )

    def forward(self, column_tokens: Tensor, context: Tensor, capacity: int) -> Tensor:
        """Return the per-column terms as ``(B, H, K, capacity, LOCATION_OUTPUTS)``."""
        return _pad_columns(self.readout(column_tokens, context), capacity)


def emit_lowrank_covariance(
    *,
    raw_diag: Tensor,
    column_factor: ColumnFactorReadout,
    hidden: Tensor,
    column_tokens: Tensor,
    capacity: int,
    min_scale: float,
    dtype: torch.dtype,
) -> Dict[str, Tensor]:
    """Emit ``lowrank_diag`` and ``lowrank_U`` of ``Sigma = diag(d) + U U^T``.

    ``d`` is the softplus of ``raw_diag`` ``(B, H, K, D)``, capped at
    ``MAX_SOFTPLUS`` and floored additively by ``min_scale`` (variance units);
    ``U`` ``(B, H, K, D, rank)`` holds the readout's rows of the present
    columns and zero rows for absent ones.
    """
    batch_size, horizon_count, _ = hidden.shape
    expected = (batch_size, horizon_count, column_factor.n_components, capacity)
    if tuple(raw_diag.shape) != expected:
        raise ValueError(
            f"raw_diag must have shape {expected}, got {tuple(raw_diag.shape)}"
        )
    floored = torch.clamp(F.softplus(raw_diag), max=MAX_SOFTPLUS) + min_scale
    factor = _pad_columns(column_factor(column_tokens, hidden), capacity)
    return {
        "lowrank_diag": floored.to(dtype=dtype),
        "lowrank_U": factor.to(dtype=dtype),
    }


__all__ = [
    "LOCATION_OUTPUTS",
    "MAX_SOFTPLUS",
    "MEAN_OUTPUT",
    "RAW_DIAG_OUTPUT",
    "ColumnFactorReadout",
    "ColumnLocationReadout",
    "emit_lowrank_covariance",
]
