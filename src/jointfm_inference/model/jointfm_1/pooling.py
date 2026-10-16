"""Set-transformer pooling of each horizon's column tokens."""

from __future__ import annotations

from typing import Optional, cast

import torch
import torch.nn as nn
from torch import Tensor


class _PMA(nn.Module):
    """Pooling by multihead attention onto learned seeds (Lee et al., 2019, arXiv:1810.00825)."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_seeds: int,
        num_layers: int = 1,
        dim_feedforward: Optional[int] = None,
        layer_norm_eps: float = 1e-5,
    ):
        """Build ``num_layers`` attention-plus-MLP blocks over learned seeds."""
        super().__init__()
        self.num_layers = num_layers
        self.S = nn.Parameter(torch.randn(1, num_seeds, d_model))

        self.layers = nn.ModuleList()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "attn": nn.MultiheadAttention(d_model, nhead, batch_first=True),
                        "norm1": nn.LayerNorm(d_model, eps=layer_norm_eps),
                        "linear1": nn.Linear(d_model, dim_feedforward),
                        "linear2": nn.Linear(dim_feedforward, d_model),
                        "norm2": nn.LayerNorm(d_model, eps=layer_norm_eps),
                        "activation": nn.GELU(),
                    }
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        """Pool a set of inputs into a learned seed set."""
        batch_size = x.size(0)
        curr_S = self.S.expand(batch_size, -1, -1)

        for layer in self.layers:
            layer = cast(nn.ModuleDict, layer)
            attn = layer["attn"]
            norm1 = layer["norm1"]
            linear1 = layer["linear1"]
            linear2 = layer["linear2"]
            norm2 = layer["norm2"]
            activation = layer["activation"]

            attn_out, _ = attn(curr_S, x, x, need_weights=False)
            x2 = norm1(curr_S + attn_out)
            ff = linear2(activation(linear1(x2)))
            curr_S = norm2(x2 + ff)

        return curr_S


__all__ = ["_PMA"]
