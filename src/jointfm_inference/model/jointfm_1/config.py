"""Configuration of the JointFM 1 network.

The network has one fixed architecture: a factored time/row attention encoder
over a ``(time, column)`` token grid in which every forecast horizon of every
column is a query row on the time axis, a set-transformer pooling of each
horizon's column tokens, and a Student-t mixture head with a low-rank
covariance per component, decoding every queried horizon in one pass.
``JointFMConfig`` carries only the sizes that vary between checkpoints of that
architecture; a checkpoint stores it under ``cfg`` as a flat mapping.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Optional


@dataclass(frozen=True)
class JointFMConfig:
    """Sizes of one JointFM 1 network.

    Attributes:
        num_layers: Factored encoder layers.
        column_capacity: Most columns one request may carry; the size of the
            column embedding table and of every per-column head output.
        n_mixtures: Student-t mixture components per horizon.
        covariance_rank: Rank of each component's low-rank covariance factor.
        d_value: Width of the scalar value embedding.
        d_model: Token width of the encoder.
        nhead: Attention heads of every encoder block.
        dim_feedforward: Hidden width of the encoder's feed-forward blocks.
        layer_norm_eps: Epsilon of every layer norm.
        max_time_ctx: Longest history, in grid steps.
        max_time_fut: Furthest forecast horizon, in grid steps.
        attention_time_blocks: Time-attention blocks per encoder layer.
        attention_row_blocks: Row (cross-column) attention blocks per layer.
        attention_cnn_kernel_size: Temporal kernel of the convolutional
            feed-forward blocks.
        attention_cnn_first_layers: Leading encoder layers whose feed-forward
            block is convolutional; ``0`` means every layer.
        horizon_embed_scale: Multiplier of the horizon embedding of query tokens.
        time_freq_base: Base frequency of the sinusoidal time embedding.
        horizon_freq_base: Base frequency of the sinusoidal horizon embedding.
        head_trunk_layers: MLP layers of the head trunk.
        head_trunk_hidden: Hidden width of the head trunk; ``None`` uses
            ``d_model``.
        head_min_df: Lower bound of every component's degrees of freedom.
        head_min_scale: Additive floor of the covariance diagonal.
        pooling_nhead: Attention heads of the set-transformer pooling.
        pooling_seeds: Learned seed vectors of the set-transformer pooling.
        pooling_layers: Layers of the set-transformer pooling.
    """

    num_layers: int
    column_capacity: int
    n_mixtures: int
    covariance_rank: int
    d_value: int
    d_model: int
    nhead: int
    dim_feedforward: int
    layer_norm_eps: float
    max_time_ctx: int
    max_time_fut: int
    attention_time_blocks: int
    attention_row_blocks: int
    attention_cnn_kernel_size: int
    attention_cnn_first_layers: int
    horizon_embed_scale: float
    time_freq_base: float
    horizon_freq_base: float
    head_trunk_layers: int
    head_trunk_hidden: Optional[int]
    head_min_df: float
    head_min_scale: float
    pooling_nhead: int
    pooling_seeds: int
    pooling_layers: int

    def __post_init__(self) -> None:
        """Refuse sizes the network cannot be built with."""
        positive = (
            "num_layers",
            "column_capacity",
            "n_mixtures",
            "covariance_rank",
            "d_value",
            "d_model",
            "nhead",
            "dim_feedforward",
            "max_time_ctx",
            "max_time_fut",
            "head_trunk_layers",
            "pooling_nhead",
            "pooling_seeds",
            "pooling_layers",
        )
        for name in positive:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.attention_time_blocks < 0 or self.attention_row_blocks < 0:
            raise ValueError(
                "attention_time_blocks and attention_row_blocks must be >= 0"
            )
        if self.attention_time_blocks + self.attention_row_blocks == 0:
            raise ValueError(
                "at least one of attention_time_blocks and attention_row_blocks must be > 0"
            )
        for name, heads in (
            ("nhead", self.nhead),
            ("pooling_nhead", self.pooling_nhead),
        ):
            if self.d_model % heads != 0:
                raise ValueError(
                    f"d_model={self.d_model} must be divisible by {name}={heads}"
                )
        if self.d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {self.d_model}")
        if self.covariance_rank > self.column_capacity:
            raise ValueError(
                f"covariance_rank must not exceed column_capacity, got "
                f"{self.covariance_rank} > {self.column_capacity}"
            )
        if self.attention_cnn_first_layers < 0:
            raise ValueError(
                f"attention_cnn_first_layers must be >= 0, got {self.attention_cnn_first_layers}"
            )
        if (
            self.attention_cnn_kernel_size <= 0
            or self.attention_cnn_kernel_size % 2 == 0
        ):
            raise ValueError(
                "attention_cnn_kernel_size must be a positive odd integer, got "
                f"{self.attention_cnn_kernel_size}"
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "JointFMConfig":
        """Build the config from a checkpoint's ``cfg`` mapping.

        Raises:
            ValueError: when a field is missing or the mapping carries a key
                that is not a field, so a checkpoint of another architecture
                fails here instead of loading into the wrong network.
        """
        names = [field.name for field in fields(cls)]
        missing = sorted(set(names) - set(raw))
        unknown = sorted(set(raw) - set(names))
        if missing or unknown:
            raise ValueError(
                "checkpoint 'cfg' does not describe a JointFM 1 network: "
                f"missing {missing}, unknown {unknown}"
            )
        return cls(**{name: raw[name] for name in names})


__all__ = ["JointFMConfig"]
