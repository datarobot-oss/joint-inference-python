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

"""Configuration of the JointFM 1 network.

The network has one fixed architecture: a factored time/row attention encoder
over a ``(time, column)`` token grid in which every forecast horizon of every
column is a query row on the time axis, a set-transformer pooling of each
horizon's column tokens, and a Student-t mixture head with a low-rank
covariance per component, decoding every queried horizon in one pass.
``JointFMConfig`` carries the sizes that vary between checkpoints of that
architecture, whether the checkpoint predicts with its sign-symmetric
mixture, and the bounds of its per-column tail map when the head has one; a
checkpoint stores it under ``cfg`` as a flat mapping.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, dataclass, fields
from typing import Any, Optional

from jointfm_inference.model.jointfm_1.tail_map import (
    TAIL_PARAMETER_CEILING,
    TAIL_PARAMETER_FLOOR,
)


@dataclass(frozen=True)
class JointFMConfig:
    """Sizes of one JointFM 1 network.

    Attributes:
        num_layers: Factored encoder layers.
        column_capacity: Most columns one request may carry; the size every
            per-column head output is padded to. It sizes no parameter.
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
        sign_symmetric_prediction: Predict with the sign-symmetric mixture:
            the equal-weight mixture of the forecast for the history and the
            negated forecast for the negated history, which negating every
            column of a request negates exactly
            (:mod:`jointfm_inference.model.jointfm_1.sign_symmetry`). Off
            predicts with the head's mixture as emitted; a ``cfg`` without
            the key is off.
        head_tail_parameter_min: Lower bound of every column's two tail
            parameters under the head's per-column tail map
            (:mod:`jointfm_inference.model.jointfm_1.tail_map`). Set together
            with ``head_tail_parameter_max``, it gives the head its tail
            readout and maps every column of the forecast; both unset (a
            ``cfg`` without the keys) is the head without a map, whose
            forecast is the Student-t mixture itself.
        head_tail_parameter_max: Upper bound of every column's two tail
            parameters; set exactly when ``head_tail_parameter_min`` is.
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
    sign_symmetric_prediction: bool = False
    head_tail_parameter_min: Optional[float] = None
    head_tail_parameter_max: Optional[float] = None

    @property
    def has_tail_map(self) -> bool:
        """Whether the head maps every column through its per-column tail map."""
        return self.head_tail_parameter_min is not None

    def __post_init__(self) -> None:
        """Refuse sizes the network cannot be built with, a non-bool switch, and bad tail bounds."""
        if not isinstance(self.sign_symmetric_prediction, bool):
            raise TypeError(
                "sign_symmetric_prediction must be a bool, got "
                f"{type(self.sign_symmetric_prediction).__name__}"
            )
        self._require_tail_bounds()
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

    def _require_tail_bounds(self) -> None:
        """Refuse tail parameter bounds the map cannot serve.

        Raises:
            ValueError: when only one bound is set; when the bounds leave
                ``TAIL_PARAMETER_FLOOR <= min < 1 < max <=
                TAIL_PARAMETER_CEILING``; when ``max`` is not below half of
                ``head_min_df``, which every component's degrees of freedom
                exceed, so a column's variance and mean could be infinite; or
                when the map is combined with ``sign_symmetric_prediction``,
                whose mixture this runner forms only for the head without a
                map.
        """
        minimum, maximum = self.head_tail_parameter_min, self.head_tail_parameter_max
        if (minimum is None) != (maximum is None):
            raise ValueError(
                "head_tail_parameter_min and head_tail_parameter_max must be set "
                f"together, got {minimum} and {maximum}"
            )
        if minimum is None or maximum is None:
            return
        if (
            not TAIL_PARAMETER_FLOOR
            <= minimum
            < 1.0
            < maximum
            <= TAIL_PARAMETER_CEILING
        ):
            raise ValueError(
                "tail parameter bounds must satisfy "
                f"{TAIL_PARAMETER_FLOOR} <= min < 1 < max <= {TAIL_PARAMETER_CEILING}, "
                f"got min {minimum} and max {maximum}"
            )
        if maximum >= self.head_min_df / 2.0:
            raise ValueError(
                f"head_tail_parameter_max {maximum} must be below head_min_df / 2 = "
                f"{self.head_min_df / 2.0}, so every column's variance is finite"
            )
        if self.sign_symmetric_prediction:
            raise ValueError(
                "sign_symmetric_prediction is not served together with a per-column "
                "tail map"
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "JointFMConfig":
        """Build the config from a checkpoint's ``cfg`` mapping.

        A field with a default (``sign_symmetric_prediction`` and the two
        tail parameter bounds) may be absent and then takes it.

        Raises:
            ValueError: when a field without a default is missing or the
                mapping carries a key that is not a field, so a checkpoint of
                another architecture fails here instead of loading into the
                wrong network.
        """
        names = [field.name for field in fields(cls)]
        required = [field.name for field in fields(cls) if field.default is MISSING]
        missing = sorted(set(required) - set(raw))
        unknown = sorted(set(raw) - set(names))
        if missing or unknown:
            raise ValueError(
                "checkpoint 'cfg' does not describe a JointFM 1 network: "
                f"missing {missing}, unknown {unknown}"
            )
        return cls(**{name: raw[name] for name in names if name in raw})


__all__ = ["JointFMConfig"]
