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

"""The sign-symmetric predictive distribution of a JointFM 1 network.

A checkpoint whose ``cfg`` turns ``sign_symmetric_prediction`` on predicts
with the equal-weight mixture of the network's forecast for the history ``x``
and the negated forecast for ``-x``. That mixture is exactly invariant under
negating every column at once: the log-density of ``y`` after ``x`` equals
that of ``-y`` after ``-x``. A lean of the network that does not flip with its
input cancels, while the part of the forecast that follows the input is kept.

The Student-t mixture head is closed under the construction. A negated
component has the negated location and the same covariance, degrees of
freedom, and weight, so the mixture is one head parameter mapping with twice
the components, and every consumer of head parameters (sampling, scoring,
conditioning) reads the component count from ``logits`` and accepts it
unchanged.

The network's input standardization is sign-equivariant: ``-x`` has the
negated center and the same scale, so the mirrored forward sees exactly the
negated standardized history.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor

from jointfm_inference.model.jointfm_1.head import AUTO_STANDARDIZATION_METADATA_KEYS

LOGITS_KEY: Final = "logits"
NEGATED_PARAM_KEYS: Final = frozenset({"means"})
"""Head parameters that flip sign with the component: the location."""
SHARED_PARAM_KEYS: Final = frozenset({"df", "lowrank_diag", "lowrank_U"})
"""Head parameters a component keeps under negation: tails and covariance."""


def symmetric_mixture(
    direct: dict[str, Tensor], mirrored: dict[str, Tensor]
) -> dict[str, Tensor]:
    """Return the equal-weight mixture of ``direct`` and the negation of ``mirrored``.

    Args:
        direct: Forward output for the history ``x``: the head parameters,
            each with the component axis where ``logits`` has its last axis,
            plus the standardization center and scale of ``x``.
        mirrored: Forward output of the same network for ``-x``.

    Returns:
        Head parameters with twice the components, the direct ones first and
        then the mirrored ones negated, and the standardization center and
        scale of ``direct``. Each half's logits are normalized first, so
        either half carries weight exactly one half.

    Raises:
        ValueError: when the two outputs carry different keys or a key whose
            behavior under negation is unknown.
    """
    if direct.keys() != mirrored.keys():
        raise ValueError(
            f"direct and mirrored forward outputs differ: {sorted(direct)} vs "
            f"{sorted(mirrored)}"
        )
    unknown = (
        set(direct)
        - NEGATED_PARAM_KEYS
        - SHARED_PARAM_KEYS
        - {LOGITS_KEY}
        - set(AUTO_STANDARDIZATION_METADATA_KEYS)
    )
    if unknown:
        raise ValueError(
            f"forward output keys {sorted(unknown)} have no sign-symmetric form"
        )
    component_axis = direct[LOGITS_KEY].ndim - 1
    out: dict[str, Tensor] = {}
    for key, value in direct.items():
        if key in AUTO_STANDARDIZATION_METADATA_KEYS:
            out[key] = value
            continue
        other = mirrored[key]
        if key == LOGITS_KEY:
            value = torch.log_softmax(value, dim=-1)
            other = torch.log_softmax(other, dim=-1)
        elif key in NEGATED_PARAM_KEYS:
            other = -other
        out[key] = torch.cat([value, other], dim=component_axis)
    return out


__all__ = [
    "NEGATED_PARAM_KEYS",
    "SHARED_PARAM_KEYS",
    "symmetric_mixture",
]
