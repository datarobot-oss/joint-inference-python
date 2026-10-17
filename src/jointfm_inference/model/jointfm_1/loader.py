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

"""Rebuild a JointFM 1 network from a checkpoint mapping.

A JointFM 1 checkpoint follows the runner's checkpoint format
(:mod:`jointfm_inference.model.checkpoint`) with generation ``jointfm-1`` and
stores the :class:`JointFMConfig` fields as a flat ``cfg`` mapping.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import torch

from jointfm_inference.model.checkpoint import (
    CHECKPOINT_CFG_KEY,
    CHECKPOINT_MODEL_STATE_KEY,
    JOINTFM_1_GENERATION,
    require_checkpoint_format,
)
from jointfm_inference.model.jointfm_1.config import JointFMConfig
from jointfm_inference.model.jointfm_1.model import JointFM


def load_model_from_checkpoint(
    checkpoint: Mapping[str, Any], *, device: Optional[torch.device] = None
) -> tuple[JointFM, JointFMConfig]:
    """Return the network rebuilt from ``checkpoint``, in evaluation mode.

    Every parameter must match the checkpoint exactly (``strict=True``);
    weights stored in a reduced dtype are upcast into the float32 parameters.

    Raises:
        ValueError: when the checkpoint does not follow the runner's checkpoint
            format (see :func:`require_checkpoint_format`), names another
            generation, or its ``cfg`` does not describe a JointFM 1 network.
        TypeError: when a checkpoint entry has the wrong type.
        RuntimeError: when the state dict does not match the network.
    """
    generation = require_checkpoint_format(checkpoint)
    if generation != JOINTFM_1_GENERATION:
        raise ValueError(
            f"the JointFM 1 loader requires generation '{JOINTFM_1_GENERATION}', "
            f"got '{generation}'"
        )
    cfg = JointFMConfig.from_mapping(checkpoint[CHECKPOINT_CFG_KEY])
    model = JointFM(cfg)
    model.load_state_dict(checkpoint[CHECKPOINT_MODEL_STATE_KEY], strict=True)
    if device is not None:
        model.to(device)
    model.eval()
    return model, cfg


__all__ = ["load_model_from_checkpoint"]
