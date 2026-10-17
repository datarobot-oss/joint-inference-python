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

"""Checkpoint loading through the backend registered for its generation."""

from __future__ import annotations

from typing import Any, Optional, Union

import torch

from jointfm_inference.model.checkpoint import (
    read_checkpoint_file,
    require_checkpoint_format,
)

from .backend_registry import resolve_serving_backend


def load_inference_components(
    checkpoint_path: str,
    *,
    map_location: Union[str, torch.device] = "cpu",
    device: Optional[Union[str, torch.device]] = None,
) -> tuple[str, torch.nn.Module, Any, dict[str, Any]]:
    """Load one checkpoint file through the backend of its generation.

    Returns ``(generation, model, model_cfg, checkpoint)``.

    Raises:
        ValueError, TypeError: when the file does not follow the runner's
            checkpoint format, before any backend is resolved.
        UnregisteredGenerationError: when no serving backend is registered
            for the checkpoint's generation, before any model is built.
    """
    checkpoint = read_checkpoint_file(checkpoint_path, map_location=map_location)
    generation = require_checkpoint_format(checkpoint)
    model, model_cfg = resolve_serving_backend(generation).load(
        checkpoint, device=torch.device(device) if device is not None else None
    )
    return generation, model, model_cfg, checkpoint


__all__ = ["load_inference_components"]
