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

"""The DataRobot CLI prompts stay in step with the documented .env settings."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = REPO_ROOT / ".datarobot" / "cli"
ENV_TEMPLATE = REPO_ROOT / ".env.template"
# A setting, set or commented out, at the start of a template line.
_TEMPLATE_SETTING = re.compile(r"^#?\s*([A-Z][A-Z0-9_]*)=", re.MULTILINE)


def _prompted_settings() -> set[str]:
    names: set[str] = set()
    for path in sorted(PROMPTS_DIR.glob("*.y*ml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        # versions.yaml maps tools to mappings; only prompt files map
        # sections to lists of prompts.
        if not all(isinstance(section, list) for section in document.values()):
            continue
        for prompts in document.values():
            names.update(prompt["env"] for prompt in prompts if "env" in prompt)
    return names


def test_the_repository_ships_prompts() -> None:
    assert "JOINTFM_INFERENCE_CHECKPOINT_PATH" in _prompted_settings()


def test_every_prompted_setting_is_documented_in_the_env_template() -> None:
    """The template may hold settings dr never asks for, but not the reverse."""
    documented = set(
        _TEMPLATE_SETTING.findall(ENV_TEMPLATE.read_text(encoding="utf-8"))
    )

    assert _prompted_settings() - documented == set()
