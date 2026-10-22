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

"""The DataRobot deploy reads its settings from .env and stages a buildable context."""

from __future__ import annotations

from pathlib import Path

import pytest

from deploy import stack


def _env(checkpoint: Path, **overrides: str) -> dict[str, str]:
    env = {
        "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2",
        "DATAROBOT_API_TOKEN": "token",
        "JOINTFM_INFERENCE_CHECKPOINT_PATH": str(checkpoint),
        "JOINTFM_INFERENCE_CHECKPOINT_VERSION": "dev",
    }
    env.update(overrides)
    return env


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "serving.pt"
    path.write_bytes(b"weights")
    return path


def test_settings_forward_only_the_service_settings_that_are_set(
    checkpoint: Path,
) -> None:
    settings = stack.read_settings(
        _env(
            checkpoint,
            JOINTFM_INFERENCE_DEVICE="cpu",
            JOINTFM_INFERENCE_PRECISION="",
            JOINTFM_INFERENCE_CHECKPOINT_URL="https://example.com/serving.pt",
            JOINTFM_INFERENCE_PORT="9000",
        )
    )

    assert settings.checkpoint_path == checkpoint.resolve()
    assert settings.service_env == {
        "JOINTFM_INFERENCE_CHECKPOINT_VERSION": "dev",
        "JOINTFM_INFERENCE_DEVICE": "cpu",
    }
    assert (settings.cpu, settings.memory, settings.gpu, settings.replicas) == (
        2.0,
        "8GB",
        None,
        1,
    )


def test_settings_read_the_workload_resources(checkpoint: Path) -> None:
    settings = stack.read_settings(
        _env(
            checkpoint,
            JOINTFM_DEPLOY_CPU="4",
            JOINTFM_DEPLOY_MEMORY="16GB",
            JOINTFM_DEPLOY_GPU="1",
            JOINTFM_DEPLOY_REPLICAS="2",
        )
    )

    assert (settings.cpu, settings.memory, settings.gpu, settings.replicas) == (
        4.0,
        "16GB",
        1.0,
        2,
    )


def test_a_relative_checkpoint_path_is_taken_from_the_repository_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pulumi runs the program from deploy/, task serve from the root.
    monkeypatch.chdir(tmp_path)

    settings = stack.read_settings(_env(Path("LICENSE")))

    assert settings.checkpoint_path == stack.REPO_ROOT / "LICENSE"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"DATAROBOT_API_TOKEN": ""}, "DATAROBOT_API_TOKEN must be set"),
        ({"DATAROBOT_ENDPOINT": "https://app.datarobot.com"}, "must end with /api/v2"),
        (
            {"JOINTFM_INFERENCE_CHECKPOINT_VERSION": ""},
            "CHECKPOINT_VERSION must be set",
        ),
        ({"JOINTFM_DEPLOY_GPU": "0"}, "JOINTFM_DEPLOY_GPU must be positive"),
        (
            {"JOINTFM_DEPLOY_REPLICAS": "0"},
            "JOINTFM_DEPLOY_REPLICAS must be at least 1",
        ),
    ],
)
def test_settings_refuse_missing_or_invalid_values(
    checkpoint: Path, overrides: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        stack.read_settings(_env(checkpoint, **overrides))


def test_settings_refuse_a_checkpoint_that_is_not_a_local_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="names no file"):
        stack.read_settings(_env(tmp_path / "missing.pt"))


def test_the_staged_context_builds_the_runner_with_the_checkpoint_baked_in(
    checkpoint: Path, tmp_path: Path
) -> None:
    context = tmp_path / "context"
    (context / ".datarobot" / "workload").mkdir(parents=True)
    (context / ".datarobot" / "workload" / "state.json").write_text("{}")
    (context / "stale.txt").write_text("left over from an earlier run")

    stack.stage_build_context(checkpoint, context)

    assert (context / ".datarobot" / "workload" / "state.json").read_text() == "{}"
    assert not (context / "stale.txt").exists()
    assert (context / "checkpoint" / "jointfm.pt").read_bytes() == b"weights"
    for name in stack.RUNNER_SOURCES:
        assert (context / name).exists()
    assert not list((context / "src").rglob("__pycache__"))
    assert (context / ".drignore").is_file()

    dockerfile = (context / "Dockerfile").read_text()
    stages = [line for line in dockerfile.splitlines() if line.startswith("FROM ")]
    assert any(line.endswith(" AS runtime") for line in stages)
    assert stages[-1] == "FROM runtime AS bundled"
    assert (
        "ENV JOINTFM_INFERENCE_CHECKPOINT_PATH=/opt/jointfm-inference/checkpoint/jointfm.pt"
        in dockerfile.splitlines()
    )
