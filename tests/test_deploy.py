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

import io
import json
from pathlib import Path
from urllib.request import Request

import pytest

from deploy import stack


def _env(checkpoint: Path, **overrides: str) -> dict[str, str]:
    env = {
        "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2",
        "DATAROBOT_API_TOKEN": "token",
        "JOINTFM_INFERENCE_CHECKPOINT_PATH": str(checkpoint),
    }
    env.update(overrides)
    return env


GIB = 1024**3
CPU_BUNDLE = stack.ResourceBundle(
    id="cpu.5xlarge",
    name="5XL",
    cpu=2.0,
    memory_bytes=8 * GIB,
    gpu=0,
    gpu_type=None,
    gpu_memory_bytes=0,
)
GPU_BUNDLE = stack.ResourceBundle(
    id="gpu.a10g",
    name="GPU - L",
    cpu=7.0,
    memory_bytes=26 * GIB,
    gpu=1,
    gpu_type="nvidia-a10g-2x",
    gpu_memory_bytes=24 * GIB,
)
BUNDLES = (CPU_BUNDLE, GPU_BUNDLE)


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
            JOINTFM_INFERENCE_CHECKPOINT_VERSION="dev",
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
    assert settings.bundle == "cpu.5xlarge"
    assert (settings.cpu, settings.memory, settings.gpu, settings.replicas) == (
        None,
        None,
        None,
        1,
    )


def test_settings_read_the_bundle_and_its_overrides(checkpoint: Path) -> None:
    settings = stack.read_settings(
        _env(
            checkpoint,
            JOINTFM_DEPLOY_BUNDLE="gpu.a10g",
            JOINTFM_DEPLOY_CPU="4",
            JOINTFM_DEPLOY_MEMORY="16GB",
            JOINTFM_DEPLOY_GPU="1",
            JOINTFM_DEPLOY_REPLICAS="2",
        )
    )

    assert settings.bundle == "gpu.a10g"
    assert (settings.cpu, settings.memory, settings.gpu, settings.replicas) == (
        4.0,
        "16GB",
        1.0,
        2,
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1024", 1024),
        ("8GB", 8 * 1000**3),
        ("4096Mi", 4096 * 1024**2),
        ("2 GiB", 2 * GIB),
    ],
)
def test_memory_sizes_follow_the_workload_api_suffixes(
    value: str, expected: int
) -> None:
    assert stack.memory_bytes(value) == expected


def test_fit_keeps_a_cpu_bundle_on_the_configured_device(checkpoint: Path) -> None:
    settings = stack.read_settings(_env(checkpoint))

    assert stack.fit_to_bundle(settings, BUNDLES) == {}


def test_fit_runs_a_gpu_bundle_on_cuda(checkpoint: Path) -> None:
    """A GPU node with the default cpu device would serve slowly and silently."""
    settings = stack.read_settings(_env(checkpoint, JOINTFM_DEPLOY_BUNDLE="gpu.a10g"))

    assert stack.fit_to_bundle(settings, BUNDLES) == {
        "JOINTFM_INFERENCE_DEVICE": "cuda"
    }


def test_fit_keeps_a_named_cuda_device(checkpoint: Path) -> None:
    settings = stack.read_settings(
        _env(
            checkpoint,
            JOINTFM_DEPLOY_BUNDLE="gpu.a10g",
            JOINTFM_INFERENCE_DEVICE="cuda:0",
        )
    )

    assert stack.fit_to_bundle(settings, BUNDLES) == {
        "JOINTFM_INFERENCE_DEVICE": "cuda:0"
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"JOINTFM_DEPLOY_BUNDLE": "gpu.h100"}, "not a Workload bundle"),
        ({"JOINTFM_DEPLOY_CPU": "3"}, "exceeds the 2 CPU"),
        ({"JOINTFM_DEPLOY_MEMORY": "9GiB"}, "exceeds the 8589934592 bytes"),
        ({"JOINTFM_DEPLOY_GPU": "1"}, "exceeds the 0 GPU"),
        ({"JOINTFM_INFERENCE_DEVICE": "cuda"}, "has no GPU"),
        (
            {"JOINTFM_DEPLOY_BUNDLE": "gpu.a10g", "JOINTFM_INFERENCE_DEVICE": "cpu"},
            "has a GPU",
        ),
    ],
)
def test_fit_refuses_what_the_bundle_cannot_hold(
    checkpoint: Path, overrides: dict[str, str], message: str
) -> None:
    settings = stack.read_settings(_env(checkpoint, **overrides))

    with pytest.raises(ValueError, match=message):
        stack.fit_to_bundle(settings, BUNDLES)


def test_bundles_are_read_from_every_catalog_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gpu_item = {
        "id": "gpu.a10g",
        "name": "GPU - L",
        "cpuCount": 7,
        "memoryBytes": 26 * GIB,
        "gpuCount": 1,
        "gpuTypeLabel": "nvidia-a10g-2x",
        "gpuMemoryBytes": 24 * GIB,
    }
    cpu_item = {
        "id": "cpu.5xlarge",
        "name": "5XL",
        "cpuCount": 2,
        "memoryBytes": 8 * GIB,
    }
    pages = {
        "https://dr.example/api/v2/mlops/compute/bundles/?useCases=workload": {
            "data": [cpu_item],
            "next": "https://dr.example/api/v2/mlops/compute/bundles/?page=2",
        },
        "https://dr.example/api/v2/mlops/compute/bundles/?page=2": {
            "data": [gpu_item],
            "next": None,
        },
    }
    seen: list[tuple[str, str | None]] = []

    def _urlopen(request: Request, timeout: float) -> io.BytesIO:
        seen.append((request.full_url, request.get_header("Authorization")))
        return io.BytesIO(json.dumps(pages[request.full_url]).encode())

    monkeypatch.setattr(stack, "urlopen", _urlopen)

    bundles = stack.fetch_workload_bundles("https://dr.example/api/v2/", "token")

    assert bundles == [CPU_BUNDLE, GPU_BUNDLE]
    assert [header for _, header in seen] == ["Bearer token", "Bearer token"]


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
        ({"JOINTFM_DEPLOY_MEMORY": "8 apples"}, "must be a size"),
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
