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

"""Deploy the inference runner to DataRobot as a Workload.

DataRobot builds the image itself from an uploaded build context (code to
workload), so nothing here needs docker or an image registry. The context is
the runner's own container build plus one extra stage that bakes the
checkpoint in. The spec sets no per-route auth; requests to the Workload
take a DataRobot API token.

Every setting comes from the environment, which the ``deploy:*`` Taskfile
targets load from the same ``.env`` that ``task serve`` reads.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pulumi
import pulumi_datarobot as datarobot

from jointfm_inference.model.service.config import (
    CHECKPOINT_PATH_ENV,
    CHECKPOINT_VERSION_ENV,
    DEVICE_ENV,
    MAP_LOCATION_ENV,
    MAX_CONCURRENT_REQUESTS_ENV,
    MAX_SAMPLE_COUNT_ENV,
    MODEL_VERSION_ENV,
    PRECISION_ENV,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DIR = REPO_ROOT / "deploy"
# The provider syncs this directory both ways and keeps its own state in it,
# so it is a dedicated staging copy, never the repository itself.
BUILD_CONTEXT_DIR = DEPLOY_DIR / "build" / "context"
PROVIDER_STATE_DIR_NAME = ".datarobot"

BASE_DOCKERFILE = REPO_ROOT / "container" / "Dockerfile"
BUNDLED_STAGE_DOCKERFILE = DEPLOY_DIR / "bundled.Dockerfile"
STAGED_CHECKPOINT = Path("checkpoint") / "jointfm.pt"
# What the base image's builder stage copies, staged at the same paths.
RUNNER_SOURCES = ("LICENSE", "README.md", "pyproject.toml", "uv.lock", "src")

CONTAINER_NAME = "jointfm-inference"
CONTAINER_PORT = 8080
HEALTH_ROUTE = "/healthz"
# Loading the checkpoint happens before the port opens, so startup gets a long
# window while readiness stays quick once the service answers.
STARTUP_PROBE_PERIOD_SECONDS = 10
STARTUP_PROBE_FAILURE_THRESHOLD = 60

DATAROBOT_ENDPOINT_ENV = "DATAROBOT_ENDPOINT"
DATAROBOT_API_TOKEN_ENV = "DATAROBOT_API_TOKEN"
DATAROBOT_API_PATH = "/api/v2"
DEPLOY_CPU_ENV = "JOINTFM_DEPLOY_CPU"
DEPLOY_MEMORY_ENV = "JOINTFM_DEPLOY_MEMORY"
DEPLOY_GPU_ENV = "JOINTFM_DEPLOY_GPU"
DEPLOY_REPLICAS_ENV = "JOINTFM_DEPLOY_REPLICAS"
DEFAULT_DEPLOY_CPU = "2"
DEFAULT_DEPLOY_MEMORY = "8GB"
DEFAULT_DEPLOY_REPLICAS = "1"

# Service settings passed through to the container when set. The checkpoint
# path, download settings, bind host, and port are fixed by the image instead.
FORWARDED_SERVICE_ENV = (
    CHECKPOINT_VERSION_ENV,
    MODEL_VERSION_ENV,
    DEVICE_ENV,
    MAP_LOCATION_ENV,
    PRECISION_ENV,
    MAX_SAMPLE_COUNT_ENV,
    MAX_CONCURRENT_REQUESTS_ENV,
)


@dataclass(frozen=True)
class DeploySettings:
    """Validated deployment settings read from the environment."""

    checkpoint_path: Path
    service_env: dict[str, str]
    cpu: float
    memory: str
    gpu: float | None
    replicas: int


def _required(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} must be set in .env to deploy")
    return value


def _optional(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name, "").strip()
    return value or None


def _positive_number(name: str, value: str) -> float:
    number = float(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return number


def read_settings(env: Mapping[str, str]) -> DeploySettings:
    """Return the deployment settings, failing on anything missing or invalid.

    The DataRobot credentials are checked here even though the provider reads
    them itself, so a missing one fails before any staging work. The
    checkpoint must be a file on this machine, since the build context carries
    it; a relative path is taken from the repository root.
    """
    endpoint = _required(env, DATAROBOT_ENDPOINT_ENV)
    if not endpoint.rstrip("/").endswith(DATAROBOT_API_PATH):
        raise ValueError(
            f"{DATAROBOT_ENDPOINT_ENV} must end with {DATAROBOT_API_PATH}, "
            f"got {endpoint!r}"
        )
    _required(env, DATAROBOT_API_TOKEN_ENV)

    # Pulumi runs this from deploy/, but task serve reads the same .env from
    # the repository root, so a relative path means the root in both.
    checkpoint_path = REPO_ROOT / Path(_required(env, CHECKPOINT_PATH_ENV)).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"{CHECKPOINT_PATH_ENV} names no file on this machine: {checkpoint_path}"
        )
    _required(env, CHECKPOINT_VERSION_ENV)
    service_env = {
        name: value
        for name in FORWARDED_SERVICE_ENV
        if (value := _optional(env, name)) is not None
    }

    gpu_value = _optional(env, DEPLOY_GPU_ENV)
    replicas = int(env.get(DEPLOY_REPLICAS_ENV) or DEFAULT_DEPLOY_REPLICAS)
    if replicas < 1:
        raise ValueError(f"{DEPLOY_REPLICAS_ENV} must be at least 1, got {replicas}")
    return DeploySettings(
        checkpoint_path=checkpoint_path.resolve(),
        service_env=service_env,
        cpu=_positive_number(
            DEPLOY_CPU_ENV, env.get(DEPLOY_CPU_ENV) or DEFAULT_DEPLOY_CPU
        ),
        memory=env.get(DEPLOY_MEMORY_ENV) or DEFAULT_DEPLOY_MEMORY,
        gpu=None if gpu_value is None else _positive_number(DEPLOY_GPU_ENV, gpu_value),
        replicas=replicas,
    )


def stage_build_context(checkpoint_path: Path, context_dir: Path) -> Path:
    """Write the build context DataRobot builds the image from, and return it.

    The context holds the runner's sources, a ``Dockerfile`` that is the base
    image's file followed by the stage that bakes the checkpoint in, and the
    checkpoint itself. Everything except the provider's own state directory
    is rewritten on every run, so a file deleted from the repository also
    leaves the context.
    """
    context_dir.mkdir(parents=True, exist_ok=True)
    for entry in context_dir.iterdir():
        if entry.name == PROVIDER_STATE_DIR_NAME:
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()

    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for name in RUNNER_SOURCES:
        source = REPO_ROOT / name
        if source.is_dir():
            shutil.copytree(source, context_dir / name, ignore=ignore)
        else:
            shutil.copy2(source, context_dir / name)
    (context_dir / "Dockerfile").write_text(
        BASE_DOCKERFILE.read_text(encoding="utf-8")
        + "\n"
        + BUNDLED_STAGE_DOCKERFILE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    staged_checkpoint = context_dir / STAGED_CHECKPOINT
    staged_checkpoint.parent.mkdir()
    shutil.copy2(checkpoint_path, staged_checkpoint)
    # An existing ignore file stops the provider from writing its default one,
    # whose patterns could drop the checkpoint from the upload.
    (context_dir / ".drignore").write_text(
        "# Upload every staged file.\n", encoding="utf-8"
    )
    return context_dir


def define_stack() -> None:
    """Declare the Artifact and Workload that serve the configured checkpoint."""
    settings = read_settings(os.environ)
    context_dir = stage_build_context(settings.checkpoint_path, BUILD_CONTEXT_DIR)
    stack = pulumi.get_stack()
    name = f"JointFM Inference [{stack}]"

    artifact = datarobot.Artifact(
        "artifact",
        name=name,
        description="JointFM inference runner built from source with its checkpoint",
        type="service",
        source=datarobot.ArtifactSourceArgs(dir=str(context_dir)),
        spec=datarobot.ArtifactSpecArgs(
            container_groups=[
                datarobot.ArtifactSpecContainerGroupArgs(
                    containers=[
                        datarobot.ArtifactSpecContainerGroupContainerArgs(
                            name=CONTAINER_NAME,
                            primary=True,
                            port=CONTAINER_PORT,
                            image_build_config=datarobot.ArtifactSpecContainerGroupContainerImageBuildConfigArgs(
                                dockerfile=datarobot.ArtifactSpecContainerGroupContainerImageBuildConfigDockerfileArgs(
                                    source="provided", path="Dockerfile"
                                )
                            ),
                            environment_vars=[
                                datarobot.ArtifactSpecContainerGroupContainerEnvironmentVarArgs(
                                    source="string", name=key, value=value
                                )
                                for key, value in settings.service_env.items()
                            ],
                            startup_probe=datarobot.ArtifactSpecContainerGroupContainerStartupProbeArgs(
                                path=HEALTH_ROUTE,
                                port=CONTAINER_PORT,
                                period_seconds=STARTUP_PROBE_PERIOD_SECONDS,
                                failure_threshold=STARTUP_PROBE_FAILURE_THRESHOLD,
                            ),
                            readiness_probe=datarobot.ArtifactSpecContainerGroupContainerReadinessProbeArgs(
                                path=HEALTH_ROUTE, port=CONTAINER_PORT
                            ),
                        )
                    ]
                )
            ]
        ),
    )

    workload = datarobot.Workload(
        "workload",
        name=name,
        description="JointFM inference service",
        artifact_id=artifact.artifact_id,
        runtime=datarobot.WorkloadRuntimeArgs(
            container_groups=[
                datarobot.WorkloadRuntimeContainerGroupArgs(
                    replica_count=settings.replicas,
                    containers=[
                        datarobot.WorkloadRuntimeContainerGroupContainerArgs(
                            name=CONTAINER_NAME,
                            resource_allocation=datarobot.WorkloadRuntimeContainerGroupContainerResourceAllocationArgs(
                                cpu=settings.cpu,
                                memory=settings.memory,
                                gpu=settings.gpu,
                            ),
                        )
                    ],
                )
            ]
        ),
    )

    pulumi.export("artifact_id", artifact.artifact_id)
    pulumi.export("workload_id", workload.id)
    pulumi.export("endpoint", workload.endpoint)
