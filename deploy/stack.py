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

A resource bundle from the cluster's catalog picks the node, and optional
CPU, memory, and GPU overrides narrow the container's share of it. Both are
checked against the live catalog before anything is staged or uploaded, and a
bundle with a GPU runs the model on CUDA.

Every setting comes from the environment, which the ``deploy:*`` Taskfile
targets load from the same ``.env`` that ``task serve`` reads.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pulumi
import pulumi_datarobot as datarobot

from jointfm_inference.model.service.config import (
    CHECKPOINT_PATH_ENV,
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
DEPLOY_BUNDLE_ENV = "JOINTFM_DEPLOY_BUNDLE"
DEPLOY_CPU_ENV = "JOINTFM_DEPLOY_CPU"
DEPLOY_MEMORY_ENV = "JOINTFM_DEPLOY_MEMORY"
DEPLOY_GPU_ENV = "JOINTFM_DEPLOY_GPU"
DEPLOY_REPLICAS_ENV = "JOINTFM_DEPLOY_REPLICAS"
DEFAULT_DEPLOY_BUNDLE = "cpu.5xlarge"
DEFAULT_DEPLOY_REPLICAS = "1"

BUNDLES_ROUTE = "/mlops/compute/bundles/"
WORKLOAD_BUNDLE_USE_CASE = "workload"
BUNDLES_TIMEOUT_SECONDS = 30.0
CUDA_DEVICE = "cuda"

# The size suffixes the Workload API accepts for memory.
_MEMORY_UNITS = {
    "": 1,
    "KB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "TB": 1000**4,
    "Ki": 1024,
    "KiB": 1024,
    "Mi": 1024**2,
    "MiB": 1024**2,
    "Gi": 1024**3,
    "GiB": 1024**3,
    "Ti": 1024**4,
    "TiB": 1024**4,
}
_MEMORY_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*([A-Za-z]*)")

# Service settings passed through to the container when set. The checkpoint
# path, download settings, bind host, and port are fixed by the image instead.
FORWARDED_SERVICE_ENV = (
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
    bundle: str
    cpu: float | None
    memory: str | None
    gpu: float | None
    replicas: int


@dataclass(frozen=True)
class ResourceBundle:
    """One entry of the cluster's compute bundle catalog."""

    id: str
    name: str
    cpu: float
    memory_bytes: int
    gpu: int
    gpu_type: str | None
    gpu_memory_bytes: int

    @classmethod
    def from_api(cls, item: Mapping[str, Any]) -> ResourceBundle:
        """Read one entry of the bundle catalog's ``data`` list."""
        return cls(
            id=item["id"],
            name=item["name"],
            cpu=float(item["cpuCount"]),
            memory_bytes=int(item["memoryBytes"]),
            gpu=int(item.get("gpuCount") or 0),
            gpu_type=item.get("gpuTypeLabel"),
            gpu_memory_bytes=int(item.get("gpuMemoryBytes") or 0),
        )


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


def memory_bytes(value: str) -> int:
    """Return the bytes of a memory size such as ``8GB``, ``4096Mi``, or ``1024``."""
    match = _MEMORY_PATTERN.fullmatch(value.strip())
    if match is None or match.group(2) not in _MEMORY_UNITS:
        raise ValueError(
            f"{DEPLOY_MEMORY_ENV} must be a size such as 8GB or 4096Mi, got {value!r}"
        )
    return int(float(match.group(1)) * _MEMORY_UNITS[match.group(2)])


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
    service_env = {
        name: value
        for name in FORWARDED_SERVICE_ENV
        if (value := _optional(env, name)) is not None
    }

    cpu_value = _optional(env, DEPLOY_CPU_ENV)
    memory = _optional(env, DEPLOY_MEMORY_ENV)
    if memory is not None:
        memory_bytes(memory)
    gpu_value = _optional(env, DEPLOY_GPU_ENV)
    replicas = int(env.get(DEPLOY_REPLICAS_ENV) or DEFAULT_DEPLOY_REPLICAS)
    if replicas < 1:
        raise ValueError(f"{DEPLOY_REPLICAS_ENV} must be at least 1, got {replicas}")
    return DeploySettings(
        checkpoint_path=checkpoint_path.resolve(),
        service_env=service_env,
        bundle=_optional(env, DEPLOY_BUNDLE_ENV) or DEFAULT_DEPLOY_BUNDLE,
        cpu=None if cpu_value is None else _positive_number(DEPLOY_CPU_ENV, cpu_value),
        memory=memory,
        gpu=None if gpu_value is None else _positive_number(DEPLOY_GPU_ENV, gpu_value),
        replicas=replicas,
    )


def fetch_workload_bundles(endpoint: str, token: str) -> list[ResourceBundle]:
    """Return the compute bundles the cluster offers Workloads."""
    url: str | None = (
        f"{endpoint.rstrip('/')}{BUNDLES_ROUTE}?useCases={WORKLOAD_BUNDLE_USE_CASE}"
    )
    bundles: list[ResourceBundle] = []
    while url is not None:
        request = Request(url, headers={"Authorization": f"Bearer {token}"})
        with urlopen(request, timeout=BUNDLES_TIMEOUT_SECONDS) as response:
            page = json.load(response)
        bundles.extend(ResourceBundle.from_api(item) for item in page["data"])
        url = page.get("next")
    return bundles


def fit_to_bundle(
    settings: DeploySettings, bundles: Sequence[ResourceBundle]
) -> dict[str, str]:
    """Check the bundle and overrides against the catalog; return the container env.

    A container that gets a GPU runs the model on CUDA, so the device is set
    to ``cuda`` unless ``.env`` already names a CUDA device. A device that
    disagrees with the GPU count is refused rather than left to serve slowly
    on the wrong hardware.
    """
    by_id = {bundle.id: bundle for bundle in bundles}
    bundle = by_id.get(settings.bundle)
    if bundle is None:
        raise ValueError(
            f"{DEPLOY_BUNDLE_ENV}={settings.bundle!r} is not a Workload bundle on "
            "this cluster; task deploy:bundles lists the ones it offers"
        )
    if settings.cpu is not None and settings.cpu > bundle.cpu:
        raise ValueError(
            f"{DEPLOY_CPU_ENV}={settings.cpu:g} exceeds the {bundle.cpu:g} CPU "
            f"of bundle {bundle.id}"
        )
    if (
        settings.memory is not None
        and memory_bytes(settings.memory) > bundle.memory_bytes
    ):
        raise ValueError(
            f"{DEPLOY_MEMORY_ENV}={settings.memory} exceeds the "
            f"{bundle.memory_bytes} bytes of bundle {bundle.id}"
        )
    if settings.gpu is not None and settings.gpu > bundle.gpu:
        raise ValueError(
            f"{DEPLOY_GPU_ENV}={settings.gpu:g} exceeds the {bundle.gpu} GPU of "
            f"bundle {bundle.id}"
        )

    gpu = bundle.gpu if settings.gpu is None else settings.gpu
    service_env = dict(settings.service_env)
    device = service_env.get(DEVICE_ENV)
    on_cuda = device is not None and device.startswith(CUDA_DEVICE)
    if gpu > 0 and device is None:
        service_env[DEVICE_ENV] = CUDA_DEVICE
    elif gpu > 0 and not on_cuda:
        raise ValueError(
            f"bundle {bundle.id} has a GPU but {DEVICE_ENV}={device}; "
            f"unset it or name a {CUDA_DEVICE} device"
        )
    elif gpu == 0 and on_cuda:
        raise ValueError(f"bundle {bundle.id} has no GPU but {DEVICE_ENV}={device}")
    return service_env


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
    service_env = fit_to_bundle(
        settings,
        fetch_workload_bundles(
            os.environ[DATAROBOT_ENDPOINT_ENV], os.environ[DATAROBOT_API_TOKEN_ENV]
        ),
    )
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
                                for key, value in service_env.items()
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

    # Without an override the container takes what the bundle provides.
    overridden = any(
        value is not None for value in (settings.cpu, settings.memory, settings.gpu)
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
                    resource_bundles=[settings.bundle],
                    containers=[
                        datarobot.WorkloadRuntimeContainerGroupContainerArgs(
                            name=CONTAINER_NAME,
                            resource_allocation=datarobot.WorkloadRuntimeContainerGroupContainerResourceAllocationArgs(
                                cpu=settings.cpu,
                                memory=settings.memory,
                                gpu=settings.gpu,
                            ),
                        )
                    ]
                    if overridden
                    else None,
                )
            ]
        ),
    )

    pulumi.export("artifact_id", artifact.artifact_id)
    pulumi.export("workload_id", workload.id)
    pulumi.export("resource_bundle", settings.bundle)
    pulumi.export("endpoint", workload.endpoint)
