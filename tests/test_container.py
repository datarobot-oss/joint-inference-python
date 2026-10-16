"""The container image ships the runner without weights or local configuration."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "container" / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"
# Everything the build context may hold; weights and `.env` are never among it.
BUILD_CONTEXT = ["README.md", "pyproject.toml", "uv.lock", "src/"]


def _instructions(keyword: str) -> list[str]:
    lines = DOCKERFILE.read_text(encoding="utf-8").splitlines()
    return [
        line[len(keyword) + 1 :] for line in lines if line.startswith(f"{keyword} ")
    ]


def test_the_build_context_admits_only_the_package_sources() -> None:
    entries = [
        line
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert entries[0] == "*"
    assert [entry[1:] for entry in entries if entry.startswith("!")] == BUILD_CONTEXT


def test_every_copy_takes_package_sources_or_the_built_environment() -> None:
    sources = [
        source
        for instruction in _instructions("COPY")
        if not instruction.startswith("--from=builder ")
        for source in instruction.split()[:-1]
    ]
    assert sorted(sources) == ["README.md", "pyproject.toml", "src", "uv.lock"]


def test_the_image_serves_from_the_models_volume() -> None:
    assert _instructions("CMD") == [json.dumps(["jointfm-inference", "serve"])]
    assert _instructions("VOLUME") == [json.dumps(["/models"])]
    assert "JOINTFM_INFERENCE_CHECKPOINT_PATH=/models/jointfm.pt" in _instructions(
        "ENV"
    )
