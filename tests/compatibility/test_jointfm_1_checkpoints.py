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

"""Backward compatibility: every released JointFM 1 checkpoint layout keeps loading.

People hold JointFM 1 checkpoints exported for earlier releases of this
runner. A failure here means this change stops one of them from loading or
changes what it predicts. Fixing the test is never the remedy: restore the old
behavior, or follow "Compatibility of released exports" in CONTRIBUTING.md,
which needs the maintainers' approval. See README.md beside this file.
"""

from pathlib import Path
from typing import Any

import pytest
import torch

from jointfm_inference.model.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CHECKPOINT_FORMAT_VERSION_KEY,
    CHECKPOINT_RUNTIME_METADATA_KEY,
    read_checkpoint_file,
)
from jointfm_inference.model.jointfm_1.loader import load_model_from_checkpoint

CHECKPOINTS = Path(__file__).parent / "jointfm_1_checkpoints"
OUTPUTS_SUFFIX = ".outputs"

# Layout name to the first release that wrote it. Each loads and predicts its frozen outputs.
SUPPORTED = {"column-equivariant": "v0.3.0"}
# Layout name to the last release that read it. Each is refused for its format version.
RETIRED = {"column-slot": "v0.2.0"}

BROKEN = (
    "BACKWARD COMPATIBILITY BROKEN: a released JointFM 1 checkpoint ({layout}, "
    "first written by {release}) {what}. Restore the old behavior, or follow "
    "'Compatibility of released exports' in CONTRIBUTING.md. Never regenerate "
    "the frozen files to make this pass."
)


def _checkpoint(layout: str) -> dict[str, Any]:
    return read_checkpoint_file(str(CHECKPOINTS / f"{layout}.pt"))


def _format_version(layout: str) -> int:
    return _checkpoint(layout)[CHECKPOINT_RUNTIME_METADATA_KEY][
        CHECKPOINT_FORMAT_VERSION_KEY
    ]


def test_every_frozen_checkpoint_is_declared_supported_or_retired() -> None:
    frozen = {
        path.stem
        for path in CHECKPOINTS.glob("*.pt")
        if not path.stem.endswith(OUTPUTS_SUFFIX)
    }
    assert frozen == SUPPORTED.keys() | RETIRED.keys(), (
        "Every file in jointfm_1_checkpoints/ must be listed in SUPPORTED or "
        "RETIRED, and every listed layout must have its file. Deleting a file "
        "does not retire a layout."
    )
    assert not SUPPORTED.keys() & RETIRED.keys()


def test_the_current_format_version_has_a_frozen_checkpoint() -> None:
    versions = {_format_version(layout) for layout in SUPPORTED}
    assert CHECKPOINT_FORMAT_VERSION in versions, (
        f"CHECKPOINT_FORMAT_VERSION is {CHECKPOINT_FORMAT_VERSION}, but no "
        "supported frozen checkpoint carries it. Freeze one with the release "
        "that introduces the version (see README.md beside this file)."
    )


@pytest.mark.parametrize("layout", sorted(SUPPORTED))
def test_a_released_checkpoint_still_loads(layout: str) -> None:
    try:
        load_model_from_checkpoint(_checkpoint(layout))
    except Exception as error:
        pytest.fail(
            BROKEN.format(
                layout=layout,
                release=SUPPORTED[layout],
                what=f"no longer loads:\n{error}",
            )
        )


@pytest.mark.parametrize("layout", sorted(SUPPORTED))
def test_a_released_checkpoint_still_predicts_what_it_did(layout: str) -> None:
    model, _ = load_model_from_checkpoint(_checkpoint(layout))
    frozen = torch.load(CHECKPOINTS / f"{layout}{OUTPUTS_SUFFIX}.pt", weights_only=True)

    with torch.no_grad():
        outputs = model(frozen["past"], frozen["horizons"])

    broken = BROKEN.format(
        layout=layout, release=SUPPORTED[layout], what="predicts differently"
    )
    assert outputs.keys() == frozen["outputs"].keys(), broken
    for key, expected in frozen["outputs"].items():
        torch.testing.assert_close(
            outputs[key], expected, msg=lambda detail: f"{broken}\n{key}: {detail}"
        )


@pytest.mark.parametrize("layout", sorted(RETIRED))
@pytest.mark.xfail(
    strict=True,
    reason="the column-slot layout was dropped before format versions covered "
    "the network, so it shares version 1 with the current layout",
)
def test_a_retired_checkpoint_is_refused_for_its_format_version(layout: str) -> None:
    version = _format_version(layout)
    assert version < CHECKPOINT_FORMAT_VERSION, (
        f"The retired {layout} layout carries format version {version}, which "
        "this runner still reads. Retiring a layout requires raising "
        "CHECKPOINT_FORMAT_VERSION."
    )

    with pytest.raises(
        ValueError,
        match=(
            f"unsupported checkpoint format version {version}; this runner reads "
            f"version {CHECKPOINT_FORMAT_VERSION}"
        ),
    ):
        load_model_from_checkpoint(_checkpoint(layout))
