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

"""Frozen exports of every released JointFM 1 layout keep their promise.

Every other loader test writes its checkpoint with the current network, so it
round-trips and cannot notice a change to the layout itself. The files in
``released_layouts/`` were written by released versions of the runner and are
never regenerated. A layout is either supported, so its export loads and
predicts what it predicted when it was frozen, or retired, so its export is
refused for its format version. Retiring a layout therefore requires raising
``CHECKPOINT_FORMAT_VERSION`` above the version its exports carry.

To freeze a layout, save ``checkpoint_mapping(tiny_model(seed=0))`` from
``tests.support.models`` at the release that introduces it.
"""

from pathlib import Path

import pytest
import torch

from jointfm_inference.model.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CHECKPOINT_FORMAT_VERSION_KEY,
    CHECKPOINT_RUNTIME_METADATA_KEY,
    read_checkpoint_file,
)
from jointfm_inference.model.jointfm_1.loader import load_model_from_checkpoint

LAYOUTS = Path(__file__).parent / "released_layouts"

# Layout name to the first release that wrote it.
SUPPORTED = {"column-equivariant": "v0.3.0"}
# Layout name to the last release that read it.
RETIRED = {"column-slot": "v0.2.0"}


def _export(layout: str) -> dict:
    return read_checkpoint_file(str(LAYOUTS / f"{layout}.pt"))


def _format_version(layout: str) -> int:
    return _export(layout)[CHECKPOINT_RUNTIME_METADATA_KEY][
        CHECKPOINT_FORMAT_VERSION_KEY
    ]


def test_every_frozen_export_is_declared_supported_or_retired() -> None:
    frozen = {path.stem for path in LAYOUTS.glob("*.pt")} - {
        f"{layout}.outputs" for layout in SUPPORTED
    }
    assert frozen == SUPPORTED.keys() | RETIRED.keys()
    assert not SUPPORTED.keys() & RETIRED.keys()


def test_the_current_format_version_has_a_frozen_export() -> None:
    assert CHECKPOINT_FORMAT_VERSION in {_format_version(name) for name in SUPPORTED}


@pytest.mark.parametrize("layout", sorted(SUPPORTED))
def test_a_supported_layout_loads_and_predicts_what_it_did_when_frozen(
    layout: str,
) -> None:
    model, _ = load_model_from_checkpoint(_export(layout))
    frozen = torch.load(LAYOUTS / f"{layout}.outputs.pt", weights_only=True)

    with torch.no_grad():
        outputs = model(frozen["past"], frozen["horizons"])

    assert outputs.keys() == frozen["outputs"].keys()
    for key, expected in frozen["outputs"].items():
        torch.testing.assert_close(outputs[key], expected, msg=key)


@pytest.mark.parametrize("layout", sorted(RETIRED))
@pytest.mark.xfail(
    strict=True,
    reason="the column-slot layout was dropped before format versions covered "
    "the network, so it shares version 1 with the current layout",
)
def test_a_retired_layout_is_refused_for_its_format_version(layout: str) -> None:
    version = _format_version(layout)
    assert version < CHECKPOINT_FORMAT_VERSION

    with pytest.raises(
        ValueError,
        match=(
            f"unsupported checkpoint format version {version}; this runner reads "
            f"version {CHECKPOINT_FORMAT_VERSION}"
        ),
    ):
        load_model_from_checkpoint(_export(layout))
