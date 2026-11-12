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

"""Stage timers: off by default, free while off, and honest about device work."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch

from jointfm_inference.core import timing
from jointfm_inference.core.timing import CudaTimer, Timer

# Building a CUDA device object needs no GPU; only synchronizing it would.
_CUDA_DEVICE = torch.device("cuda:1")


@pytest.fixture(autouse=True)
def _timing_off() -> Iterator[None]:
    """Leave every test with timing off and an empty registry."""
    timing.set_enabled(False)
    timing.clear_timings()
    yield
    timing.set_enabled(False)
    timing.clear_timings()


@pytest.fixture
def synchronized(monkeypatch: pytest.MonkeyPatch) -> list[torch.device]:
    """Record every device a timer synchronizes instead of synchronizing it."""
    devices: list[torch.device] = []

    def _record(device: torch.device) -> None:
        devices.append(device)

    monkeypatch.setattr(torch.cuda, "synchronize", _record)
    return devices


def test_timing_is_off_by_default_and_then_records_and_synchronizes_nothing(
    synchronized: list[torch.device],
) -> None:
    assert not timing.is_enabled()
    with Timer("unit.host"), CudaTimer("unit.device", _CUDA_DEVICE):
        pass
    assert timing.stage_seconds() == {}
    assert synchronized == []


def test_each_region_appends_its_seconds_under_its_name() -> None:
    timing.set_enabled(True)
    for _ in range(2):
        with Timer("unit.host"):
            pass

    recorded = timing.stage_seconds()
    assert list(recorded) == ["unit.host"]
    assert len(recorded["unit.host"]) == 2
    assert all(seconds >= 0.0 for seconds in recorded["unit.host"])


def test_cuda_timer_synchronizes_its_own_device_at_both_ends(
    synchronized: list[torch.device],
) -> None:
    """A service on a second card must wait for that card, not the current one."""
    timing.set_enabled(True)
    with CudaTimer("unit.device", _CUDA_DEVICE):
        assert synchronized == [_CUDA_DEVICE]
    assert synchronized == [_CUDA_DEVICE, _CUDA_DEVICE]
    assert list(timing.stage_seconds()) == ["unit.device"]


def test_cuda_timer_on_a_host_device_never_synchronizes(
    synchronized: list[torch.device],
) -> None:
    timing.set_enabled(True)
    with CudaTimer("unit.cpu", torch.device("cpu")):
        pass
    assert synchronized == []
    assert list(timing.stage_seconds()) == ["unit.cpu"]


def test_a_region_left_by_an_exception_records_nothing_and_skips_the_closing_sync(
    synchronized: list[torch.device],
) -> None:
    """Synchronizing after a failed kernel would raise again and hide the first error."""
    timing.set_enabled(True)
    with pytest.raises(RuntimeError, match="boom"):
        with CudaTimer("unit.failed", _CUDA_DEVICE):
            raise RuntimeError("boom")
    assert synchronized == [_CUDA_DEVICE]
    assert timing.stage_seconds() == {}


def test_a_region_entered_while_off_records_nothing_after_timing_turns_on() -> None:
    with Timer("unit.straddling"):
        timing.set_enabled(True)
    assert timing.stage_seconds() == {}
