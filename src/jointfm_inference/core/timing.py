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

"""Per-stage wall-clock timing of served requests.

Timing is off by default, and every timer is then a no-op that takes no clock
reading and never synchronizes a device, so the timers stay in the serving
path at no cost. :func:`set_enabled` turns it on for the process;
``jointfm-inference profile`` does so around the requests it times.

While timing is on, each completed region appends its seconds to a
process-wide registry under its stage name, read with :func:`stage_seconds`
and emptied with :func:`clear_timings`, and logs
``[timing] <name>: <seconds>s`` at ``DEBUG`` on the ``jointfm_inference.timing``
logger. A region left by an exception records nothing.

:class:`Timer` measures host wall time. :class:`CudaTimer` also synchronizes
its CUDA device before it starts and before it stops the clock, so a region
that launches kernels is charged with their run time instead of their launch;
unsynchronized, that run time would land in whichever later stage first waits
for the device, such as the copy of the response to the host. Each
synchronization stalls the host, so timers mark the few stage boundaries of
a request, never the inside of a per-sample loop.

The registry is shared by every thread, and a synchronization waits for all
work on its device, so a per-request split needs requests run one at a time.
"""

from __future__ import annotations

import logging
import threading
import time
from types import TracebackType

import torch

# Stage names of one served request. A name that extends another names a part
# of that stage; the parts need not cover the whole stage.
REQUEST_STAGE = "service.request"
EXECUTE_STAGE = "inference.execute"
PREPARE_STAGE = "inference.prepare"
FORWARD_STAGE = "inference.forward"
ENCODER_STAGE = "inference.forward.encoder"
POOL_HEAD_STAGE = "inference.forward.pool_head"
READOUT_STAGE = "inference.readout"
SAMPLE_CHOLESKY_STAGE = "inference.readout.cholesky"
SAMPLE_GAUSSIAN_STAGE = "inference.readout.gaussian"
SAMPLE_GAMMA_STAGE = "inference.readout.gamma"
SAMPLE_ASSEMBLE_STAGE = "inference.readout.assemble"
SERIALIZE_STAGE = "service.serialize"
ENCODE_STAGE = "service.encode"

_LOGGER = logging.getLogger("jointfm_inference.timing")
_REGISTRY_LOCK = threading.Lock()
_STAGE_SECONDS: dict[str, list[float]] = {}
_ENABLED = False


def set_enabled(enabled: bool) -> None:
    """Turn timing on or off for every timer in the process."""
    global _ENABLED
    _ENABLED = enabled


def is_enabled() -> bool:
    """Return whether timers currently record."""
    return _ENABLED


def clear_timings() -> None:
    """Drop every recorded duration."""
    with _REGISTRY_LOCK:
        _STAGE_SECONDS.clear()


def stage_seconds() -> dict[str, list[float]]:
    """Return a copy of the recorded durations, in seconds, by stage name."""
    with _REGISTRY_LOCK:
        return {name: list(values) for name, values in _STAGE_SECONDS.items()}


def _record(name: str, seconds: float) -> None:
    """Append one duration under ``name`` and log it."""
    with _REGISTRY_LOCK:
        _STAGE_SECONDS.setdefault(name, []).append(seconds)
    _LOGGER.debug("[timing] %s: %.6fs", name, seconds)


class Timer:
    """Context manager that records the host wall time of its region."""

    def __init__(self, name: str) -> None:
        """Name the stage the region's duration is recorded under."""
        self.name = name
        self._start: float | None = None

    def _synchronize(self) -> None:
        """Wait for device work the region must be charged with; none on the host."""

    def __enter__(self) -> Timer:
        """Start the clock while timing is on."""
        if _ENABLED:
            self._synchronize()
            self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Record the region unless it was not timed or raised."""
        start, self._start = self._start, None
        if start is None or exc_type is not None:
            return
        self._synchronize()
        _record(self.name, time.perf_counter() - start)


class CudaTimer(Timer):
    """Timer that synchronizes ``device`` at both ends when it is a CUDA device.

    On any other device it measures host wall time like :class:`Timer`.
    """

    def __init__(self, name: str, device: torch.device) -> None:
        """Name the stage and the device whose work the region launches."""
        super().__init__(name)
        self._device = device

    def _synchronize(self) -> None:
        """Wait for every kernel queued on the CUDA device."""
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)


__all__ = [
    "ENCODER_STAGE",
    "ENCODE_STAGE",
    "EXECUTE_STAGE",
    "FORWARD_STAGE",
    "POOL_HEAD_STAGE",
    "PREPARE_STAGE",
    "READOUT_STAGE",
    "REQUEST_STAGE",
    "SAMPLE_ASSEMBLE_STAGE",
    "SAMPLE_CHOLESKY_STAGE",
    "SAMPLE_GAMMA_STAGE",
    "SAMPLE_GAUSSIAN_STAGE",
    "SERIALIZE_STAGE",
    "CudaTimer",
    "Timer",
    "clear_timings",
    "is_enabled",
    "set_enabled",
    "stage_seconds",
]
