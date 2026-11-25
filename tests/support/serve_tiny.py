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

"""Serve the untrained tiny checkpoint over HTTP, for capturing client fixtures.

The ``jointfm-client`` SDK keeps copies of this service's real answers as its
test fixtures and captures them from a running service. Serving the seeded tiny
network of :mod:`tests.support.models` gives every capture the same weights
without a released checkpoint, so its answers show the contract and nothing of
a trained model. The service runs on the CPU whatever the environment names,
because an accelerator would change the captured numbers.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

from jointfm_inference.model.service import server
from jointfm_inference.service_settings import CHECKPOINT_PATH_ENV, DEVICE_ENV
from tests.support.models import write_checkpoint

TINY_CHECKPOINT_FILE_NAME = "tiny.pt"
TINY_SERVICE_DEVICE = "cpu"


def tiny_service_environment(directory: Path) -> dict[str, str]:
    """Write the tiny checkpoint into ``directory`` and return the settings that serve it.

    The returned ``JOINTFM_INFERENCE_*`` variables name the checkpoint and the
    CPU; every other setting keeps the value the environment gives it.
    """
    checkpoint_path = directory / TINY_CHECKPOINT_FILE_NAME
    write_checkpoint(checkpoint_path)
    return {
        CHECKPOINT_PATH_ENV: os.fspath(checkpoint_path),
        DEVICE_ENV: TINY_SERVICE_DEVICE,
    }


def main() -> None:
    """Serve the tiny checkpoint until the process is stopped, then delete it."""
    with tempfile.TemporaryDirectory() as directory:
        os.environ.update(tiny_service_environment(Path(directory)))
        server.main()


if __name__ == "__main__":
    main()
