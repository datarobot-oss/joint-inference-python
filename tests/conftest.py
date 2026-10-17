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

"""Pytest configuration shared by the whole suite."""

import os

import pytest
import torch

GIT_ENVIRONMENT_PREFIX = "GIT_"

# Every xdist worker is its own process; torch's default intra-op thread count
# per worker oversubscribes the CPU and multiplies the suite's wall time.
torch.set_num_threads(1)


def pytest_configure(config: pytest.Config) -> None:
    """Remove every ``GIT_*`` variable before any fixture or xdist worker starts.

    A hook run by ``git commit -a`` exports an absolute ``GIT_INDEX_FILE`` that
    points at this repository's ``index.lock``; a test that runs git in a
    temporary repository would otherwise read and write that index instead.
    """
    for name in [
        name for name in os.environ if name.startswith(GIT_ENVIRONMENT_PREFIX)
    ]:
        del os.environ[name]
