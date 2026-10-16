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
