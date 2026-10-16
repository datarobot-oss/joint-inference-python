"""Tests run free of the environment a calling git process exports."""

from __future__ import annotations

import os


def test_tests_see_no_git_environment_variables() -> None:
    """A hook run by ``git commit -a`` exports ``GIT_INDEX_FILE``; tests never see it."""
    assert [name for name in os.environ if name.startswith("GIT_")] == []
