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

"""Version helpers for installed JointFM distributions."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version


def package_version(distribution_name: str) -> str:
    """Return the installed version for one Python distribution.

    Args:
        distribution_name: Installed distribution name as exposed through
            ``importlib.metadata``.
    Returns:
        The installed version string.
    Raises:
        RuntimeError: The requested distribution metadata is unavailable.
    """

    try:
        return distribution_version(distribution_name)
    except PackageNotFoundError as error:
        raise RuntimeError(
            "Required package metadata is unavailable for "
            f"'{distribution_name}'. Install that distribution before requesting "
            "its version."
        ) from error


__all__ = ["package_version"]
