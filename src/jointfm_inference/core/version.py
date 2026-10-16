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
