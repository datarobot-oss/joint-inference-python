import tomllib
from pathlib import Path

import jointfm_inference

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_version_matches_project_metadata() -> None:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    assert jointfm_inference.__version__ == project["version"]


def test_package_ships_type_marker() -> None:
    package_dir = Path(jointfm_inference.__file__).parent
    assert (package_dir / "py.typed").is_file()
