from __future__ import annotations

from functools import lru_cache
from importlib import metadata
from importlib.metadata import PackageNotFoundError
from pathlib import Path
import re
from typing import Optional

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None


@lru_cache(maxsize=1)
def get_version() -> str:
    """Return the known SAVANT version string."""
    version = _version_from_metadata() or _version_from_pyproject()
    return version or "Unknown"


def _version_from_metadata() -> Optional[str]:
    try:
        return metadata.version("savant")
    except PackageNotFoundError:
        return None


def _version_from_pyproject() -> Optional[str]:
    current = Path(__file__).resolve()
    for ancestor in current.parents:
        candidate = ancestor / "pyproject.toml"
        if candidate.exists():
            version = _read_version_from_pyproject(candidate)
            if version:
                return version
    return None


def _read_version_from_pyproject(pyproject_path: Path) -> Optional[str]:
    if tomllib is not None:
        try:
            with pyproject_path.open("rb") as fh:
                data = tomllib.load(fh)
        except (OSError, tomllib.TOMLDecodeError):
            data = None
        if isinstance(data, dict):
            project = data.get("project")
            if isinstance(project, dict):
                version = project.get("version")
                if isinstance(version, str):
                    return version

    try:
        contents = pyproject_path.read_text(encoding="utf-8")
    except OSError:
        return None

    match = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"', contents)
    if match:
        return match.group(1)
    return None

