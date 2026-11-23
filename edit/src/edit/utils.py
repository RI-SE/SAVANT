"""Module containing useful util functions"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import IO, Callable


def read_json(file_path: str) -> dict:
    """Reads a JSON file and returns its contents as a dictionary."""
    with open(file_path, "r") as f:
        return json.load(f)


def write_atomic(target: Path, write: Callable[[IO[str]], None]) -> None:
    """Write a file through write(f) so that the target is replaced only after a complete write (keeps its mode)."""
    target = Path(target)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            write(f)
        if target.exists():
            shutil.copymode(target, tmp_name)
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
