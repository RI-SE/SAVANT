"""Periodic autosave of unsaved annotations to <project>/.autosave/, offered for restore on the next open."""

from __future__ import annotations

import hashlib
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from edit.utils import write_atomic

if TYPE_CHECKING:
    from edit.models.OpenLabel import OpenLabel
    from edit.services.project_state import ProjectState

logger = logging.getLogger(__name__)

AUTOSAVE_DIR = ".autosave"
TICK_SECONDS = 30  # how often the timer checks; the user interval is in minutes


def autosave_path(openlabel_path: str | Path) -> Path:
    """Autosave file of an OpenLABEL file: same name in a hidden folder next to it."""
    path = Path(openlabel_path)
    return path.parent / AUTOSAVE_DIR / path.name


def delete_autosave(openlabel_path: str | Path) -> None:
    """Remove the autosave of a file (and the folder if it is then empty); errors are only logged."""
    path = autosave_path(openlabel_path)
    try:
        path.unlink(missing_ok=True)
        if path.parent.is_dir() and not any(path.parent.iterdir()):
            path.parent.rmdir()
    except OSError:
        logger.warning("Could not delete autosave %s", path)


def pending_autosave(project_state: ProjectState) -> OpenLabel | None:
    """The autosave of the loaded file if it is newer than the file and differs from it; None otherwise.

    An autosave that matches the file (e.g. written just before a manual save) is deleted.
    """
    if project_state.open_label_path is None:
        return None
    original = Path(project_state.open_label_path)
    candidate = autosave_path(original)
    try:
        if not candidate.is_file() or candidate.stat().st_mtime <= original.stat().st_mtime:
            return None
        config = project_state.parse_openlabel_file(str(candidate))
    except Exception:  # an unreadable autosave must never block opening the project
        logger.warning("Ignoring unreadable autosave %s", candidate, exc_info=True)
        return None
    if project_state.digest_of(config) == project_state.digest_of(project_state.annotation_config):
        delete_autosave(original)
        return None
    return config


class AutosaveService:
    """Writes the in-memory annotations to the autosave file when they changed since the last save.

    Call tick() regularly (a Qt timer in main.py). The snapshot is serialized on the calling thread, so it
    is consistent; only the file write runs in the background.
    """

    def __init__(
        self,
        project_state: ProjectState,
        interval_minutes: Callable[[], int],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._project_state = project_state
        self._interval_minutes = interval_minutes
        self._clock = clock
        self._last_attempt = clock()
        self._written: tuple[str, str] | None = None  # (path, digest) of the last successful autosave
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="autosave")
        self._pending: Future | None = None

    def tick(self) -> None:
        """Autosave if the interval has passed and there are changes not yet saved or autosaved."""
        minutes = self._interval_minutes()
        if minutes <= 0 or self._busy():
            return
        state = self._project_state
        if state.annotation_config is None or state.open_label_path is None:
            return
        if self._clock() - self._last_attempt < minutes * 60:
            return
        self._last_attempt = self._clock()
        path = state.open_label_path
        text = state.annotation_config.model_dump_json()
        digest = hashlib.sha256(text.encode()).hexdigest()
        if state.is_saved_digest(digest):
            return
        if self._written == (path, digest) and autosave_path(path).exists():
            return
        self._start_write(path, text, digest)

    def wait(self) -> None:
        """Block until a running write has finished (tests, shutdown)."""
        if self._pending is not None:
            self._pending.exception()

    def _busy(self) -> bool:
        return self._pending is not None and not self._pending.done()

    def _start_write(self, openlabel_path: str, text: str, digest: str) -> None:
        target = autosave_path(openlabel_path)

        def write() -> None:
            target.parent.mkdir(exist_ok=True)
            write_atomic(target, lambda f: f.write('{"openlabel": ' + text + "}"))

        def finished(future: Future) -> None:
            if future.exception() is None:
                self._written = (openlabel_path, digest)
                logger.info("Autosaved to %s", target)
            else:
                logger.warning("Autosave to %s failed; retrying at the next interval", target,
                               exc_info=future.exception())

        self._pending = self._executor.submit(write)
        self._pending.add_done_callback(finished)
