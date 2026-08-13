"""Background autosave service for the edit application."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal

from edit.services.project_state import ProjectState

logger = logging.getLogger(__name__)

AUTOSAVE_DIR = ".autosave"
_OPENLABEL_WRAPPER_KEY = "openlabel"


class AutosaveWorker(QThread):
    """Writes a pre-built snapshot dict to the autosave directory in a background thread."""

    finished = pyqtSignal(bool)  # True on success

    def __init__(
        self,
        openlabel_snapshot: dict,
        openlabel_path: Path,
        project_config_snapshot: dict | None,
        config_path: Path | None,
    ) -> None:
        super().__init__()
        self._openlabel_snapshot = openlabel_snapshot
        self._openlabel_path = openlabel_path
        self._project_config_snapshot = project_config_snapshot
        self._config_path = config_path

    def run(self) -> None:
        try:
            self._openlabel_path.parent.mkdir(parents=True, exist_ok=True)
            with self._openlabel_path.open("w", encoding="utf-8") as fh:
                json.dump(
                    {_OPENLABEL_WRAPPER_KEY: self._openlabel_snapshot}, fh, indent=2
                )
            if self._project_config_snapshot is not None and self._config_path is not None:
                self._config_path.parent.mkdir(parents=True, exist_ok=True)
                with self._config_path.open("w", encoding="utf-8") as fh:
                    json.dump(self._project_config_snapshot, fh, indent=2)
            self.finished.emit(True)
        except Exception:
            logger.exception("Autosave failed")
            self.finished.emit(False)


class AutosaveService(QObject):
    """Periodically saves a snapshot of the annotation state to a .autosave/ directory.

    The snapshot is taken (deepcopy) on the main thread to guarantee consistency,
    then written by AutosaveWorker so the UI is never blocked.
    """

    def __init__(
        self,
        project_state: ProjectState,
        *,
        get_project_config_snapshot: Callable[[], dict | None] | None = None,
    ) -> None:
        super().__init__()
        self._project_state = project_state
        self._get_project_config_snapshot = get_project_config_snapshot
        self._dirty = False
        self._worker: AutosaveWorker | None = None

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer_tick)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mark_dirty(self) -> None:
        self._dirty = True

    def clear_dirty(self) -> None:
        self._dirty = False

    def set_interval(self, minutes: int) -> None:
        """Set the autosave interval. Pass 0 to disable."""
        self._timer.stop()
        if minutes > 0:
            self._timer.start(minutes * 60 * 1000)

    @staticmethod
    def autosave_openlabel_path(project_dir: Path, openlabel_path: Path) -> Path:
        return project_dir / AUTOSAVE_DIR / openlabel_path.name

    @staticmethod
    def autosave_config_path(project_dir: Path, config_filename: str) -> Path:
        return project_dir / AUTOSAVE_DIR / config_filename

    def delete_autosaves(self, project_dir: Path, openlabel_path: Path, config_filename: str) -> None:
        """Remove autosave files after a successful manual save."""
        for path in (
            self.autosave_openlabel_path(project_dir, openlabel_path),
            self.autosave_config_path(project_dir, config_filename),
        ):
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                logger.warning("Could not delete autosave file: %s", path)

        autosave_dir = project_dir / AUTOSAVE_DIR
        try:
            if autosave_dir.is_dir() and not any(autosave_dir.iterdir()):
                autosave_dir.rmdir()
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_timer_tick(self) -> None:
        if not self._dirty:
            return
        if self._project_state.annotation_config is None:
            return
        if self._project_state.open_label_path is None:
            return
        if self._worker is not None and self._worker.isRunning():
            logger.debug("Autosave skipped: previous save still in progress")
            return

        openlabel_path = Path(self._project_state.open_label_path)
        project_dir = openlabel_path.parent

        # Snapshot on main thread for consistency
        openlabel_snapshot = self._project_state.annotation_config.model_dump(mode="json")
        project_config_snapshot = (
            self._get_project_config_snapshot()
            if self._get_project_config_snapshot is not None
            else None
        )

        autosave_ol_path = self.autosave_openlabel_path(project_dir, openlabel_path)
        autosave_cfg_path = (
            self.autosave_config_path(project_dir, "savant_project_config.json")
            if project_config_snapshot is not None
            else None
        )

        self._worker = AutosaveWorker(
            openlabel_snapshot=openlabel_snapshot,
            openlabel_path=autosave_ol_path,
            project_config_snapshot=project_config_snapshot,
            config_path=autosave_cfg_path,
        )
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()
        logger.debug("Autosave started → %s", autosave_ol_path)

    def _on_worker_finished(self, success: bool) -> None:
        if success:
            self._dirty = False
            logger.info("Autosave completed successfully")
        else:
            logger.warning("Autosave failed — will retry on next interval")
        self._worker = None
