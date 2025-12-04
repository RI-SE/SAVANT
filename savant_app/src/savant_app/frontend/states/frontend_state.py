from typing import Callable

from PyQt6.QtCore import QObject, pyqtSignal


class FrontendState(QObject):
    frameChanged = pyqtSignal(int)
    playingChanged = pyqtSignal(bool)
    zoomChanged = pyqtSignal(float)
    selectionChanged = pyqtSignal(object)
    toolChanged = pyqtSignal(str)
    boxesChanged = pyqtSignal(int, object)
    statusChanged = pyqtSignal(str)
    confidenceIssuesChanged = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame_index: int = 0
        self._is_playing: bool = False
        self._zoom: float = 1.0
        self._selected_object_id: str | None = None
        self._tool: str = "select"
        self._confidence_issues: dict[int, list] = {}
        self._warning_frames: set[int] = set()
        self._error_frames: set[int] = set()
        self._current_annotator: str = ""
        self._annotator_prompt: Callable[[], str | None] | None = None
        self._frame_tag_details: dict[int, list[dict]] = {}

    @property
    def frame_index(self):
        return self._frame_index

    @property
    def is_playing(self):
        return self._is_playing

    @property
    def zoom(self):
        return self._zoom

    @property
    def selected_object_id(self):
        return self._selected_object_id

    @property
    def tool(self):
        return self._tool

    def register_annotator_prompt(self, prompt: Callable[[], str | None]) -> None:
        self._annotator_prompt = prompt

    def set_current_annotator(self, new_annotator: str | None) -> str:
        normalized = (new_annotator or "").strip()
        self._current_annotator = normalized
        return self._current_annotator

    def get_current_annotator(self) -> str:
        return self._current_annotator

    def require_current_annotator(self) -> str | None:
        annotator = self.get_current_annotator()
        if annotator:
            return annotator
        if callable(self._annotator_prompt):
            provided = self._annotator_prompt() or ""
            if provided:
                self.set_current_annotator(provided)
                return provided
        return None

    def set_frame(self, idx: int):
        if idx != self._frame_index and idx >= 0:
            self._frame_index = idx
            self.frameChanged.emit(idx)

    def set_playing(self, playing: bool):
        if playing != self._is_playing:
            self._is_playing = playing
            self.playingChanged.emit(playing)

    def set_zoom(self, zoom: float):
        if zoom != self._zoom and zoom > 0:
            self._zoom = zoom
            self.zoomChanged.emit(zoom)

    def set_selected_object(self, obj_id: str | None):
        if obj_id != self._selected_object_id:
            self._selected_object_id = obj_id
            self.selectionChanged.emit(obj_id)

    def set_tool(self, tool: str):
        if tool != self._tool:
            self._tool = tool
            self.toolChanged.emit(tool)

    def publish_boxes(self, frame_index: int, items: list[tuple[str, object]]):
        self.boxesChanged.emit(frame_index, items)

    def set_status(self, text: str):
        self.statusChanged.emit(text)

    def set_confidence_issues(self, issues: dict[int, list] | None):
        issues = issues or {}
        if issues == self._confidence_issues:
            return
        warning_frames: set[int] = set()
        error_frames: set[int] = set()
        for frame_index, entries in issues.items():
            has_warning = False
            has_error = False
            for entry in entries:
                severity = getattr(entry, "severity", None)
                if severity == "warning":
                    has_warning = True
                elif severity == "error":
                    has_error = True
            if has_warning:
                warning_frames.add(frame_index)
            if has_error:
                error_frames.add(frame_index)
        self._confidence_issues = issues
        self._warning_frames = warning_frames
        self._error_frames = error_frames
        self.confidenceIssuesChanged.emit(issues)

    def confidence_issues(self) -> dict[int, list]:
        return dict(self._confidence_issues)

    def warning_frames(self) -> set[int]:
        return set(self._warning_frames)

    def error_frames(self) -> set[int]:
        return set(self._error_frames)

    def set_frame_tag_details(self, details: dict[int, list[dict]] | None):
        normalized: dict[int, list[dict]] = {}
        for frame, entries in (details or {}).items():
            try:
                frame_idx = int(frame)
            except (ValueError, TypeError):
                continue
            normalized[frame_idx] = list(entries or [])
        self._frame_tag_details = normalized

    def frame_tag_details_for(self, frame_index: int) -> list[dict]:
        try:
            key = int(frame_index)
        except (ValueError, TypeError):
            return []
        return list(self._frame_tag_details.get(key, []))
