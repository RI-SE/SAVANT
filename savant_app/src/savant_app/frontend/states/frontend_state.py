from PyQt6.QtCore import QObject, pyqtSignal


class FrontendState(QObject):
    frameChanged = pyqtSignal(int)
    playingChanged = pyqtSignal(bool)
    zoomChanged = pyqtSignal(float)
    selectionChanged = pyqtSignal(object)
    toolChanged = pyqtSignal(str)
    boxesChanged = pyqtSignal(int, object)
    statusChanged = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame_index: int = 0
        self._is_playing: bool = False
        self._zoom: float = 1.0
        self._selected_object_id: str | None = None
        self._tool: str = "select"

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
