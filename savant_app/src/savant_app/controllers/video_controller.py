# controllers/video_controller.py
from __future__ import annotations
import cv2
import numpy as np
from PyQt6.QtGui import QImage, QPixmap
from savant_app.services.video_reader import VideoReader


class VideoController:
    def __init__(self) -> None:
        self.reader: VideoReader | None = None

    # lifecycle
    def load_video(self, path: str) -> None:
        self.close_video()
        self.reader = VideoReader(path)

    def close_video(self) -> None:
        if self.reader:
            self.reader.release()
            self.reader = None

    # frames / navigation
    def next_frame(self) -> tuple[QPixmap, int] | tuple[None, None]:
        """Advance to next frame and return (pixmap, index)."""
        self._ensure()
        try:
            frame = next(self.reader)
        except StopIteration:
            return None, None
        return self._to_qpixmap(frame), self.reader.current_index

    def previous_frame(self) -> tuple[QPixmap, int] | tuple[None, None]:
        self._ensure()
        frame = self.reader.previous_frame()
        if frame is None:
            return None, None
        return self._to_qpixmap(frame), self.reader.current_index

    def jump_to_frame(self, index: int) -> tuple[QPixmap, int] | tuple[None, None]:
        """Jump to an exact frame and return (pixmap, index)."""
        self._ensure()
        frame = self.reader.get_frame(index)
        if frame is None:
            return None, None
        return self._to_qpixmap(frame), self.reader.current_index

    def skip_frames(self, delta: int) -> tuple[QPixmap, int] | tuple[None, None]:
        """Move delta frames from current position, clamped to [0, frame_count-1]."""
        self._ensure()
        cur = max(self.reader.current_index, 0)
        last = self.reader.frame_count - 1
        target = min(max(cur + delta, 0), last)
        frame = self.reader.get_frame(target)
        if frame is None:
            return None, None
        return self._to_qpixmap(frame), self.reader.current_index

    # metadata
    def total_frames(self) -> int:
        self._ensure()
        return self.reader.frame_count

    def current_index(self) -> int:
        self._ensure()
        return self.reader.current_index

    def fps(self) -> float:
        self._ensure()
        return float(self.reader.fps)

    def size(self) -> tuple[int, int]:
        self._ensure()
        return (self.reader.width, self.reader.height)

    # helpers
    def _ensure(self) -> None:
        if self.reader is None:
            raise RuntimeError("No video loaded")

    def _to_qpixmap(self, bgr: np.ndarray) -> QPixmap:
        """Convert OpenCV BGR ndarray to QPixmap."""
        if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("Expected BGR image with shape (H, W, 3)")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)
