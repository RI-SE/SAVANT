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
    def next_frame(self) -> QPixmap:
        self._ensure()
        frame = next(self.reader)
        return self._to_qpixmap(frame)

    def previous_frame(self) -> QPixmap:
        self._ensure()
        frame = self.reader.previous_frame()
        return self._to_qpixmap(frame)

    def jump_to_frame(self, index: int) -> QPixmap:
        self._ensure()
        frame = self.reader.get_frame(index)
        return self._to_qpixmap(frame)

    def skip_frames(self, delta: int) -> QPixmap:
        """
        Move delta frames from current position, clamped to [0, frame_count-1].
        Uses OpenCV's current position (current_index property).
        """
        self._ensure()
        cur = max(self.reader.current_index, 0)
        last = self.reader.frame_count - 1
        target = min(max(cur + delta, 0), last)
        frame = self.reader.get_frame(target)
        return self._to_qpixmap(frame)

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
