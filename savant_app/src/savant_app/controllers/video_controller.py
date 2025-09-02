# controllers/video_controller.py
from __future__ import annotations
import cv2
import numpy as np
from PyQt6.QtGui import QImage, QPixmap
from savant_app.services.video_reader import VideoReader


class VideoController:
    def __init__(self, reader: VideoReader) -> None:
        self.reader: VideoReader = reader

    # lifecycle
    def load_video(self, path: str) -> None:
        self.reader.load_video(path)

    def close_video(self) -> None:
        self.reader.release()

    # frames / navigation
    def next_frame(self) -> tuple[QPixmap, int] | tuple[None, None]:
        """Advance to next frame and return (pixmap, index)."""
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
        frame = self.reader.get_frame(index)
        if frame is None:
            return None, None
        return self._to_qpixmap(frame), self.reader.current_index

    def skip_frames(self, delta: int) -> tuple[QPixmap, int] | tuple[None, None]:
        """Move delta frames from current position, clamped to [0, frame_count-1]."""
        cur = max(self.reader.current_index, 0)
        last = self.reader.metadata["frame_count"] - 1
        target = min(max(cur + delta, 0), last)
        frame = self.reader.get_frame(target)
        if frame is None:
            return None, None
        return self._to_qpixmap(frame), self.reader.current_index

    # metadata
    def total_frames(self) -> int:
        return self.reader.metadata["frame_count"]

    def current_index(self) -> int:
        return self.reader.current_index

    def fps(self) -> float:
        return float(self.reader.metadata["fps"])

    def size(self) -> tuple[int, int]:
        return (self.reader.metadata["width"], self.reader.metadata["height"])
    
    def _ensure(self) -> None:
        """Ensure a video is loaded before using the reader."""
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
