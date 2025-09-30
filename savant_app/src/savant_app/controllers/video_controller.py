# controllers/video_controller.py
from __future__ import annotations
import cv2
import numpy as np
from PyQt6.QtGui import QPixmap, QImage
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
            return self._convert_frame_to_pixmap(frame), self.reader.current_index
        except (StopIteration, RuntimeError):
            return None, None

    def previous_frame(self) -> tuple[QPixmap, int] | tuple[None, None]:
        try:
            frame = self.reader.previous_frame()
            return self._convert_frame_to_pixmap(frame), self.reader.current_index
        except (IndexError, RuntimeError):
            return None, None

    def jump_to_frame(self, index: int) -> tuple[QPixmap, int] | tuple[None, None]:
        """Jump to an exact frame and return (pixmap, index)."""
        try:
            frame = self.reader.get_frame(index)
            return self._convert_frame_to_pixmap(frame), self.reader.current_index
        except (IndexError, RuntimeError):
            return None, None

    def skip_frames(self, delta: int) -> tuple[QPixmap, int] | tuple[None, None]:
        """Move delta frames from current position, clamped to [0, frame_count-1]."""
        try:
            frame = self.reader.skip_frames(delta)
            return self._convert_frame_to_pixmap(frame), self.reader.current_index
        except (IndexError, RuntimeError):
            return None, None

    # metadata
    def total_frames(self) -> int:
        return self.reader.metadata["frame_count"]

    def current_index(self) -> int:
        return self.reader.current_index

    def fps(self) -> float:
        return float(self.reader.metadata["fps"])

    def size(self) -> tuple[int, int]:
        return (self.reader.metadata["width"], self.reader.metadata["height"])

    def _convert_frame_to_pixmap(self, frame: np.ndarray) -> QPixmap:
        """Convert OpenCV BGR ndarray to QPixmap"""
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected BGR image with shape (H, W, 3)")
        
        # Convert color space from BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        
        # Create QImage from numpy array
        qimage = QImage(
            rgb.data, 
            width, 
            height, 
            channels * width, 
            QImage.Format.Format_RGB888
        )
        
        return QPixmap.fromImage(qimage)
