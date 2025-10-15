# controllers/video_controller.py
from __future__ import annotations
import cv2
import numpy as np
from PyQt6.QtGui import QPixmap, QImage
from savant_app.services.video_reader import VideoReader
from .error_handler_middleware import error_handler


class VideoController:
    def __init__(self, reader: VideoReader) -> None:
        self.reader: VideoReader = reader

    @error_handler
    def load_video(self, path: str) -> None:
        self.reader.load_video(path)

    @error_handler
    def close_video(self) -> None:
        self.reader.release()

    @error_handler
    def next_frame(self) -> tuple[QPixmap, int] | tuple[None, None]:
        """Advance to next frame and return (pixmap, index)."""
        try:
            frame = next(self.reader)
            return self._convert_frame_to_pixmap(frame), self.reader.current_index
        except StopIteration:
            # Handle end of video stream case
            return None, None

    @error_handler
    def previous_frame(self) -> tuple[QPixmap, int] | tuple[None, None]:
        # try:
        frame = self.reader.previous_frame()
        return self._convert_frame_to_pixmap(frame), self.reader.current_index
        # except (IndexError, RuntimeError):
        #    return None, None

    @error_handler
    def jump_to_frame(self, index: int) -> tuple[QPixmap, int] | tuple[None, None]:
        """Jump to an exact frame and return (pixmap, index)."""
        frame = self.reader.get_frame(index)
        return self._convert_frame_to_pixmap(frame), self.reader.current_index

    @error_handler
    def skip_frames(self, delta: int) -> tuple[QPixmap, int] | tuple[None, None]:
        """Move delta frames from current position, clamped to [0, frame_count-1]."""
        frame = self.reader.skip_frames(delta)
        return self._convert_frame_to_pixmap(frame), self.reader.current_index

    @error_handler
    def current_index(self) -> int:
        return self.reader.current_index

    # TODO: Can eventually move this to frontend for full SOC
    def _convert_frame_to_pixmap(self, frame: np.ndarray) -> QPixmap:
        """Convert OpenCV BGR ndarray to QPixmap"""
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected BGR image with shape (H, W, 3)")

        # Convert color space from BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape

        # Create QImage from numpy array
        qimage = QImage(
            rgb.data, width, height, channels * width, QImage.Format.Format_RGB888
        )

        return QPixmap.fromImage(qimage)
