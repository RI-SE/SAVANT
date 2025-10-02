import cv2
import numpy as np
from .exceptions import VideoLoadError, VideoFrameIndexError, VideoReadError


class VideoReader:
    """Wrapper around OpenCV VideoCapture for frame iteration and random access."""

    def __init__(self) -> None:
        """Initialize the video reader."""
        self.capture: cv2.VideoCapture = None
        self.metadata = {"frame_count": 0, "width": 0, "height": 0, "fps": 0.0}

    def load_video(self, path: str) -> None:
        if self.capture and self.capture.isOpened():
            self.release()
        self.capture = cv2.VideoCapture(path)
        if not self.capture.isOpened():
            raise VideoLoadError(f"Could not open video file from path: {path}")

        self.metadata.update(
            {
                "frame_count": int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                "width": int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": self.capture.get(cv2.CAP_PROP_FPS),
            }
        )

    @property
    def current_index(self) -> int:
        """
        Get the index of the most recently read frame.

        OpenCV points CAP_PROP_POS_FRAMES to the *next* frame,
        so subtract 1 to get the last successfully returned frame.
        """
        self._validate_video_loaded()
        pos = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
        return pos - 1

    def __iter__(self):
        """Return self as iterator."""
        return self

    def __next__(self) -> np.ndarray:
        """Return the next frame, or raise StopIteration at end of video."""
        self._validate_video_loaded()
        success, frame = self.capture.read()
        if not success:
            raise StopIteration
        return frame

    def skip_frames(self, delta: int) -> np.ndarray:
        """
        Move delta frames from current position, clamped to [0, frame_count-1]
        Returns the frame at the new position.
        """
        self._validate_video_loaded()
        cur = max(self.current_index, 0)
        last = self.metadata["frame_count"] - 1
        target = min(max(cur + delta, 0), last)
        return self.get_frame(target)

    def get_frame(self, index: int) -> np.ndarray:
        """
        Retrieve a specific frame by index.

        Args:
            index (int): Frame index (0-based).

        Returns:
            np.ndarray: The requested frame.

        Raises:
            IndexError: If the index is out of range.
        """
        self._validate_video_loaded()
        if not (0 <= index < self.metadata["frame_count"]):
            raise VideoFrameIndexError("Frame index out of range")

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = self.capture.read()
        if not success:
            raise VideoReadError(f"Failed to read frame {index}")
        return frame

    def previous_frame(self) -> np.ndarray:
        """
        Retrieve the previous frame relative to the current index.
        
        Returns:
            np.ndarray: The previous frame
            
        Raises:
            IndexError: If already at the first frame.
        """
        self._validate_video_loaded()
        target = self.current_index - 1
        if target < 0:
            raise VideoFrameIndexError("Already at the first frame")
        return self.get_frame(target)

    def release(self) -> None:
        """Release the video capture resource."""
        if self.capture:
            self.capture.release()
        self.metadata.update({"frame_count": 0, "width": 0, "height": 0, "fps": 0.0})

    def _validate_video_loaded(self):
        """Ensure a video is loaded before performing operations."""
        if not self.capture or not self.capture.isOpened():
            raise VideoLoadError("No video loaded - please ensure you have imported a project.")
