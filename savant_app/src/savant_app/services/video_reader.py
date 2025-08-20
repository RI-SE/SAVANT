import cv2
import numpy as np


class VideoReader:
    """Wrapper around OpenCV VideoCapture for frame iteration and random access."""

    def __init__(self, video_path: str) -> None:
        """
        Initialize the video reader.

        Args:
            video_path (str): Path to the video file.
        """
        self.video_path: str = video_path
        self.capture: cv2.VideoCapture = cv2.VideoCapture(video_path)

        if not self.capture.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Video metadata
        self.frame_count: int = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width: int = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps: float = self.capture.get(cv2.CAP_PROP_FPS)

    @property
    def current_index(self) -> int:
        """
        Get the index of the most recently read frame.

        OpenCV points CAP_PROP_POS_FRAMES to the *next* frame,
        so subtract 1 to get the last successfully returned frame.
        """
        pos = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
        return pos - 1

    def __iter__(self):
        """Return self as iterator."""
        return self

    def __next__(self) -> np.ndarray:
        """Return the next frame, or raise StopIteration at end of video."""
        success, frame = self.capture.read()
        if not success:
            raise StopIteration
        return frame

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
        if not (0 <= index < self.frame_count):
            raise IndexError("Frame index out of range")

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = self.capture.read()
        if not success:
            raise RuntimeError(f"Failed to read frame {index}")
        return frame

    def previous_frame(self) -> np.ndarray:
        """
        Retrieve the previous frame relative to the current index.

        Returns:
            np.ndarray: The previous frame.

        Raises:
            IndexError: If already at the first frame.
        """
        target = self.current_index - 1
        if target < 0:
            raise IndexError("Already at the first frame")
        return self.get_frame(target)

    def release(self) -> None:
        """Release the video capture resource."""
        self.capture.release()
