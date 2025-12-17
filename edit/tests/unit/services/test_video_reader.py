# tests/unit/services/test_video_reader.py

import os
import pytest
import numpy as np
from src.edit.services.video_reader import VideoReader
from src.edit.services.exceptions import VideoFrameIndexError
from src.edit.services.types import VideoMetadata


# Mock project state class for testing
class SimpleProjectState:
    def __init__(self):
        self.video_metadata = VideoMetadata()


@pytest.fixture(scope="module")
def test_video_path() -> str:
    """
    Fixture that points to a local test video file.
    Adjust the path below to where your asset actually lives.
    """
    path = "tests/assets/Kraklanda_short.mp4"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected test video not found at {path}. "
            "Please add one under tests/assets."
        )
    return path


@pytest.fixture(scope="function")
def video_reader(test_video_path: str):
    """Fixture to initialize and release VideoReader for each test function"""
    project_state = SimpleProjectState()
    vr = VideoReader(project_state)
    vr.load_video(test_video_path)
    yield vr
    vr.release()


def test_init_metadata_and_position(video_reader: VideoReader):
    """__init__: opens the video, exposes metadata, and current_index starts at -1."""
    vr = video_reader
    assert vr.project_state.video_metadata.frame_count > 0
    assert vr.project_state.video_metadata.width > 0
    assert vr.project_state.video_metadata.height > 0
    assert vr.project_state.video_metadata.fps >= 0
    assert vr.current_index == -1


def test_next_advances_and_updates_current_index(video_reader: VideoReader):
    """__next__: returns frames and advances OpenCV's internal position."""
    vr = video_reader
    f0 = next(vr)
    assert isinstance(f0, np.ndarray)
    assert vr.current_index == 0
    next(vr)
    assert vr.current_index == 1


def test_get_frame_and_updates_current_index(video_reader: VideoReader):
    """get_frame(i): jumps to frame i and sets current_index to i."""
    vr = video_reader
    f5 = vr.get_frame(5)
    assert isinstance(f5, np.ndarray)
    assert vr.current_index == 5


def test_previous_frame_steps_back_when_possible(video_reader: VideoReader):
    """previous_frame(): steps back one frame when not at the first frame."""
    vr = video_reader
    next(vr)
    next(vr)
    prev = vr.previous_frame()
    assert isinstance(prev, np.ndarray)
    assert vr.current_index == 0


def test_previous_frame_raises_at_first_frame(video_reader: VideoReader):
    """previous_frame(): raises IndexError if we're at the first frame (no previous)."""
    vr = video_reader
    next(vr)
    with pytest.raises(VideoFrameIndexError):
        vr.previous_frame()


def test_stop_iteration_at_end(video_reader: VideoReader):
    """__next__: raises StopIteration after the last frame."""
    vr = video_reader
    for _ in range(vr.project_state.video_metadata.frame_count):
        frame = next(vr)
        assert isinstance(frame, np.ndarray)
    with pytest.raises(StopIteration):
        next(vr)


def test_get_frame_out_of_bounds_raises(video_reader: VideoReader):
    """get_frame: raises IndexError for indices outside [0, frame_count-1]."""
    vr = video_reader
    with pytest.raises(VideoFrameIndexError):
        vr.get_frame(-1)
    with pytest.raises(VideoFrameIndexError):
        vr.get_frame(vr.project_state.video_metadata.frame_count)


def test_release_closes_capture(test_video_path: str):
    """release: closes the underlying cv2.VideoCapture handle."""
    project_state = SimpleProjectState()
    vr = VideoReader(project_state)
    vr.load_video(test_video_path)
    vr.release()
    assert not vr.capture.isOpened()
