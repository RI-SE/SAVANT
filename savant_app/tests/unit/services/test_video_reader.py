# tests/unit/services/test_video_reader.py

import os
import pytest
import numpy as np
from src.savant_app.services.video_reader import VideoReader


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


def test_init_metadata_and_position(test_video_path: str):
    """__init__: opens the video, exposes metadata, and current_index starts at -1."""
    vr = VideoReader()
    vr.load_video(test_video_path)
    assert vr.metadata["frame_count"] > 0
    assert vr.metadata["width"] > 0
    assert vr.metadata["height"] > 0
    assert vr.metadata["fps"] >= 0
    assert vr.current_index == -1
    vr.release()


def test_next_advances_and_updates_current_index(test_video_path: str):
    """__next__: returns frames and advances OpenCV's internal position."""
    vr = VideoReader()
    vr.load_video(test_video_path)
    f0 = next(vr)
    assert isinstance(f0, np.ndarray)
    assert vr.current_index == 0
    next(vr)
    assert vr.current_index == 1

    vr.release()


def test_get_frame_and_updates_current_index(test_video_path: str):
    """get_frame(i): jumps to frame i and sets current_index to i."""
    vr = VideoReader()
    vr.load_video(test_video_path)
    f5 = vr.get_frame(5)
    assert isinstance(f5, np.ndarray)
    assert vr.current_index == 5
    vr.release()


def test_previous_frame_steps_back_when_possible(test_video_path: str):
    """previous_frame(): steps back one frame when not at the first frame."""
    vr = VideoReader()
    vr.load_video(test_video_path)
    next(vr)
    next(vr)
    prev = vr.previous_frame()
    assert isinstance(prev, np.ndarray)
    assert vr.current_index == 0
    vr.release()


def test_previous_frame_raises_at_first_frame(test_video_path: str):
    """previous_frame(): raises IndexError if we're at the first frame (no previous)."""
    vr = VideoReader()
    vr.load_video(test_video_path)
    next(vr)
    with pytest.raises(IndexError):
        vr.previous_frame()
    vr.release()


def test_stop_iteration_at_end(test_video_path: str):
    """__next__: raises StopIteration after the last frame."""
    vr = VideoReader()
    vr.load_video(test_video_path)
    for _ in range(vr.metadata["frame_count"]):
        frame = next(vr)
        assert isinstance(frame, np.ndarray)
    with pytest.raises(StopIteration):
        next(vr)
    vr.release()


def test_get_frame_out_of_bounds_raises(test_video_path: str):
    """get_frame: raises IndexError for indices outside [0, frame_count-1]."""
    vr = VideoReader()
    vr.load_video(test_video_path)
    with pytest.raises(IndexError):
        vr.get_frame(-1)
    with pytest.raises(IndexError):
        vr.get_frame(vr.metadata["frame_count"])
    vr.release()


def test_release_closes_capture(test_video_path: str):
    """release: closes the underlying cv2.VideoCapture handle."""
    vr = VideoReader()
    vr.load_video(test_video_path)
    vr.release()
    assert not vr.capture.isOpened()
