"""Tests for the optical-flow SimpleTracker."""

from markit.markitlib.processing.engines import SimpleTracker


def test_one_track_is_not_assigned_to_two_detections_in_a_frame():
    tracker = SimpleTracker(max_distance=50, min_iou=0)
    first = tracker.get_id((100, 100), frame_idx=0)

    a = tracker.get_id((105, 100), frame_idx=1)
    b = tracker.get_id((95, 100), frame_idx=1)

    assert a == first
    assert b != a


def test_track_is_matched_again_in_next_frame():
    tracker = SimpleTracker(max_distance=50, min_iou=0)
    first = tracker.get_id((100, 100), frame_idx=0)
    assert tracker.get_id((102, 100), frame_idx=1) == first
    assert tracker.get_id((104, 100), frame_idx=2) == first
