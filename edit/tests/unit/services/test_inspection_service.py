# edit/tests/unit/services/test_inspection_service.py
import math
from unittest.mock import MagicMock

from edit.services.inspection_service import (
    _bbox_corners,
    _bboxes_overlap,
    _intersection_area,
    _polygon_area,
    detect_double_frames,
    detect_ghost_frames,
    run_inspection,
)


def _make_bbox(cx, cy, w, h, rot=0.0):
    """Create a minimal RotatedBBox-like object for testing."""
    bbox = MagicMock()
    bbox.x_center = cx
    bbox.y_center = cy
    bbox.width = w
    bbox.height = h
    bbox.rotation = rot
    return bbox


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


class TestBboxCorners:
    def test_axis_aligned(self):
        bbox = _make_bbox(0.0, 0.0, 4.0, 2.0, rot=0.0)
        corners = _bbox_corners(bbox)
        assert corners.shape == (4, 2)
        xs = sorted(corners[:, 0])
        ys = sorted(corners[:, 1])
        assert abs(xs[0] - (-2.0)) < 1e-9
        assert abs(xs[-1] - 2.0) < 1e-9
        assert abs(ys[0] - (-1.0)) < 1e-9
        assert abs(ys[-1] - 1.0) < 1e-9

    def test_90_degree_rotation_swaps_dimensions(self):
        bbox = _make_bbox(0.0, 0.0, 4.0, 2.0, rot=math.pi / 2)
        corners = _bbox_corners(bbox)
        xs = sorted(corners[:, 0])
        ys = sorted(corners[:, 1])
        # width and height are swapped after 90° rotation
        assert abs(xs[-1] - xs[0] - 2.0) < 1e-6
        assert abs(ys[-1] - ys[0] - 4.0) < 1e-6


class TestPolygonArea:
    def test_unit_square(self):
        import numpy as np
        pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        assert abs(_polygon_area(pts) - 1.0) < 1e-9

    def test_fewer_than_3_points(self):
        import numpy as np
        pts = np.array([[0, 0], [1, 0]], dtype=float)
        assert _polygon_area(pts) == 0.0


class TestIntersectionArea:
    def test_identical_unit_squares(self):
        """Intersection of a box with itself should equal its area."""
        import numpy as np
        sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
        area = _intersection_area(sq, sq)
        assert abs(area - 4.0) < 1e-9

    def test_non_overlapping(self):
        import numpy as np
        a = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        b = np.array([[2, 0], [3, 0], [3, 1], [2, 1]], dtype=float)
        assert _intersection_area(a, b) == 0.0

    def test_partial_overlap(self):
        """Two 2x2 squares, shifted by 1 horizontally — overlap is a 1x2 rectangle."""
        import numpy as np
        a = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
        b = np.array([[1, 0], [3, 0], [3, 2], [1, 2]], dtype=float)
        area = _intersection_area(a, b)
        assert abs(area - 2.0) < 1e-9


class TestBboxesOverlap:
    def test_completely_separate(self):
        a = _make_bbox(0.0, 0.0, 2.0, 2.0)
        b = _make_bbox(10.0, 0.0, 2.0, 2.0)
        assert not _bboxes_overlap(a, b, 0.5)

    def test_identical_boxes_fully_overlap(self):
        a = _make_bbox(0.0, 0.0, 2.0, 2.0)
        b = _make_bbox(0.0, 0.0, 2.0, 2.0)
        assert _bboxes_overlap(a, b, 0.99)

    def test_half_overlap(self):
        """Shift by exactly half the width — 50% of smaller box is covered."""
        a = _make_bbox(0.0, 0.0, 2.0, 2.0)  # area = 4, extends -1..1 in x
        b = _make_bbox(1.0, 0.0, 2.0, 2.0)  # extends 0..2 in x
        # intersection is 1x2=2; min_area=4; ratio=0.5 → 50% exactly
        assert _bboxes_overlap(a, b, 0.5)
        assert not _bboxes_overlap(a, b, 0.51)


# ---------------------------------------------------------------------------
# Ghost detection
# ---------------------------------------------------------------------------


class TestDetectGhostFrames:
    def _make_controller(self, object_frames: dict[str, list[int]]):
        ctrl = MagicMock()
        ctrl.list_object_ids.return_value = list(object_frames.keys())
        ctrl.frames_for_object.side_effect = lambda obj_id: object_frames[obj_id]
        return ctrl

    def test_objects_with_few_frames_are_flagged(self):
        ctrl = self._make_controller({"a": [0, 1, 2], "b": [5, 6, 7, 8, 9, 10]})
        result = detect_ghost_frames(ctrl, max_ghost_frames=5)
        assert result == {0, 1, 2}

    def test_objects_at_boundary_are_included(self):
        ctrl = self._make_controller({"a": [0, 1, 2, 3, 4]})
        result = detect_ghost_frames(ctrl, max_ghost_frames=5)
        assert result == {0, 1, 2, 3, 4}

    def test_objects_above_threshold_not_flagged(self):
        ctrl = self._make_controller({"a": [0, 1, 2, 3, 4, 5]})
        result = detect_ghost_frames(ctrl, max_ghost_frames=5)
        assert result == set()

    def test_multiple_ghost_objects_union(self):
        ctrl = self._make_controller({"a": [0, 1], "b": [3, 4]})
        result = detect_ghost_frames(ctrl, max_ghost_frames=5)
        assert result == {0, 1, 3, 4}


# ---------------------------------------------------------------------------
# Double detection
# ---------------------------------------------------------------------------


class TestDetectDoubleFrames:
    def _make_annotation_controller(self, frame_bboxes: dict[int, dict[str, object]]):
        """frame_bboxes: {frame_idx: {obj_id: RotatedBBox-like}}"""
        all_ids = set()
        for bboxes in frame_bboxes.values():
            all_ids.update(bboxes.keys())
        ctrl = MagicMock()
        ctrl.list_object_ids.return_value = sorted(all_ids)

        def _try_get(frame_idx, obj_id):
            return frame_bboxes.get(frame_idx, {}).get(obj_id, None)

        ctrl.try_get_bbox.side_effect = _try_get
        return ctrl

    def _make_state_controller(self, frame_count: int):
        ctrl = MagicMock()
        ctrl.get_frame_count.return_value = frame_count
        return ctrl

    def test_no_overlapping_boxes(self):
        a = _make_bbox(0.0, 0.0, 2.0, 2.0)
        b = _make_bbox(10.0, 0.0, 2.0, 2.0)
        ann = self._make_annotation_controller({0: {"obj1": a, "obj2": b}})
        state = self._make_state_controller(1)
        result = detect_double_frames(ann, state, 0.5)
        assert result == set()

    def test_overlapping_boxes_flagged(self):
        a = _make_bbox(0.0, 0.0, 2.0, 2.0)
        b = _make_bbox(0.0, 0.0, 2.0, 2.0)
        ann = self._make_annotation_controller({0: {"obj1": a, "obj2": b}})
        state = self._make_state_controller(1)
        result = detect_double_frames(ann, state, 0.5)
        assert 0 in result

    def test_only_problematic_frames_flagged(self):
        a = _make_bbox(0.0, 0.0, 2.0, 2.0)
        b = _make_bbox(0.0, 0.0, 2.0, 2.0)
        far = _make_bbox(100.0, 100.0, 2.0, 2.0)
        ann = self._make_annotation_controller(
            {0: {"obj1": a, "obj2": far}, 1: {"obj1": a, "obj2": b}}
        )
        state = self._make_state_controller(2)
        result = detect_double_frames(ann, state, 0.5)
        assert result == {1}


# ---------------------------------------------------------------------------
# run_inspection integration
# ---------------------------------------------------------------------------


class TestRunInspection:
    def test_returns_sorted_union(self):
        ann = MagicMock()
        ann.list_object_ids.return_value = ["obj1"]
        ann.frames_for_object.return_value = [5, 6]  # ghost: 2 frames <= 5
        ann.try_get_bbox.return_value = None

        state = MagicMock()
        state.get_frame_count.return_value = 10

        result = run_inspection(ann, state, max_ghost_frames=5, overlap_percent=50.0)
        assert result["ghost_frames"] == {5, 6}
        assert result["double_frames"] == set()
        assert result["all_frames"] == sorted({5, 6})

    def test_all_frames_is_sorted_list(self):
        ann = MagicMock()
        ann.list_object_ids.return_value = []
        ann.try_get_bbox.return_value = None
        state = MagicMock()
        state.get_frame_count.return_value = 0

        result = run_inspection(ann, state, max_ghost_frames=5, overlap_percent=50.0)
        assert result["all_frames"] == []
        assert isinstance(result["all_frames"], list)
