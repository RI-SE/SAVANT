"""
Unit tests for geometry module - IoU calculations and polygon operations.
"""

import numpy as np
import pytest

from markit.markitlib.geometry import BBoxOverlapCalculator


class TestBBoxOverlapCalculator:
    """Tests for BBoxOverlapCalculator class."""

    def test_identical_bboxes_return_iou_one(self, sample_obb_bbox):
        """Identical bounding boxes should have IoU = 1.0."""
        iou = BBoxOverlapCalculator.calculate_intersection_over_union(
            sample_obb_bbox, sample_obb_bbox
        )
        assert np.isclose(iou, 1.0, atol=1e-6)

    def test_non_overlapping_bboxes_return_iou_zero(self):
        """Non-overlapping bounding boxes should have IoU = 0.0."""
        bbox1 = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0]
        ])
        bbox2 = np.array([
            [20.0, 20.0],
            [30.0, 20.0],
            [30.0, 30.0],
            [20.0, 30.0]
        ])
        iou = BBoxOverlapCalculator.calculate_intersection_over_union(bbox1, bbox2)
        assert iou == 0.0

    def test_half_overlapping_bboxes(self):
        """Test IoU calculation for 50% overlap."""
        bbox1 = np.array([
            [0.0, 0.0],
            [20.0, 0.0],
            [20.0, 10.0],
            [0.0, 10.0]
        ])
        bbox2 = np.array([
            [10.0, 0.0],
            [30.0, 0.0],
            [30.0, 10.0],
            [10.0, 10.0]
        ])
        iou = BBoxOverlapCalculator.calculate_intersection_over_union(bbox1, bbox2)
        # Area of bbox1: 200, Area of bbox2: 200
        # Intersection area: 100 (10x10)
        # Union area: 200 + 200 - 100 = 300
        # IoU = 100/300 = 0.333...
        expected_iou = 100.0 / 300.0
        assert np.isclose(iou, expected_iou, atol=1e-6)

    def test_nested_bbox_smaller_inside_larger(self):
        """Test IoU when smaller bbox is completely inside larger one."""
        bbox_large = np.array([
            [0.0, 0.0],
            [100.0, 0.0],
            [100.0, 100.0],
            [0.0, 100.0]
        ])
        bbox_small = np.array([
            [25.0, 25.0],
            [75.0, 25.0],
            [75.0, 75.0],
            [25.0, 75.0]
        ])
        iou = BBoxOverlapCalculator.calculate_intersection_over_union(bbox_large, bbox_small)
        # Area of large: 10000, Area of small: 2500
        # Intersection area: 2500 (entire small bbox)
        # Union area: 10000 + 2500 - 2500 = 10000
        # IoU = 2500/10000 = 0.25
        expected_iou = 2500.0 / 10000.0
        assert np.isclose(iou, expected_iou, atol=1e-6)

    def test_rotated_bbox_with_itself(self, sample_rotated_bbox):
        """Rotated bbox with itself should have IoU = 1.0."""
        iou = BBoxOverlapCalculator.calculate_intersection_over_union(
            sample_rotated_bbox, sample_rotated_bbox
        )
        assert np.isclose(iou, 1.0, atol=1e-6)

    def test_rotated_bbox_partial_overlap(self):
        """Test IoU for rotated bounding boxes with partial overlap."""
        # Axis-aligned bbox
        bbox1 = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0]
        ])
        # Rotated 45 degrees around origin, shifted to overlap
        angle = np.pi / 4
        size = 10.0
        center_x, center_y = 5.0, 5.0

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        half_diag = size / np.sqrt(2)

        bbox2 = np.array([
            [center_x - half_diag, center_y],
            [center_x, center_y - half_diag],
            [center_x + half_diag, center_y],
            [center_x, center_y + half_diag]
        ])

        iou = BBoxOverlapCalculator.calculate_intersection_over_union(bbox1, bbox2)
        # Should have some overlap but not complete
        assert 0.0 < iou < 1.0

    def test_polygon_area_square(self):
        """Test polygon area calculation for a square."""
        square = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0]
        ])
        area = BBoxOverlapCalculator._polygon_area(square)
        assert np.isclose(area, 100.0, atol=1e-6)

    def test_polygon_area_rectangle(self):
        """Test polygon area calculation for a rectangle."""
        rectangle = np.array([
            [0.0, 0.0],
            [20.0, 0.0],
            [20.0, 5.0],
            [0.0, 5.0]
        ])
        area = BBoxOverlapCalculator._polygon_area(rectangle)
        assert np.isclose(area, 100.0, atol=1e-6)

    def test_polygon_area_triangle(self):
        """Test polygon area calculation for a triangle."""
        triangle = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [5.0, 10.0]
        ])
        area = BBoxOverlapCalculator._polygon_area(triangle)
        assert np.isclose(area, 50.0, atol=1e-6)

    def test_polygon_area_empty_returns_zero(self):
        """Empty polygon should return zero area."""
        empty = np.array([])
        area = BBoxOverlapCalculator._polygon_area(empty)
        assert area == 0.0

    def test_polygon_area_single_point_returns_zero(self):
        """Single point should return zero area."""
        point = np.array([[5.0, 5.0]])
        area = BBoxOverlapCalculator._polygon_area(point)
        assert area == 0.0

    def test_invalid_bbox_format_returns_zero(self):
        """Invalid bbox format should return IoU = 0.0."""
        bbox1 = np.array([[0.0, 0.0]])  # Invalid: only 1 point
        bbox2 = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0]
        ])
        iou = BBoxOverlapCalculator.calculate_intersection_over_union(bbox1, bbox2)
        assert iou == 0.0

    def test_zero_area_bbox_returns_zero(self):
        """Bbox with zero area should return IoU = 0.0."""
        bbox1 = np.array([
            [5.0, 5.0],
            [5.0, 5.0],
            [5.0, 5.0],
            [5.0, 5.0]
        ])  # All points the same (zero area)
        bbox2 = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0]
        ])
        iou = BBoxOverlapCalculator.calculate_intersection_over_union(bbox1, bbox2)
        assert iou == 0.0

    def test_sutherland_hodgman_clip_complete_overlap(self):
        """Test clipping when subject is completely inside clip polygon."""
        subject = np.array([
            [2.0, 2.0],
            [8.0, 2.0],
            [8.0, 8.0],
            [2.0, 8.0]
        ])
        clip = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0]
        ])
        result = BBoxOverlapCalculator._sutherland_hodgman_clip(subject, clip)
        # Result should be approximately the subject polygon
        assert len(result) == 4
        result_area = BBoxOverlapCalculator._polygon_area(result)
        subject_area = BBoxOverlapCalculator._polygon_area(subject)
        assert np.isclose(result_area, subject_area, atol=1e-6)

    def test_sutherland_hodgman_clip_no_overlap(self):
        """Test clipping when polygons don't overlap."""
        subject = np.array([
            [0.0, 0.0],
            [5.0, 0.0],
            [5.0, 5.0],
            [0.0, 5.0]
        ])
        clip = np.array([
            [10.0, 10.0],
            [20.0, 10.0],
            [20.0, 20.0],
            [10.0, 20.0]
        ])
        result = BBoxOverlapCalculator._sutherland_hodgman_clip(subject, clip)
        # Result should be empty or have very small area
        assert len(result) == 0 or BBoxOverlapCalculator._polygon_area(result) < 1e-6
