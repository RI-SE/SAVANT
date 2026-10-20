"""Tests for converting annotation boxes to YOLO OBB label corners."""

import math

import cv2
import numpy as np
import pytest

from trainit.extract_yolo_training import YOLODatasetExtractor

W, H = 200, 100
to_corners = YOLODatasetExtractor._bbox_to_corners


def _pixels(corners):
    return (np.array(corners, np.float32).reshape(4, 2) * np.float32([W, H])).astype(np.float32)


def _trained_box(pts):
    """Ultralytics fits a min-area rectangle to the 4 label points."""
    return cv2.boxPoints(cv2.minAreaRect(pts))


def test_inside_box_is_unchanged():
    pts = _pixels(to_corners(None, 100, 50, 40, 20, 0.3, W, H))
    assert math.isclose(cv2.contourArea(pts), 800, rel_tol=1e-3)


def test_edge_crossing_box_matches_visible_part_better_than_corner_clamping():
    from savant_common.geometry import BBoxOverlapCalculator, bbox_to_corners

    box = bbox_to_corners(190, 50, 40, 20, 0.4).astype(np.float32)
    image = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    visible = cv2.intersectConvexConvex(box, image)[1].reshape(-1, 2)
    pts = _pixels(to_corners(None, 190, 50, 40, 20, 0.4, W, H))
    clamped = np.clip(box, [0, 0], [W, H]).astype(np.float32)

    assert pts.min() >= 0 and (pts[:, 0] <= W).all() and (pts[:, 1] <= H).all()
    iou = BBoxOverlapCalculator.calculate_intersection_over_union
    assert iou(_trained_box(pts), visible) > iou(_trained_box(clamped), visible)


@pytest.mark.parametrize("cx, cy", [(-100, 50), (100, 500)])
def test_box_fully_outside_is_dropped(cx, cy):
    assert to_corners(None, cx, cy, 40, 20, 0.0, W, H) is None
