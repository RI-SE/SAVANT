"""Geometric calculations and utilities.

Contains overlap (IoU / IoMin) calculation for oriented bounding boxes.
"""

import logging
import math
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BBoxOverlapCalculator:
    """Overlap metrics for oriented bounding boxes given as corner points.

    Uses OpenCV's convex polygon intersection, which accepts corners in either winding
    order (the previous Sutherland-Hodgman code silently assumed counter-clockwise).
    """

    @staticmethod
    def overlap_areas(bbox1: np.ndarray, bbox2: np.ndarray) -> Tuple[float, float, float]:
        """Return (intersection_area, area1, area2); intersection is 0 for degenerate boxes."""
        try:
            pts1 = np.asarray(bbox1, dtype=np.float32).reshape(-1, 2)
            pts2 = np.asarray(bbox2, dtype=np.float32).reshape(-1, 2)
            area1 = BBoxOverlapCalculator._polygon_area(pts1)
            area2 = BBoxOverlapCalculator._polygon_area(pts2)
            if area1 <= 0 or area2 <= 0:
                return 0.0, area1, area2
            intersection, _ = cv2.intersectConvexConvex(pts1, pts2)
            return max(float(intersection), 0.0), area1, area2
        except (cv2.error, ValueError) as e:
            logger.debug(f"Overlap calculation failed: {e}")
            return 0.0, 0.0, 0.0

    @staticmethod
    def calculate_iou_and_iomin(bbox1: np.ndarray, bbox2: np.ndarray) -> Tuple[float, float]:
        """Return (IoU, intersection over the smaller area) from one intersection."""
        intersection, area1, area2 = BBoxOverlapCalculator.overlap_areas(bbox1, bbox2)
        if intersection <= 0:
            return 0.0, 0.0
        union = area1 + area2 - intersection
        iou = intersection / union if union > 0 else 0.0
        return iou, intersection / min(area1, area2)

    @staticmethod
    def calculate_intersection_over_union(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """IoU between two oriented boxes (4 corner points each), in [0, 1]."""
        return BBoxOverlapCalculator.calculate_iou_and_iomin(bbox1, bbox2)[0]

    @staticmethod
    def calculate_intersection_over_min(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Intersection over the smaller box's area; detects containment that IoU hides."""
        return BBoxOverlapCalculator.calculate_iou_and_iomin(bbox1, bbox2)[1]

    @staticmethod
    def _polygon_area(points: np.ndarray) -> float:
        """Absolute polygon area (0 for fewer than 3 points)."""
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 3:
            return 0.0
        return float(cv2.contourArea(pts))


def bbox_to_corners(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    theta: float,
) -> np.ndarray:
    """Convert center-based bbox to corner points.

    Args:
        center_x: Center x coordinate
        center_y: Center y coordinate
        width: Box width
        height: Box height
        theta: Rotation angle in radians

    Returns:
        Array of 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    half_w = width / 2
    half_h = height / 2

    # Local corner offsets (counter-clockwise from top-left)
    local_corners = [
        (-half_w, -half_h),
        (half_w, -half_h),
        (half_w, half_h),
        (-half_w, half_h),
    ]

    corners = []
    for lx, ly in local_corners:
        # Rotate and translate
        rx = lx * cos_t - ly * sin_t + center_x
        ry = lx * sin_t + ly * cos_t + center_y
        corners.append([rx, ry])

    return np.array(corners, dtype=np.float32)


def rotated_to_axis_aligned(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    theta: float,
) -> Tuple[float, float, float, float]:
    """Convert a rotated bbox to its axis-aligned bounding rect.

    Args:
        center_x: Center x coordinate
        center_y: Center y coordinate
        width: Box width
        height: Box height
        theta: Rotation angle in radians

    Returns:
        Tuple of (x, y, w, h) for axis-aligned bounding rect (x, y is top-left)
    """
    corners = bbox_to_corners(center_x, center_y, width, height, theta)

    min_x = corners[:, 0].min()
    max_x = corners[:, 0].max()
    min_y = corners[:, 1].min()
    max_y = corners[:, 1].max()

    return (min_x, min_y, max_x - min_x, max_y - min_y)
