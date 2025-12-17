"""
geometry - Geometric calculations and utilities

Contains IoU calculation and polygon clipping algorithms for oriented bounding boxes.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class BBoxOverlapCalculator:
    """Optimized utility class for calculating IoU between oriented bounding boxes."""

    @staticmethod
    def calculate_intersection_over_union(
        bbox1: np.ndarray, bbox2: np.ndarray
    ) -> float:
        """Calculate IoU between two oriented bounding boxes using OpenCV.

        Args:
            bbox1: First OBB as 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            bbox2: Second OBB as 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

        Returns:
            IoU value between 0 and 1
        """
        try:
            # Ensure correct format
            pts1 = np.array(bbox1, dtype=np.float32).reshape(-1, 2)
            pts2 = np.array(bbox2, dtype=np.float32).reshape(-1, 2)

            # Calculate areas using the shoelace formula
            area1 = BBoxOverlapCalculator._polygon_area(pts1)
            area2 = BBoxOverlapCalculator._polygon_area(pts2)

            if area1 <= 0 or area2 <= 0:
                return 0.0

            # Find intersection using Sutherland-Hodgman clipping
            intersection_points = BBoxOverlapCalculator._sutherland_hodgman_clip(
                pts1, pts2
            )

            if len(intersection_points) < 3:
                return 0.0  # No meaningful intersection

            intersection_area = BBoxOverlapCalculator._polygon_area(intersection_points)

            if intersection_area <= 0:
                return 0.0

            # Calculate IoU
            union_area = area1 + area2 - intersection_area

            if union_area <= 0:
                return 0.0

            return intersection_area / union_area

        except Exception as e:
            logger.debug(f"IoU calculation failed: {e}")
            return 0.0

    @staticmethod
    def _polygon_area(points: np.ndarray) -> float:
        """Calculate polygon area using the shoelace formula.

        Args:
            points: Array of polygon vertices [[x1,y1], [x2,y2], ...]

        Returns:
            Polygon area
        """
        if len(points) < 3:
            return 0.0

        x = points[:, 0]
        y = points[:, 1]

        # Shoelace formula
        area = 0.5 * abs(
            sum(
                x[i] * y[(i + 1) % len(x)] - x[(i + 1) % len(x)] * y[i]
                for i in range(len(x))
            )
        )
        return area

    @staticmethod
    def _sutherland_hodgman_clip(
        subject_polygon: np.ndarray, clip_polygon: np.ndarray
    ) -> np.ndarray:
        """Clip subject polygon against clip polygon using Sutherland-Hodgman algorithm.

        Args:
            subject_polygon: Polygon to be clipped
            clip_polygon: Clipping polygon

        Returns:
            Intersection polygon vertices
        """

        def _is_inside(
            point: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray
        ) -> bool:
            """Check if point is inside the edge (left side of directed edge)."""
            edge_vec = edge_end - edge_start
            point_vec = point - edge_start
            return (edge_vec[0] * point_vec[1] - edge_vec[1] * point_vec[0]) >= 0

        def _intersection_point(
            p1: np.ndarray, p2: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray
        ) -> np.ndarray:
            """Find intersection point between line segment p1-p2 and edge."""
            d1 = edge_end - edge_start
            d2 = p2 - p1

            denominator = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(denominator) < 1e-10:
                return p1  # Lines are parallel

            p1_vec = p1 - edge_start
            t = (p1_vec[0] * d2[1] - p1_vec[1] * d2[0]) / denominator
            return edge_start + t * d1

        output_list = subject_polygon.tolist()

        # Process each edge of the clipping polygon
        for i in range(len(clip_polygon)):
            if not output_list:
                break

            edge_start = clip_polygon[i]
            edge_end = clip_polygon[(i + 1) % len(clip_polygon)]

            input_list = output_list
            output_list = []

            if not input_list:
                continue

            if len(input_list) > 0:
                s = np.array(input_list[-1])

                for vertex in input_list:
                    e = np.array(vertex)

                    if _is_inside(e, edge_start, edge_end):
                        if not _is_inside(s, edge_start, edge_end):
                            # Entering the clipping area
                            intersection = _intersection_point(
                                s, e, edge_start, edge_end
                            )
                            output_list.append(intersection.tolist())
                        output_list.append(e.tolist())
                    elif _is_inside(s, edge_start, edge_end):
                        # Leaving the clipping area
                        intersection = _intersection_point(s, e, edge_start, edge_end)
                        output_list.append(intersection.tolist())

                    s = e

        return np.array(output_list) if output_list else np.array([])
