# edit/services/inspection_service.py
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from edit.controllers.annotation_controller import AnnotationController
    from edit.controllers.project_state_controller import ProjectStateController


def _bbox_corners(val) -> np.ndarray:
    """Return the 4 corners of a rotated bounding box as a (4, 2) float64 array.

    val must expose .cx, .cy, .width, .height, .rotation (radians).
    """
    cx, cy = float(val.x_center), float(val.y_center)
    w, h = float(val.width) / 2.0, float(val.height) / 2.0
    angle = float(val.rotation)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    # local corners: (-w,-h), (w,-h), (w,h), (-w,h)
    offsets = np.array([[-w, -h], [w, -h], [w, h], [-w, h]], dtype=np.float64)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
    return (offsets @ rot.T) + np.array([cx, cy], dtype=np.float64)


def _polygon_area(pts: np.ndarray) -> float:
    """Signed area via the shoelace formula. Returns the absolute value."""
    n = len(pts)
    if n < 3:
        return 0.0
    xs, ys = pts[:, 0], pts[:, 1]
    return abs(
        float(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))) * 0.5
    )


def _clip_polygon_by_half_plane(
    polygon: list, edge_p1: np.ndarray, edge_p2: np.ndarray
) -> list:
    """Sutherland-Hodgman: clip *polygon* against the half-plane left of p1→p2."""
    if not polygon:
        return []
    output = []
    n = len(polygon)
    for i in range(n):
        cur = polygon[i]
        prv = polygon[(i - 1) % n]
        # cross product sign: >= 0 means cur is on the left (inside) side

        def _inside(p):
            return (
                (edge_p2[0] - edge_p1[0]) * (p[1] - edge_p1[1])
                - (edge_p2[1] - edge_p1[1]) * (p[0] - edge_p1[0])
            ) >= 0.0

        if _inside(cur):
            if not _inside(prv):
                # compute intersection of prv→cur with the edge
                d_cur = np.array(cur, dtype=np.float64)
                d_prv = np.array(prv, dtype=np.float64)
                d1 = d_cur - d_prv
                d2 = edge_p2 - edge_p1
                denom = d1[0] * d2[1] - d1[1] * d2[0]
                if abs(denom) > 1e-12:
                    t = (
                        (edge_p1[0] - d_prv[0]) * d2[1]
                        - (edge_p1[1] - d_prv[1]) * d2[0]
                    ) / denom
                    intersection = d_prv + t * d1
                    output.append(intersection.tolist())
            output.append(cur)
        elif _inside(prv):
            d_cur = np.array(cur, dtype=np.float64)
            d_prv = np.array(prv, dtype=np.float64)
            d1 = d_cur - d_prv
            d2 = edge_p2 - edge_p1
            denom = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(denom) > 1e-12:
                t = (
                    (edge_p1[0] - d_prv[0]) * d2[1]
                    - (edge_p1[1] - d_prv[1]) * d2[0]
                ) / denom
                intersection = d_prv + t * d1
                output.append(intersection.tolist())
    return output


def _intersection_area(corners_a: np.ndarray, corners_b: np.ndarray) -> float:
    """Compute the area of intersection of two convex polygons (Sutherland-Hodgman)."""
    subject = corners_a.tolist()
    clip = corners_b
    n = len(clip)
    for i in range(n):
        p1 = clip[i]
        p2 = clip[(i + 1) % n]
        subject = _clip_polygon_by_half_plane(subject, p1, p2)
        if not subject:
            return 0.0
    if len(subject) < 3:
        return 0.0
    return _polygon_area(np.array(subject, dtype=np.float64))


def _bboxes_overlap(val_a, val_b, threshold: float) -> bool:
    """Return True if the intersection / min(area_a, area_b) >= threshold."""
    corners_a = _bbox_corners(val_a)
    corners_b = _bbox_corners(val_b)
    area_a = _polygon_area(corners_a)
    area_b = _polygon_area(corners_b)
    min_area = min(area_a, area_b)
    if min_area <= 0.0:
        return False
    inter = _intersection_area(corners_a, corners_b)
    return (inter / min_area) >= threshold


def detect_ghost_frames(
    annotation_controller: "AnnotationController",
    max_ghost_frames: int,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> dict[str, list[int]]:
    """Return ghost objects mapped to their problem frame indices.

    A ghost object is one whose total annotated frame count is <= max_ghost_frames.
    Only frames within [start_frame, end_frame] are included.
    Returns {object_id: [frame_indices]}.
    """
    object_ids = annotation_controller.list_object_ids()
    result: dict[str, list[int]] = {}
    for obj_id in object_ids:
        frames = annotation_controller.frames_for_object(obj_id)
        if len(frames) <= max_ghost_frames:
            in_range = [
                f for f in frames
                if f >= start_frame and (end_frame is None or f <= end_frame)
            ]
            if in_range:
                result[obj_id] = sorted(in_range)
    return result


def detect_double_frames(
    annotation_controller: "AnnotationController",
    project_state_controller: "ProjectStateController",
    overlap_threshold: float,
    start_frame: int = 0,
    end_frame: int | None = None,
    progress_callback=None,
) -> dict[int, list[str]]:
    """Return frames where two bboxes overlap, mapped to the involved object IDs.

    progress_callback, if provided, is called with (frames_scanned, total_to_scan)
    after each frame is processed.
    Returns {frame_idx: [obj_id_a, obj_id_b, ...]}.
    """
    frame_count = project_state_controller.get_frame_count()
    if frame_count <= 0:
        return {}

    effective_end = min(
        frame_count - 1, end_frame if end_frame is not None else frame_count - 1
    )
    effective_start = max(0, start_frame)
    if effective_start > effective_end:
        return {}

    frames_to_scan = range(effective_start, effective_end + 1)
    total = len(frames_to_scan)
    object_ids = annotation_controller.list_object_ids()
    result: dict[int, list[str]] = {}

    for i, frame_idx in enumerate(frames_to_scan):
        if progress_callback is not None:
            progress_callback(i, total)
        # Build list of (obj_id, bbox) for objects present in this frame
        present: list[tuple[str, object]] = []
        for obj_id in object_ids:
            val = annotation_controller.try_get_bbox(frame_idx, obj_id)
            if val is not None:
                present.append((obj_id, val))
        if len(present) < 2:
            continue
        involved: set[str] = set()
        for a in range(len(present)):
            for b in range(a + 1, len(present)):
                if _bboxes_overlap(present[a][1], present[b][1], overlap_threshold):
                    involved.add(present[a][0])
                    involved.add(present[b][0])
        if involved:
            result[frame_idx] = sorted(involved)
    return result


def run_inspection(
    annotation_controller: "AnnotationController",
    project_state_controller: "ProjectStateController",
    max_ghost_frames: int,
    overlap_percent: float,
    start_frame: int = 0,
    end_frame: int | None = None,
    progress_callback=None,
) -> dict:
    """Run all detectors and return ghost_detections, double_detections, all_frames.

    start_frame / end_frame restrict detection to that inclusive frame range.
    progress_callback is forwarded to detect_double_frames.
    Returns:
      {
        "ghost_detections": dict[str, list[int]],   # obj_id → frames
        "double_detections": dict[int, list[str]],  # frame_idx → obj_ids
        "all_frames": list[int],
      }
    """
    overlap_threshold = overlap_percent / 100.0
    ghost_detections = detect_ghost_frames(
        annotation_controller, max_ghost_frames,
        start_frame=start_frame, end_frame=end_frame,
    )
    double_detections = detect_double_frames(
        annotation_controller,
        project_state_controller,
        overlap_threshold,
        start_frame=start_frame,
        end_frame=end_frame,
        progress_callback=progress_callback,
    )
    ghost_frames: set[int] = {f for frames in ghost_detections.values() for f in frames}
    double_frames: set[int] = set(double_detections.keys())
    all_frames = sorted(ghost_frames | double_frames)
    return {
        "ghost_detections": ghost_detections,
        "double_detections": double_detections,
        "all_frames": all_frames,
    }
