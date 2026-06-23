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
) -> set[int]:
    """Return frame indices within [start_frame, end_frame] belonging to ghost objects.

    A ghost object is one whose total annotated frame count is <= max_ghost_frames.
    Only frames within the specified range are returned as problem frames.
    """
    object_ids = annotation_controller.list_object_ids()
    problem_frames: set[int] = set()
    for obj_id in object_ids:
        frames = annotation_controller.frames_for_object(obj_id)
        if len(frames) <= max_ghost_frames:
            for f in frames:
                if f < start_frame:
                    continue
                if end_frame is not None and f > end_frame:
                    continue
                problem_frames.add(f)
    return problem_frames


def detect_double_frames(
    annotation_controller: "AnnotationController",
    project_state_controller: "ProjectStateController",
    overlap_threshold: float,
    start_frame: int = 0,
    end_frame: int | None = None,
    progress_callback=None,
) -> set[int]:
    """Return frame indices within [start_frame, end_frame] where two bboxes overlap.

    progress_callback, if provided, is called with (frames_scanned, total_to_scan)
    after each frame is processed.
    """
    frame_count = project_state_controller.get_frame_count()
    if frame_count <= 0:
        return set()

    effective_end = min(
        frame_count - 1, end_frame if end_frame is not None else frame_count - 1
    )
    effective_start = max(0, start_frame)
    if effective_start > effective_end:
        return set()

    frames_to_scan = range(effective_start, effective_end + 1)
    total = len(frames_to_scan)
    object_ids = annotation_controller.list_object_ids()
    problem_frames: set[int] = set()

    for i, frame_idx in enumerate(frames_to_scan):
        if progress_callback is not None:
            progress_callback(i, total)
        bboxes = []
        for obj_id in object_ids:
            val = annotation_controller.try_get_bbox(frame_idx, obj_id)
            if val is not None:
                bboxes.append(val)
        if len(bboxes) < 2:
            continue
        found = False
        for a in range(len(bboxes)):
            if found:
                break
            for b in range(a + 1, len(bboxes)):
                if _bboxes_overlap(bboxes[a], bboxes[b], overlap_threshold):
                    problem_frames.add(frame_idx)
                    found = True
                    break
    return problem_frames


def run_inspection(
    annotation_controller: "AnnotationController",
    project_state_controller: "ProjectStateController",
    max_ghost_frames: int,
    overlap_percent: float,
    start_frame: int = 0,
    end_frame: int | None = None,
    progress_callback=None,
) -> dict:
    """Run all detectors and return a dict with ghost_frames, double_frames, all_frames.

    start_frame / end_frame restrict detection to that inclusive frame range.
    progress_callback is forwarded to detect_double_frames and called with
    (frames_scanned, total_to_scan) for each frame processed.
    """
    overlap_threshold = overlap_percent / 100.0
    ghost_frames = detect_ghost_frames(
        annotation_controller, max_ghost_frames, start_frame=start_frame,
        end_frame=end_frame,
    )
    double_frames = detect_double_frames(
        annotation_controller,
        project_state_controller,
        overlap_threshold,
        start_frame=start_frame,
        end_frame=end_frame,
        progress_callback=progress_callback,
    )
    all_frames = sorted(ghost_frames | double_frames)
    return {
        "ghost_frames": ghost_frames,
        "double_frames": double_frames,
        "all_frames": all_frames,
    }
