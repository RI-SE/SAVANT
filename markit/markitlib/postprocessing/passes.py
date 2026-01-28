"""
passes - Postprocessing pass implementations

Contains all postprocessing passes for gap detection/filling, duplicate removal,
rotation adjustment, sudden event detection, and frame interval calculation.
"""

import logging
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict

import numpy as np

from .base import PostprocessingPass
from ..geometry import BBoxOverlapCalculator
from ..utils import (
    normalize_angle_to_pi,
    normalize_angle_to_2pi_range,
    rebase_angle_if_needed,
)

logger = logging.getLogger(__name__)


class GapDetectionPass(PostprocessingPass):
    """Detect gaps in object ID frame sequences."""

    def __init__(self):
        self.gaps_detected = {}
        self.objects_with_gaps = set()

    def process(self, openlabel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect gaps in object tracking sequences.

        Args:
            openlabel_data: Complete OpenLabel data structure

        Returns:
            Unmodified OpenLabel data (detection only, no fixes yet)
        """
        frames = openlabel_data.get("openlabel", {}).get("frames", {})

        object_frames = defaultdict(list)

        for frame_idx_str, frame_data in frames.items():
            frame_idx = int(frame_idx_str)
            objects = frame_data.get("objects", {})

            for obj_id_str in objects.keys():
                object_frames[obj_id_str].append(frame_idx)

        for obj_id, frame_list in object_frames.items():
            if len(frame_list) < 2:
                continue

            frame_list_sorted = sorted(frame_list)
            gaps = []

            for i in range(len(frame_list_sorted) - 1):
                current_frame = frame_list_sorted[i]
                next_frame = frame_list_sorted[i + 1]
                gap_size = next_frame - current_frame - 1

                if gap_size > 0:
                    gaps.append(
                        {
                            "start_frame": current_frame,
                            "end_frame": next_frame,
                            "gap_size": gap_size,
                        }
                    )

            if gaps:
                self.gaps_detected[obj_id] = {
                    "frame_range": (frame_list_sorted[0], frame_list_sorted[-1]),
                    "total_frames": len(frame_list_sorted),
                    "gaps": gaps,
                }
                self.objects_with_gaps.add(obj_id)

                logger.warning(
                    f"Object ID {obj_id}: detected {len(gaps)} gap(s) in frame sequence "
                    f"[{frame_list_sorted[0]}-{frame_list_sorted[-1]}]"
                )
                for gap in gaps:
                    logger.warning(
                        f"  Gap: frames {gap['start_frame']} -> {gap['end_frame']} "
                        f"(missing {gap['gap_size']} frame(s))"
                    )

        return openlabel_data

    def get_statistics(self) -> Dict[str, Any]:
        """Get gap detection statistics.

        Returns:
            Dictionary with gap detection statistics
        """
        total_gaps = sum(len(info["gaps"]) for info in self.gaps_detected.values())

        return {
            "objects_with_gaps": len(self.objects_with_gaps),
            "total_gaps_detected": total_gaps,
            "gap_details": self.gaps_detected,
        }


class GapFillingPass(PostprocessingPass):
    """Fill gaps in object ID frame sequences by interpolating positions."""

    def __init__(self):
        self.gaps_filled = 0
        self.frames_added = 0
        self.objects_processed = set()

    def process(self, openlabel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fill gaps in object tracking sequences by interpolating positions.

        Args:
            openlabel_data: Complete OpenLabel data structure

        Returns:
            Modified OpenLabel data with gaps filled
        """
        frames = openlabel_data.get("openlabel", {}).get("frames", {})

        object_frames = defaultdict(list)

        for frame_idx_str, frame_data in frames.items():
            frame_idx = int(frame_idx_str)
            objects = frame_data.get("objects", {})

            for obj_id_str in objects.keys():
                object_frames[obj_id_str].append(frame_idx)

        for obj_id, frame_list in object_frames.items():
            if len(frame_list) < 2:
                continue

            frame_list_sorted = sorted(frame_list)

            for i in range(len(frame_list_sorted) - 1):
                frame_before = frame_list_sorted[i]
                frame_after = frame_list_sorted[i + 1]
                gap_size = frame_after - frame_before - 1

                if gap_size > 0:
                    self._fill_gap(
                        openlabel_data, obj_id, frame_before, frame_after, gap_size
                    )

        return openlabel_data

    def _fill_gap(
        self,
        openlabel_data: Dict[str, Any],
        obj_id: str,
        frame_before: int,
        frame_after: int,
        gap_size: int,
    ) -> None:
        """Fill a specific gap by interpolating object positions.

        Args:
            openlabel_data: OpenLabel data structure
            obj_id: Object ID string
            frame_before: Last frame before gap
            frame_after: First frame after gap
            gap_size: Number of missing frames
        """
        frames = openlabel_data["openlabel"]["frames"]

        obj_data_before = frames[str(frame_before)]["objects"][obj_id]["object_data"]
        obj_data_after = frames[str(frame_after)]["objects"][obj_id]["object_data"]

        rbbox_before = obj_data_before["rbbox"][0]["val"]
        rbbox_after = obj_data_after["rbbox"][0]["val"]

        x_before, y_before, w_before, h_before, r_before = rbbox_before
        x_after, y_after, w_after, h_after, r_after = rbbox_after

        # Calculate deltas for all parameters
        delta_x = x_after - x_before
        delta_y = y_after - y_before
        delta_w = w_after - w_before
        delta_h = h_after - h_before
        # Use shortest angular path for rotation interpolation
        delta_r = normalize_angle_to_pi(r_after - r_before)

        total_steps = gap_size + 1

        for step in range(1, gap_size + 1):
            interpolation_factor = step / total_steps

            # Interpolate all bbox parameters
            x_interpolated = x_before + delta_x * interpolation_factor
            y_interpolated = y_before + delta_y * interpolation_factor
            w_interpolated = w_before + delta_w * interpolation_factor
            h_interpolated = h_before + delta_h * interpolation_factor
            r_interpolated = r_before + delta_r * interpolation_factor

            missing_frame_idx = frame_before + step
            missing_frame_str = str(missing_frame_idx)

            if missing_frame_str not in frames:
                frames[missing_frame_str] = {"objects": {}}

            frames[missing_frame_str]["objects"][obj_id] = {
                "object_data": {
                    "rbbox": [
                        {
                            "name": "shape",
                            "val": [
                                x_interpolated,
                                y_interpolated,
                                w_interpolated,
                                h_interpolated,
                                r_interpolated,
                            ],
                        }
                    ],
                    "vec": [
                        {"name": "annotator", "val": ["markit_housekeeping(gap)"]},
                        {"name": "confidence", "val": [0.6666]},
                    ],
                }
            }

            self.frames_added += 1

        self.gaps_filled += 1
        self.objects_processed.add(obj_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get gap filling statistics.

        Returns:
            Dictionary with gap filling statistics
        """
        return {
            "objects_processed": len(self.objects_processed),
            "gaps_filled": self.gaps_filled,
            "frames_added": self.frames_added,
        }


class DuplicateRemovalPass(PostprocessingPass):
    """Remove duplicate bounding boxes based on IOU threshold."""

    def __init__(self, avg_iou_threshold: float = 0.7, min_iou_threshold: float = 0.3):
        self.objects_deleted = 0
        self.duplicate_pairs_found = 0
        self.frames_modified = 0
        self.iou_calculator = BBoxOverlapCalculator()
        self.deletion_details = []
        self.avg_iou_threshold = avg_iou_threshold
        self.min_iou_threshold = min_iou_threshold

    def process(self, openlabel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove duplicate objects based on IOU analysis.

        Args:
            openlabel_data: Complete OpenLabel data structure

        Returns:
            Modified OpenLabel data with duplicates removed
        """
        frames = openlabel_data.get("openlabel", {}).get("frames", {})
        objects = openlabel_data.get("openlabel", {}).get("objects", {})

        object_frame_map = defaultdict(list)

        for frame_idx_str, frame_data in frames.items():
            frame_idx = int(frame_idx_str)
            frame_objects = frame_data.get("objects", {})

            for obj_id_str in frame_objects.keys():
                object_frame_map[obj_id_str].append(frame_idx)

        objects_to_delete = set()
        object_ids = list(objects.keys())

        for i in range(len(object_ids)):
            for j in range(i + 1, len(object_ids)):
                obj_a = object_ids[i]
                obj_b = object_ids[j]

                if obj_a in objects_to_delete or obj_b in objects_to_delete:
                    continue

                if self._are_duplicates(obj_a, obj_b, object_frame_map, frames):
                    self.duplicate_pairs_found += 1

                    obj_to_delete = self._choose_object_to_delete(
                        obj_a, obj_b, object_frame_map, frames
                    )
                    obj_to_keep = obj_b if obj_to_delete == obj_a else obj_a
                    objects_to_delete.add(obj_to_delete)

                    frames_list = sorted(object_frame_map[obj_to_delete])
                    self.deletion_details.append(
                        {
                            "deleted_object": obj_to_delete,
                            "kept_object": obj_to_keep,
                            "frame_start": frames_list[0] if frames_list else None,
                            "frame_end": frames_list[-1] if frames_list else None,
                        }
                    )

        for obj_id in objects_to_delete:
            if obj_id in objects:
                del objects[obj_id]
                self.objects_deleted += 1

            for frame_idx_str, frame_data in frames.items():
                frame_objects = frame_data.get("objects", {})
                if obj_id in frame_objects:
                    del frame_objects[obj_id]
                    self.frames_modified += 1

        for detail in self.deletion_details:
            logger.info(
                f"Deleted object {detail['deleted_object']} (duplicate of {detail['kept_object']}) "
                f"from frames {detail['frame_start']}-{detail['frame_end']}"
            )

        return openlabel_data

    def _are_duplicates(
        self,
        obj_a: str,
        obj_b: str,
        object_frame_map: Dict[str, List[int]],
        frames: Dict[str, Any],
    ) -> bool:
        """Check if two objects are duplicates based on IOU thresholds.

        Args:
            obj_a: First object ID
            obj_b: Second object ID
            object_frame_map: Mapping of object IDs to frame lists
            frames: Frame data

        Returns:
            True if objects are duplicates (avg IOU > 0.8 and min IOU > 0.5)
        """
        frames_a = set(object_frame_map.get(obj_a, []))
        frames_b = set(object_frame_map.get(obj_b, []))
        shared_frames = frames_a.intersection(frames_b)

        if len(shared_frames) == 0:
            return False

        ious = []

        for frame_idx in shared_frames:
            frame_str = str(frame_idx)
            frame_data = frames[frame_str]
            frame_objects = frame_data.get("objects", {})

            bbox_a = self._extract_bbox(frame_objects[obj_a])
            bbox_b = self._extract_bbox(frame_objects[obj_b])

            if bbox_a is not None and bbox_b is not None:
                iou = self.iou_calculator.calculate_intersection_over_union(
                    bbox_a, bbox_b
                )
                ious.append(iou)

        if len(ious) == 0:
            return False

        avg_iou = sum(ious) / len(ious)
        min_iou = min(ious)

        return avg_iou > self.avg_iou_threshold and min_iou > self.min_iou_threshold

    def _extract_bbox(self, object_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract bounding box from object data and convert to corner points.

        Args:
            object_data: Object data containing rbbox

        Returns:
            4 corner points as numpy array, or None if extraction fails
        """
        try:
            rbbox_val = object_data["object_data"]["rbbox"][0]["val"]
            x, y, w, h, r = rbbox_val

            cos_r = np.cos(r)
            sin_r = np.sin(r)

            hw = w / 2
            hh = h / 2

            corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])

            rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            rotated_corners = corners @ rotation_matrix.T

            oriented_bbox = rotated_corners + np.array([x, y])

            return oriented_bbox.astype(np.float32)

        except (KeyError, IndexError, ValueError) as e:
            logger.debug(f"Failed to extract bbox: {e}")
            return None

    def _choose_object_to_delete(
        self,
        obj_a: str,
        obj_b: str,
        object_frame_map: Dict[str, List[int]],
        frames: Dict[str, Any],
    ) -> str:
        """Choose which object to delete from a duplicate pair.

        Args:
            obj_a: First object ID
            obj_b: Second object ID
            object_frame_map: Mapping of object IDs to frame lists
            frames: Frame data

        Returns:
            Object ID to delete
        """
        frames_a = len(object_frame_map.get(obj_a, []))
        frames_b = len(object_frame_map.get(obj_b, []))

        if frames_a != frames_b:
            return obj_a if frames_a < frames_b else obj_b

        conf_a = self._calculate_average_confidence(obj_a, object_frame_map, frames)
        conf_b = self._calculate_average_confidence(obj_b, object_frame_map, frames)

        return obj_a if conf_a < conf_b else obj_b

    def _calculate_average_confidence(
        self,
        obj_id: str,
        object_frame_map: Dict[str, List[int]],
        frames: Dict[str, Any],
    ) -> float:
        """Calculate average confidence for an object across all its frames.

        Args:
            obj_id: Object ID
            object_frame_map: Mapping of object IDs to frame lists
            frames: Frame data

        Returns:
            Average confidence value
        """
        confidences = []

        for frame_idx in object_frame_map.get(obj_id, []):
            frame_str = str(frame_idx)
            frame_data = frames[frame_str]
            frame_objects = frame_data.get("objects", {})

            if obj_id in frame_objects:
                try:
                    vec_list = frame_objects[obj_id]["object_data"]["vec"]
                    for vec_item in vec_list:
                        if vec_item.get("name") == "confidence":
                            conf_values = vec_item.get("val", [])
                            if conf_values:
                                confidences.append(conf_values[-1])
                            break
                except (KeyError, IndexError):
                    pass

        return sum(confidences) / len(confidences) if confidences else 0.0

    def get_statistics(self) -> Dict[str, Any]:
        """Get duplicate removal statistics.

        Returns:
            Dictionary with duplicate removal statistics
        """
        return {
            "objects_deleted": self.objects_deleted,
            "duplicate_pairs_found": self.duplicate_pairs_found,
            "frames_modified": self.frames_modified,
        }


class FirstDetectionRefinementPass(PostprocessingPass):
    """Refine initial angles for first detections using lookahead.

    This is a MANDATORY pass that improves initial angle estimates for newly
    detected objects by looking at their movement direction in subsequent frames.
    Falls back to base angle for stationary objects.
    """

    def __init__(self, lookahead_frames: int = 5, min_movement_pixels: float = 5.0):
        """Initialize first detection refinement pass.

        Args:
            lookahead_frames: Number of future frames to check for movement (default: 5)
            min_movement_pixels: Minimum movement to use for angle refinement (default: 5.0)
        """
        self.lookahead_frames = lookahead_frames
        self.min_movement_pixels = min_movement_pixels
        self.refined_objects: Set[int] = set()  # Track which objects have been refined
        self.objects_refined = 0
        self.objects_kept_base = 0

    def process(self, openlabel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Refine first detection angles using lookahead.

        Args:
            openlabel_data: Complete OpenLabel data structure

        Returns:
            Modified OpenLabel data with refined first detection angles
        """
        frames = openlabel_data.get("openlabel", {}).get("frames", {})

        # Build object frame map
        object_frame_map = defaultdict(dict)

        for frame_idx_str, frame_data in frames.items():
            frame_idx = int(frame_idx_str)
            frame_objects = frame_data.get("objects", {})

            for obj_id_str in frame_objects.keys():
                # Extract rbbox data for this frame
                rbbox = frame_objects[obj_id_str]["object_data"]["rbbox"][0]["val"]
                object_frame_map[obj_id_str][frame_idx] = rbbox

        # Iterate through all tracked objects
        for obj_id_str, frames_data in object_frame_map.items():
            obj_id = int(obj_id_str)

            # Skip if already refined
            if obj_id in self.refined_objects:
                continue

            # Find first frame where this object appears
            frame_indices = sorted(frames_data.keys())
            if not frame_indices:
                continue

            first_frame_idx = frame_indices[0]
            first_rbbox = frames_data[first_frame_idx]

            # Extract current position and dimensions
            _, _ = first_rbbox[0], first_rbbox[1]  # cx, cy - not used here
            w, h = first_rbbox[2], first_rbbox[3]
            base_angle = first_rbbox[4]

            # Try to find movement in future frames
            movement_dir = self._calculate_movement_direction(
                obj_id_str, first_frame_idx, frames_data
            )

            if movement_dir is not None:
                # Object is moving - refine angle based on movement
                aspect_ratio = max(w, h) / max(min(w, h), 1.0)

                if aspect_ratio > 1.5:
                    # Elongated object: align long axis (width) with movement
                    # Since w is semantic long axis, directly use movement direction
                    target_angle = movement_dir
                else:
                    # Circular/square object: use movement direction
                    target_angle = movement_dir

                # Find continuous angle closest to target
                # The base_angle might be off by k*π from true orientation
                angle_diff = target_angle - base_angle
                k = round(angle_diff / np.pi)
                refined_angle = base_angle + k * np.pi

                # Update the first detection
                first_rbbox[4] = refined_angle

                logger.info(
                    f"FirstDetection: obj {obj_id} base_angle={np.degrees(base_angle):.1f}° "
                    f"→ refined={np.degrees(refined_angle):.1f}° (movement={np.degrees(movement_dir):.1f}°)"
                )
                self.objects_refined += 1
            else:
                # No significant movement detected - keep base angle
                logger.debug(
                    f"FirstDetection: obj {obj_id} - no movement, keeping base angle"
                )
                self.objects_kept_base += 1

            # Mark as refined
            self.refined_objects.add(obj_id)

        return openlabel_data

    def _calculate_movement_direction(
        self, obj_id_str: str, start_frame: int, frames_data: Dict[int, List[float]]
    ) -> Optional[float]:
        """Calculate movement direction by looking ahead several frames.

        Args:
            obj_id_str: Object ID as string
            start_frame: Starting frame index
            frames_data: Frame data for this object

        Returns:
            Movement direction in radians (arctan2 result), or None if insufficient movement
        """
        frame_indices = sorted(frames_data.keys())
        start_idx_in_list = frame_indices.index(start_frame)

        # Look ahead up to lookahead_frames
        lookahead_indices = frame_indices[
            start_idx_in_list + 1 : start_idx_in_list + 1 + self.lookahead_frames
        ]

        if not lookahead_indices:
            return None

        start_cx, start_cy = frames_data[start_frame][0], frames_data[start_frame][1]

        # Check multiple future frames, use the one with most movement
        max_movement = 0.0
        best_direction = None

        for future_frame in lookahead_indices:
            future_cx, future_cy = (
                frames_data[future_frame][0],
                frames_data[future_frame][1],
            )

            delta_x = future_cx - start_cx
            delta_y = future_cy - start_cy
            movement = np.sqrt(delta_x**2 + delta_y**2)

            if movement > max_movement and movement >= self.min_movement_pixels:
                max_movement = movement
                best_direction = np.arctan2(delta_y, delta_x)

        return best_direction

    def get_statistics(self) -> Dict[str, Any]:
        """Get first detection refinement statistics.

        Returns:
            Dictionary with refinement statistics
        """
        return {
            "objects_refined": self.objects_refined,
            "objects_kept_base": self.objects_kept_base,
            "total_processed": self.objects_refined + self.objects_kept_base,
        }


class RotationAdjustmentPass(PostprocessingPass):
    """Adjust rotation values based on movement direction with improved temporal smoothing."""

    def __init__(
        self,
        rotation_threshold: float = 0.1,
        min_movement_pixels: float = 5.0,
        min_total_movement: float = 30.0,
        temporal_smoothing: float = 0.3,
        max_rotation_change: float = 0.524,
        aspect_instability_window: int = 5,
    ):
        """Initialize rotation adjustment pass.

        Args:
            rotation_threshold: Minimum angle difference to trigger adjustment (radians, default: 0.1)
            min_movement_pixels: Minimum movement distance to consider for rotation calculation (default: 5.0)
            min_total_movement: Minimum cumulative movement across all vectors to trust direction (default: 30.0)
            temporal_smoothing: Temporal smoothing factor (0-1, higher = more smoothing between frames, default: 0.3)
            max_rotation_change: Maximum rotation change per frame in radians (default: 0.524 ≈ 30°)
            aspect_instability_window: Number of frames to check for aspect ratio instability (default: 5)
        """
        self.rotations_adjusted = 0
        self.rotations_kept = 0
        self.rotations_copied = 0
        self.rotations_gap_skipped = 0
        self.rotations_skipped_slow = 0
        self.rotations_skipped_unstable = 0
        self.objects_processed = 0
        self.rotation_threshold = rotation_threshold
        self.min_movement_pixels = min_movement_pixels
        self.min_total_movement = min_total_movement
        self.temporal_smoothing = temporal_smoothing
        self.max_rotation_change = max_rotation_change
        self.aspect_instability_window = aspect_instability_window

    def process(self, openlabel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust rotation values based on movement direction with temporal smoothing.

        Args:
            openlabel_data: Complete OpenLabel data structure

        Returns:
            Modified OpenLabel data with adjusted rotations
        """
        frames = openlabel_data.get("openlabel", {}).get("frames", {})

        object_frame_map = defaultdict(list)

        for frame_idx_str, frame_data in frames.items():
            frame_idx = int(frame_idx_str)
            frame_objects = frame_data.get("objects", {})

            for obj_id_str in frame_objects.keys():
                object_frame_map[obj_id_str].append(frame_idx)

        for obj_id, frame_list in object_frame_map.items():
            if len(frame_list) < 2:
                continue

            self.objects_processed += 1
            frame_list_sorted = sorted(frame_list)
            last_valid_angle = None
            previous_smoothed_angle = None

            for i in range(len(frame_list_sorted)):
                current_frame = frame_list_sorted[i]
                current_frame_str = str(current_frame)
                frame_obj_data = frames[current_frame_str]["objects"][obj_id]
                is_last_frame = i == len(frame_list_sorted) - 1

                # Skip gap-filled frames - they have unreliable position data
                # Just copy the last valid angle if available
                if self._is_gap_filled(frame_obj_data):
                    if last_valid_angle is not None:
                        rbbox = frame_obj_data["object_data"]["rbbox"][0]["val"]
                        r_current = rbbox[4]
                        if abs(last_valid_angle - r_current) > self.rotation_threshold:
                            self._apply_rotation_adjustment(
                                frame_obj_data, last_valid_angle
                            )
                            self.rotations_copied += 1
                    self.rotations_gap_skipped += 1
                    continue

                if is_last_frame:
                    if last_valid_angle is not None:
                        rbbox = frame_obj_data["object_data"]["rbbox"][0]["val"]
                        r_current = rbbox[4]

                        if abs(last_valid_angle - r_current) > self.rotation_threshold:
                            self._apply_rotation_adjustment(
                                frame_obj_data, last_valid_angle
                            )
                            self.rotations_copied += 1
                    break

                r_new = self._calculate_smoothed_rotation(
                    frames, obj_id, current_frame, frame_list_sorted, i
                )

                rbbox = frame_obj_data["object_data"]["rbbox"][0]["val"]
                r_current = rbbox[4]

                if r_new is None:
                    if last_valid_angle is not None:
                        r_new = last_valid_angle
                        self._apply_rotation_adjustment(frame_obj_data, r_new)
                        self.rotations_copied += 1
                    continue
                else:
                    # Apply temporal smoothing with previous frame's smoothed angle
                    if previous_smoothed_angle is not None:
                        r_new = self._apply_temporal_smoothing(
                            previous_smoothed_angle, r_new
                        )

                    last_valid_angle = r_new
                    previous_smoothed_angle = r_new

                if abs(r_new - r_current) > self.rotation_threshold:
                    self._apply_rotation_adjustment(frame_obj_data, r_new)
                    self.rotations_adjusted += 1
                else:
                    self.rotations_kept += 1

        return openlabel_data

    def _is_gap_filled(self, frame_obj_data: Dict[str, Any]) -> bool:
        """Check if frame was created by gap filling.

        Gap-filled frames have unreliable position data (linear interpolation)
        and should not be used for movement direction calculation.

        Args:
            frame_obj_data: Frame object data

        Returns:
            True if frame was gap-filled, False otherwise
        """
        vec_list = frame_obj_data.get("object_data", {}).get("vec", [])
        for vec_item in vec_list:
            if vec_item.get("name") == "annotator":
                return "markit_housekeeping(gap)" in vec_item.get("val", [])
        return False

    def _apply_rotation_adjustment(
        self, frame_obj_data: Dict[str, Any], r_new: float
    ) -> None:
        """Apply rotation adjustment and update annotator/confidence.

        With the new semantic representation, width/height never swap.
        Only the rotation value is updated.

        Args:
            frame_obj_data: Frame object data
            r_new: New rotation value
        """
        rbbox = frame_obj_data["object_data"]["rbbox"][0]["val"]

        # Rebase angle if needed (only if |angle| > 2π)
        adjusted_rotation = rebase_angle_if_needed(r_new)

        # Update only rotation - width/height are semantic and don't swap
        rbbox[4] = adjusted_rotation

        vec_list = frame_obj_data["object_data"]["vec"]
        annotator_found = False
        confidence_found = False

        for vec_item in vec_list:
            if vec_item.get("name") == "annotator":
                vec_item["val"].insert(0, "markit_housekeeping(rot)")
                annotator_found = True
            elif vec_item.get("name") == "confidence":
                vec_item["val"].insert(0, 0.8888)
                confidence_found = True

        if not annotator_found:
            vec_list.insert(
                0, {"name": "annotator", "val": ["markit_housekeeping(rot)"]}
            )

        if not confidence_found:
            vec_list.append({"name": "confidence", "val": [0.8888]})

    def _apply_temporal_smoothing(self, prev_angle: float, curr_angle: float) -> float:
        """Apply temporal smoothing with maximum per-frame rotation limit.

        Prevents sudden rotation flips by:
        1. Limiting max rotation change per frame (configurable via max_rotation_change)
        2. Using EMA smoothing for gradual transitions

        Args:
            prev_angle: Previous frame's smoothed angle (radians, continuous)
            curr_angle: Current frame's calculated angle (radians)

        Returns:
            Temporally smoothed angle (radians, continuous)
        """
        # Calculate shortest angular difference
        angle_diff = curr_angle - prev_angle
        angle_diff = normalize_angle_to_pi(angle_diff)

        # Limit maximum rotation change per frame to prevent sudden flips
        if abs(angle_diff) > self.max_rotation_change:
            # Clamp the change to max allowed
            clamped_diff = np.sign(angle_diff) * self.max_rotation_change
            smoothed = prev_angle + clamped_diff
        else:
            # Small change - apply EMA smoothing
            smoothed = prev_angle + (1 - self.temporal_smoothing) * angle_diff

        smoothed = rebase_angle_if_needed(smoothed)
        return smoothed

    def _is_aspect_ratio_unstable(
        self,
        frames: Dict[str, Any],
        obj_id: str,
        frame_list_sorted: List[int],
        current_idx: int,
    ) -> bool:
        """Check if aspect ratio is unstable (w/h swap in progress).

        Returns True if the long axis (w vs h) differs across the lookback window,
        indicating the object's aspect ratio is changing and rotation adjustment
        should be skipped to prevent drift.

        Args:
            frames: Frame data
            obj_id: Object ID
            frame_list_sorted: Sorted list of frames for this object
            current_idx: Index in frame_list_sorted

        Returns:
            True if aspect ratio is unstable, False otherwise
        """
        window = self.aspect_instability_window
        long_axis_is_width = []

        start_idx = max(0, current_idx - window)
        end_idx = min(len(frame_list_sorted), current_idx + window + 1)

        for i in range(start_idx, end_idx):
            frame = frame_list_sorted[i]
            frame_str = str(frame)
            if frame_str not in frames:
                continue
            if obj_id not in frames[frame_str].get("objects", {}):
                continue

            obj = frames[frame_str]["objects"][obj_id]

            # Skip gap-filled frames - their dimensions are interpolated
            if self._is_gap_filled(obj):
                continue

            rbbox = obj["object_data"]["rbbox"][0]["val"]
            w, h = rbbox[2], rbbox[3]
            long_axis_is_width.append(w > h)

        if len(long_axis_is_width) < 3:
            return False  # Not enough data to determine stability

        # If long axis flips within window, aspect ratio is unstable
        all_width = all(long_axis_is_width)
        all_height = not any(long_axis_is_width)

        return not (all_width or all_height)

    def _calculate_smoothed_rotation(
        self,
        frames: Dict[str, Any],
        obj_id: str,
        current_frame: int,
        frame_list_sorted: List[int],
        current_idx: int,
    ) -> Optional[float]:
        """Calculate rotation based on movement direction with distance weighting.

        Uses movement vectors from neighboring frames, weighted by:
        - Distance traveled (longer = more reliable)
        - Temporal proximity (closer frames = higher weight)

        The temporal smoothing in the caller will prevent sudden flips.

        Args:
            frames: Frame data
            obj_id: Object ID
            current_frame: Current frame index
            frame_list_sorted: Sorted list of frames for this object
            current_idx: Index in frame_list_sorted

        Returns:
            Rotation angle in radians, or None if insufficient movement
        """
        # Skip rotation adjustment if aspect ratio is unstable (w/h swapping)
        if self._is_aspect_ratio_unstable(frames, obj_id, frame_list_sorted, current_idx):
            self.rotations_skipped_unstable += 1
            return None

        current_frame_str = str(current_frame)
        current_obj = frames[current_frame_str]["objects"][obj_id]
        current_rbbox = current_obj["object_data"]["rbbox"][0]["val"]
        x_current, y_current = current_rbbox[0], current_rbbox[1]
        w_current, h_current = current_rbbox[2], current_rbbox[3]

        angles = []
        weights = []
        total_distance = 0.0

        # Look backward (1-4 frames), skip gap-filled frames
        for lookback in range(1, 5):
            if current_idx - lookback < 0:
                break

            past_frame = frame_list_sorted[current_idx - lookback]
            past_frame_str = str(past_frame)
            past_obj = frames[past_frame_str]["objects"][obj_id]

            # Skip gap-filled frames - their positions are interpolated, not real
            if self._is_gap_filled(past_obj):
                continue

            past_rbbox = past_obj["object_data"]["rbbox"][0]["val"]
            x_past, y_past = past_rbbox[0], past_rbbox[1]

            delta_x = x_current - x_past
            delta_y = y_current - y_past
            distance = np.sqrt(delta_x**2 + delta_y**2)

            if distance >= self.min_movement_pixels:
                total_distance += distance
                angle = np.arctan2(delta_y, delta_x)
                # Weight by distance (longer movement = more reliable) and proximity
                weight = distance * (2.0 / lookback)
                angles.append(angle)
                weights.append(weight)

        # Look forward (1-8 frames), skip gap-filled frames
        for lookahead in range(1, 9):
            if current_idx + lookahead >= len(frame_list_sorted):
                break

            future_frame = frame_list_sorted[current_idx + lookahead]
            future_frame_str = str(future_frame)
            future_obj = frames[future_frame_str]["objects"][obj_id]

            # Skip gap-filled frames - their positions are interpolated, not real
            if self._is_gap_filled(future_obj):
                continue

            future_rbbox = future_obj["object_data"]["rbbox"][0]["val"]
            x_future, y_future = future_rbbox[0], future_rbbox[1]

            delta_x = x_future - x_current
            delta_y = y_future - y_current
            distance = np.sqrt(delta_x**2 + delta_y**2)

            if distance >= self.min_movement_pixels:
                total_distance += distance
                angle = np.arctan2(delta_y, delta_x)
                # Weight by distance and proximity
                weight = distance * (9.0 - lookahead)
                angles.append(angle)
                weights.append(weight)

        if not angles:
            return None

        # Skip if total movement is too small - direction is unreliable for slow objects
        if total_distance < self.min_total_movement:
            self.rotations_skipped_slow += 1
            return None

        # Circular averaging using sin/cos
        weighted_sin = sum(np.sin(a) * w for a, w in zip(angles, weights))
        weighted_cos = sum(np.cos(a) * w for a, w in zip(angles, weights))
        weight_sum = sum(weights)

        avg_sin = weighted_sin / weight_sum
        avg_cos = weighted_cos / weight_sum

        # Check consistency - if angles point in many directions, result is unreliable
        consistency = np.sqrt(avg_sin**2 + avg_cos**2)
        if consistency < 0.5:
            return None

        movement_direction = float(np.arctan2(avg_sin, avg_cos))

        # Determine rotation based on aspect ratio
        aspect_ratio = max(w_current, h_current) / max(min(w_current, h_current), 1.0)

        if aspect_ratio > 1.5:  # Elongated object
            if h_current > w_current:
                # Height is long axis, add 90° to movement direction
                correct_rotation = movement_direction + np.pi / 2
            else:
                # Width is long axis, use movement direction
                correct_rotation = movement_direction
        else:
            correct_rotation = movement_direction

        return correct_rotation

    def get_statistics(self) -> Dict[str, Any]:
        """Get rotation adjustment statistics.

        Returns:
            Dictionary with rotation adjustment statistics
        """
        return {
            "objects_processed": self.objects_processed,
            "rotations_adjusted": self.rotations_adjusted,
            "rotations_kept": self.rotations_kept,
            "rotations_copied": self.rotations_copied,
            "rotations_gap_skipped": self.rotations_gap_skipped,
            "rotations_skipped_slow": self.rotations_skipped_slow,
            "rotations_skipped_unstable": self.rotations_skipped_unstable,
        }


class SuddenPass(PostprocessingPass):
    """Detect sudden appearance/disappearance of objects near frame edges."""

    def __init__(self, edge_distance: int = 200):
        self.edge_distance = edge_distance
        self.sudden_appear_count = 0
        self.sudden_disappear_count = 0
        self.objects_with_events = set()

    def process(self, openlabel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect and record sudden appearance/disappearance events.

        Args:
            openlabel_data: Complete OpenLabel data structure

        Returns:
            Modified OpenLabel data with sudden events recorded
        """
        frames = openlabel_data.get("openlabel", {}).get("frames", {})
        objects = openlabel_data.get("openlabel", {}).get("objects", {})

        if not hasattr(self, "frame_width") or not hasattr(self, "frame_height"):
            logger.warning("SuddenPass: Video properties not set, skipping")
            return openlabel_data

        object_frame_map = defaultdict(list)

        for frame_idx_str, frame_data in frames.items():
            frame_idx = int(frame_idx_str)
            frame_objects = frame_data.get("objects", {})

            for obj_id_str in frame_objects.keys():
                object_frame_map[obj_id_str].append(frame_idx)

        frame_indices = sorted([int(f) for f in frames.keys()])
        if not frame_indices:
            return openlabel_data

        first_frame = frame_indices[0]
        last_frame = frame_indices[-1]

        for obj_id, frame_list in object_frame_map.items():
            frame_list_sorted = sorted(frame_list)

            sudden_appear_frames = []
            sudden_disappear_frames = []

            for i, frame_idx in enumerate(frame_list_sorted):
                if frame_idx == first_frame:
                    continue

                is_first_appearance = i == 0
                is_last_appearance = i == len(frame_list_sorted) - 1

                frame_str = str(frame_idx)
                frame_obj = frames[frame_str]["objects"][obj_id]
                rbbox = frame_obj["object_data"]["rbbox"][0]["val"]
                x, y, w, h, r = rbbox

                is_near_edge = self._is_near_edge(x, y, w, h)

                if is_first_appearance and frame_idx != first_frame and is_near_edge:
                    sudden_appear_frames.append(frame_idx)
                    self.sudden_appear_count += 1

                if is_last_appearance and frame_idx != last_frame and is_near_edge:
                    sudden_disappear_frames.append(frame_idx)
                    self.sudden_disappear_count += 1

            if sudden_appear_frames or sudden_disappear_frames:
                self.objects_with_events.add(obj_id)

                if obj_id not in objects:
                    continue

                if "object_data" not in objects[obj_id]:
                    objects[obj_id]["object_data"] = {}

                if "vec" not in objects[obj_id]["object_data"]:
                    objects[obj_id]["object_data"]["vec"] = []

                vec_list = objects[obj_id]["object_data"]["vec"]

                if sudden_appear_frames:
                    vec_list.append(
                        {"name": "suddenappear", "val": sudden_appear_frames}
                    )

                if sudden_disappear_frames:
                    vec_list.append(
                        {"name": "suddendisappear", "val": sudden_disappear_frames}
                    )

        return openlabel_data

    def _is_near_edge(self, x: float, y: float, w: float, h: float) -> bool:
        """Check if bounding box is near frame edge.

        Args:
            x: Center x coordinate
            y: Center y coordinate
            w: Width
            h: Height

        Returns:
            True if any part of bbox is within edge_distance of frame edge
        """
        x_min = x - w / 2
        x_max = x + w / 2
        y_min = y - h / 2
        y_max = y + h / 2

        near_left = x_min < self.edge_distance
        near_right = x_max > (self.frame_width - self.edge_distance)
        near_top = y_min < self.edge_distance
        near_bottom = y_max > (self.frame_height - self.edge_distance)

        return near_left or near_right or near_top or near_bottom

    def get_statistics(self) -> Dict[str, Any]:
        """Get sudden event statistics.

        Returns:
            Dictionary with sudden event statistics
        """
        return {
            "objects_with_events": len(self.objects_with_events),
            "sudden_appear_count": self.sudden_appear_count,
            "sudden_disappear_count": self.sudden_disappear_count,
        }


class FrameIntervalPass(PostprocessingPass):
    """Add frame_intervals to objects based on their frame appearances."""

    def __init__(self):
        self.intervals_added = 0
        self.intervals_skipped_existing = 0
        self.intervals_skipped_no_frames = 0

    def process(self, openlabel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add frame_intervals to objects based on frame appearances.

        Args:
            openlabel_data: Complete OpenLabel data structure

        Returns:
            Modified OpenLabel data with frame_intervals added
        """
        frames = openlabel_data.get("openlabel", {}).get("frames", {})
        objects = openlabel_data.get("openlabel", {}).get("objects", {})

        object_frame_map = defaultdict(list)

        for frame_idx_str, frame_data in frames.items():
            frame_idx = int(frame_idx_str)
            frame_objects = frame_data.get("objects", {})

            for obj_id_str in frame_objects.keys():
                object_frame_map[obj_id_str].append(frame_idx)

        for obj_id, obj_data in objects.items():
            if "frame_intervals" in obj_data:
                self.intervals_skipped_existing += 1
                continue

            if obj_id not in object_frame_map or len(object_frame_map[obj_id]) == 0:
                self.intervals_skipped_no_frames += 1
                continue

            frame_list = sorted(object_frame_map[obj_id])
            frame_start = frame_list[0]
            frame_end = frame_list[-1]

            obj_data["frame_intervals"] = [
                {"frame_start": frame_start, "frame_end": frame_end}
            ]
            self.intervals_added += 1

        return openlabel_data

    def get_statistics(self) -> Dict[str, Any]:
        """Get frame interval addition statistics.

        Returns:
            Dictionary with frame interval statistics
        """
        return {
            "intervals_added": self.intervals_added,
            "intervals_skipped_existing": self.intervals_skipped_existing,
            "intervals_skipped_no_frames": self.intervals_skipped_no_frames,
        }


class StaticObjectRemovalPass(PostprocessingPass):
    """Remove DynamicObject instances that don't move beyond threshold.
    NOTE: This pass will remove e.g. parked cars, pedestrians standing still, etc. If this
    is not desired, do not use this pass or use --static-mark to mark instead of remove.
    """

    def __init__(self, static_threshold: int = 20, mark_only: bool = False):
        """Initialize static object removal pass.

        Args:
            static_threshold: Movement threshold in pixels (default: 20)
            mark_only: If True, mark static objects instead of removing them (default: False)
        """
        self.static_threshold = static_threshold
        self.mark_only = mark_only
        self.objects_checked = 0
        self.objects_removed = 0
        self.objects_marked = 0
        self.frames_modified = 0
        self.removal_details = []
        self.marking_details = []

    def process(self, openlabel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove DynamicObjects that don't move beyond threshold.

        Args:
            openlabel_data: Complete OpenLabel data structure

        Returns:
            Modified OpenLabel data with static DynamicObjects removed
        """
        # Check if ontology path is set
        if not hasattr(self, "ontology_path") or not self.ontology_path:
            logger.warning("StaticObjectRemovalPass: Ontology path not set, skipping")
            return openlabel_data

        # Import ontology functions (done here to avoid circular imports)
        import os
        from savant_common.ontology import get_class_by_label

        # Check if ontology file exists
        if not os.path.exists(self.ontology_path):
            logger.warning(
                f"StaticObjectRemovalPass: Ontology file not found: {self.ontology_path}"
            )
            return openlabel_data

        frames = openlabel_data.get("openlabel", {}).get("frames", {})
        objects = openlabel_data.get("openlabel", {}).get("objects", {})

        # Build object-to-frames mapping
        object_frame_map = defaultdict(list)
        for frame_idx_str, frame_data in frames.items():
            frame_idx = int(frame_idx_str)
            frame_objects = frame_data.get("objects", {})
            for obj_id_str in frame_objects.keys():
                object_frame_map[obj_id_str].append(frame_idx)

        # Check each object
        objects_to_remove = []

        for obj_id, obj_data in objects.items():
            obj_type = obj_data.get("type", "")

            # Look up class in ontology
            try:
                class_info = get_class_by_label(
                    self.ontology_path, obj_type, case_sensitive=False
                )

                # Skip if not found in ontology
                if class_info is None:
                    logger.debug(
                        f"StaticObjectRemovalPass: Class '{obj_type}' not found in ontology, skipping object {obj_id}"
                    )
                    continue

                # Check if top-level class is DynamicObject
                top_level_class_name = class_info.get("top_level_class_name")
                if top_level_class_name != "DynamicObject":
                    continue

                # This is a DynamicObject, check movement
                self.objects_checked += 1

                # Get all frame appearances
                frame_list = object_frame_map.get(obj_id, [])
                if len(frame_list) == 0:
                    continue

                # Calculate movement
                x_positions = []
                y_positions = []

                for frame_idx in frame_list:
                    frame_str = str(frame_idx)
                    if frame_str not in frames:
                        continue

                    frame_obj = frames[frame_str]["objects"].get(obj_id)
                    if not frame_obj:
                        continue

                    try:
                        rbbox = frame_obj["object_data"]["rbbox"][0]["val"]
                        x, y = rbbox[0], rbbox[1]  # Center coordinates
                        x_positions.append(x)
                        y_positions.append(y)
                    except (KeyError, IndexError, TypeError) as e:
                        logger.debug(
                            f"StaticObjectRemovalPass: Error extracting position for object {obj_id} in frame {frame_idx}: {e}"
                        )
                        continue

                # Check if we have position data
                if len(x_positions) == 0 or len(y_positions) == 0:
                    continue

                # Calculate max-min movement in each dimension
                delta_x = max(x_positions) - min(x_positions)
                delta_y = max(y_positions) - min(y_positions)

                # Check if both dimensions are below threshold
                if (
                    delta_x <= self.static_threshold
                    and delta_y <= self.static_threshold
                ):
                    objects_to_remove.append(obj_id)

                    # Store first frame for marking
                    first_frame = min(frame_list)

                    if self.mark_only:
                        self.marking_details.append(
                            {
                                "object_id": obj_id,
                                "type": obj_type,
                                "delta_x": delta_x,
                                "delta_y": delta_y,
                                "frame_count": len(frame_list),
                                "first_frame": first_frame,
                            }
                        )
                        logger.info(
                            f"Marked static object {obj_id} (type: {obj_type}) - "
                            f"movement: dx={delta_x}px, dy={delta_y}px, frames={len(frame_list)}"
                        )
                    else:
                        self.removal_details.append(
                            {
                                "object_id": obj_id,
                                "type": obj_type,
                                "delta_x": delta_x,
                                "delta_y": delta_y,
                                "frame_count": len(frame_list),
                            }
                        )
                        logger.info(
                            f"Removed static object {obj_id} (type: {obj_type}) - "
                            f"movement: dx={delta_x}px, dy={delta_y}px, frames={len(frame_list)}"
                        )

            except Exception as e:
                logger.warning(
                    f"StaticObjectRemovalPass: Error processing object {obj_id}: {e}"
                )
                continue

        # Mark or remove objects
        if self.mark_only:
            # Mark objects by adding "staticdynamic" annotation
            for detail in self.marking_details:
                obj_id = detail["object_id"]
                first_frame = detail["first_frame"]

                if obj_id not in objects:
                    continue

                if "object_data" not in objects[obj_id]:
                    objects[obj_id]["object_data"] = {}

                if "vec" not in objects[obj_id]["object_data"]:
                    objects[obj_id]["object_data"]["vec"] = []

                vec_list = objects[obj_id]["object_data"]["vec"]
                vec_list.append({"name": "staticdynamic", "val": [first_frame]})

                self.objects_marked += 1
        else:
            # Remove objects
            for obj_id in objects_to_remove:
                # Remove from objects dictionary
                if obj_id in objects:
                    del objects[obj_id]
                    self.objects_removed += 1

                # Remove from all frames
                for frame_idx_str, frame_data in frames.items():
                    frame_objects = frame_data.get("objects", {})
                    if obj_id in frame_objects:
                        del frame_objects[obj_id]
                        self.frames_modified += 1

        return openlabel_data

    def get_statistics(self) -> Dict[str, Any]:
        """Get static object removal/marking statistics.

        Returns:
            Dictionary with removal or marking statistics
        """
        if self.mark_only:
            return {
                "objects_checked": self.objects_checked,
                "objects_marked": self.objects_marked,
            }
        else:
            return {
                "objects_checked": self.objects_checked,
                "objects_removed": self.objects_removed,
                "frames_modified": self.frames_modified,
            }


class BboxSmoothingPass(PostprocessingPass):
    """Apply temporal smoothing to bbox size parameters (w, h) only.

    Position (x, y) is NOT smoothed - raw positions are acceptable and smoothing
    can actually increase jitter in some cases. Size smoothing uses bidirectional
    EMA with special handling for objects near frame edges.
    """

    def __init__(
        self,
        smoothing_factor: float = 0.7,
        edge_margin: int = 100,
        edge_size_mode: str = "freeze",
    ):
        """Initialize bbox smoothing pass.

        Args:
            smoothing_factor: Base EMA retention factor (0-1). Represents how much of the
                previous smoothed value to keep. Higher = more smoothing/stability.
                With factor 0.7: new_smoothed = 0.7 * old_smoothed + 0.3 * raw_value
                Default 0.7 provides good noise rejection while tracking real movement.
            edge_margin: Pixels from frame edge for special handling (default: 100)
            edge_size_mode: How to handle size near edges - "freeze" or "normal" (default: "freeze")
        """
        self.smoothing_factor = smoothing_factor
        self.edge_margin = edge_margin
        self.edge_size_mode = edge_size_mode

        # Statistics
        self.objects_smoothed = 0
        self.frames_smoothed = 0
        self.edge_frames_handled = 0

    def process(self, openlabel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply temporal smoothing to bbox parameters.

        Args:
            openlabel_data: Complete OpenLabel data structure

        Returns:
            Modified OpenLabel data with smoothed bbox parameters
        """
        frames = openlabel_data.get("openlabel", {}).get("frames", {})

        if not hasattr(self, "frame_width") or not hasattr(self, "frame_height"):
            logger.warning("BboxSmoothingPass: Video properties not set, skipping")
            return openlabel_data

        # Build object-to-frames mapping
        object_frame_map = defaultdict(list)
        for frame_idx_str, frame_data in frames.items():
            frame_idx = int(frame_idx_str)
            frame_objects = frame_data.get("objects", {})
            for obj_id_str in frame_objects.keys():
                object_frame_map[obj_id_str].append(frame_idx)

        # Process each object
        for obj_id, frame_list in object_frame_map.items():
            if len(frame_list) < 2:
                continue

            frame_list_sorted = sorted(frame_list)
            self._smooth_object_trajectory(frames, obj_id, frame_list_sorted)
            self.objects_smoothed += 1

        logger.info(
            f"BboxSmoothing: Smoothed {self.objects_smoothed} objects, "
            f"{self.frames_smoothed} frames, {self.edge_frames_handled} edge frames handled"
        )

        return openlabel_data

    def _smooth_object_trajectory(
        self,
        frames: Dict[str, Any],
        obj_id: str,
        frame_list: List[int],
    ) -> None:
        """Smooth size (w, h) of a single object using bidirectional EMA.

        Position (x, y) is NOT smoothed - raw positions are acceptable.

        Bidirectional smoothing eliminates lag by:
        1. Forward pass: EMA from start to end
        2. Backward pass: EMA from end to start
        3. Average the two passes

        Args:
            frames: Frame data dictionary
            obj_id: Object ID string
            frame_list: Sorted list of frame indices for this object
        """
        # Collect raw values
        raw_values = []
        for frame_idx in frame_list:
            frame_str = str(frame_idx)
            rbbox = frames[frame_str]["objects"][obj_id]["object_data"]["rbbox"][0]["val"]
            x, y, w, h, r = rbbox
            is_near_edge = self._is_near_edge(x, y)
            raw_values.append({
                "frame_idx": frame_idx,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "r": r,
                "is_near_edge": is_near_edge,
            })

        n = len(raw_values)
        if n == 0:
            return

        # Forward pass (size only)
        forward_w = [0.0] * n
        forward_h = [0.0] * n

        forward_w[0] = raw_values[0]["w"]
        forward_h[0] = raw_values[0]["h"]

        factor = self.smoothing_factor
        for i in range(1, n):
            w, h = raw_values[i]["w"], raw_values[i]["h"]
            forward_w[i] = factor * forward_w[i-1] + (1 - factor) * w
            forward_h[i] = factor * forward_h[i-1] + (1 - factor) * h

        # Backward pass (size only)
        backward_w = [0.0] * n
        backward_h = [0.0] * n

        backward_w[n-1] = raw_values[n-1]["w"]
        backward_h[n-1] = raw_values[n-1]["h"]

        for i in range(n-2, -1, -1):
            w, h = raw_values[i]["w"], raw_values[i]["h"]
            backward_w[i] = factor * backward_w[i+1] + (1 - factor) * w
            backward_h[i] = factor * backward_h[i+1] + (1 - factor) * h

        # First pass to find interior sizes for edge handling
        interior_sizes = []
        for i in range(n):
            avg_w = (forward_w[i] + backward_w[i]) / 2
            avg_h = (forward_h[i] + backward_h[i]) / 2
            if not raw_values[i]["is_near_edge"]:
                interior_sizes.append((i, avg_w, avg_h))

        # Apply smoothed size values (position unchanged)
        for i in range(n):
            frame_idx = raw_values[i]["frame_idx"]
            frame_str = str(frame_idx)
            rbbox = frames[frame_str]["objects"][obj_id]["object_data"]["rbbox"][0]["val"]
            is_near_edge = raw_values[i]["is_near_edge"]

            # Average bidirectional smoothing for size
            smoothed_w = (forward_w[i] + backward_w[i]) / 2
            smoothed_h = (forward_h[i] + backward_h[i]) / 2

            # Handle size near edges - use nearest interior size
            if is_near_edge and self.edge_size_mode == "freeze" and interior_sizes:
                # Find nearest interior frame
                nearest = min(interior_sizes, key=lambda x: abs(x[0] - i))
                smoothed_w, smoothed_h = nearest[1], nearest[2]
                self.edge_frames_handled += 1

            # Only update size - position (x, y) stays unchanged
            rbbox[2] = smoothed_w
            rbbox[3] = smoothed_h

            self._update_annotator(frames[frame_str]["objects"][obj_id])
            self.frames_smoothed += 1

    def _is_near_edge(self, x: float, y: float) -> bool:
        """Check if position is near frame edge.

        Args:
            x: Center x coordinate
            y: Center y coordinate

        Returns:
            True if position is within edge_margin of any frame edge
        """
        return (
            x < self.edge_margin
            or x > self.frame_width - self.edge_margin
            or y < self.edge_margin
            or y > self.frame_height - self.edge_margin
        )

    def _update_annotator(self, obj_data: Dict[str, Any]) -> None:
        """Update annotator field to indicate smoothing was applied.

        Args:
            obj_data: Object data dictionary
        """
        vec_list = obj_data["object_data"]["vec"]
        for vec_item in vec_list:
            if vec_item.get("name") == "annotator":
                # Only add if not already present
                if "markit_housekeeping(smooth)" not in vec_item["val"]:
                    vec_item["val"].insert(0, "markit_housekeeping(smooth)")
                return

        # No annotator field found, add one
        vec_list.insert(0, {"name": "annotator", "val": ["markit_housekeeping(smooth)"]})

    def get_statistics(self) -> Dict[str, Any]:
        """Get bbox smoothing statistics.

        Returns:
            Dictionary with smoothing statistics
        """
        return {
            "objects_smoothed": self.objects_smoothed,
            "frames_smoothed": self.frames_smoothed,
            "edge_frames_handled": self.edge_frames_handled,
        }


class SizeOutlierFilterPass(PostprocessingPass):
    """Detect and fix frames where object size is an outlier compared to its history.

    Uses MAD (Median Absolute Deviation) to identify outlier frames for each object
    individually. This catches motion streak artifacts (suddenly much larger area)
    without rejecting legitimately large vehicles like buses.

    Outlier frames have their size interpolated from neighboring non-outlier frames.
    """

    def __init__(self, outlier_threshold: float = 3.0, min_frames: int = 5):
        """Initialize size outlier filter pass.

        Args:
            outlier_threshold: Number of MAD from median to consider outlier (default: 3.0)
            min_frames: Minimum frames an object must have for outlier detection (default: 5)
        """
        self.outlier_threshold = outlier_threshold
        self.min_frames = min_frames
        self.outliers_fixed = 0
        self.objects_processed = 0

    def process(self, openlabel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect and fix size outliers per object.

        Args:
            openlabel_data: Complete OpenLabel data structure

        Returns:
            Modified OpenLabel data with outliers fixed
        """
        frames = openlabel_data.get("openlabel", {}).get("frames", {})

        # Build object-to-frames mapping
        object_frame_map = defaultdict(list)
        for frame_idx_str, frame_data in frames.items():
            frame_idx = int(frame_idx_str)
            frame_objects = frame_data.get("objects", {})
            for obj_id_str in frame_objects.keys():
                object_frame_map[obj_id_str].append(frame_idx)

        # Process each object
        for obj_id, frame_list in object_frame_map.items():
            if len(frame_list) < self.min_frames:
                continue

            self.objects_processed += 1
            frame_list_sorted = sorted(frame_list)
            self._fix_object_outliers(frames, obj_id, frame_list_sorted)

        logger.info(
            f"SizeOutlierFilter: Processed {self.objects_processed} objects, "
            f"fixed {self.outliers_fixed} outlier frames"
        )

        return openlabel_data

    def _fix_object_outliers(
        self,
        frames: Dict[str, Any],
        obj_id: str,
        frame_list: List[int],
    ) -> None:
        """Fix size outliers for a single object.

        Args:
            frames: Frame data dictionary
            obj_id: Object ID string
            frame_list: Sorted list of frame indices for this object
        """
        # Collect size data (area = w * h)
        size_data = []
        for frame_idx in frame_list:
            frame_str = str(frame_idx)
            rbbox = frames[frame_str]["objects"][obj_id]["object_data"]["rbbox"][0]["val"]
            w, h = rbbox[2], rbbox[3]
            area = w * h
            size_data.append({
                "frame_idx": frame_idx,
                "w": w,
                "h": h,
                "area": area,
            })

        # Calculate median area and MAD
        areas = np.array([s["area"] for s in size_data])
        median_area = np.median(areas)
        mad = np.median(np.abs(areas - median_area))

        # Avoid division by zero - if MAD is very small, sizes are very consistent
        if mad < 1.0:
            mad = 1.0

        # Identify outliers
        outlier_indices = []
        for i, s in enumerate(size_data):
            deviation = abs(s["area"] - median_area) / mad
            if deviation > self.outlier_threshold:
                outlier_indices.append(i)

        if not outlier_indices:
            return

        # Fix outliers by interpolating from neighbors
        for outlier_idx in outlier_indices:
            frame_idx = size_data[outlier_idx]["frame_idx"]
            frame_str = str(frame_idx)

            # Find nearest non-outlier neighbors
            prev_good = None
            next_good = None

            # Search backward for non-outlier
            for i in range(outlier_idx - 1, -1, -1):
                if i not in outlier_indices:
                    prev_good = size_data[i]
                    break

            # Search forward for non-outlier
            for i in range(outlier_idx + 1, len(size_data)):
                if i not in outlier_indices:
                    next_good = size_data[i]
                    break

            # Interpolate size from neighbors
            if prev_good and next_good:
                # Linear interpolation between neighbors
                t = (frame_idx - prev_good["frame_idx"]) / (
                    next_good["frame_idx"] - prev_good["frame_idx"]
                )
                new_w = prev_good["w"] + t * (next_good["w"] - prev_good["w"])
                new_h = prev_good["h"] + t * (next_good["h"] - prev_good["h"])
            elif prev_good:
                # Use previous good value
                new_w, new_h = prev_good["w"], prev_good["h"]
            elif next_good:
                # Use next good value
                new_w, new_h = next_good["w"], next_good["h"]
            else:
                # No good neighbors - skip
                continue

            # Apply fix
            rbbox = frames[frame_str]["objects"][obj_id]["object_data"]["rbbox"][0]["val"]
            rbbox[2] = new_w
            rbbox[3] = new_h

            self._update_annotator(frames[frame_str]["objects"][obj_id])
            self.outliers_fixed += 1

    def _update_annotator(self, obj_data: Dict[str, Any]) -> None:
        """Update annotator field to indicate outlier fix was applied."""
        vec_list = obj_data["object_data"]["vec"]
        for vec_item in vec_list:
            if vec_item.get("name") == "annotator":
                if "markit_housekeeping(outlier)" not in vec_item["val"]:
                    vec_item["val"].insert(0, "markit_housekeeping(outlier)")
                return
        vec_list.insert(0, {"name": "annotator", "val": ["markit_housekeeping(outlier)"]})

    def get_statistics(self) -> Dict[str, Any]:
        """Get size outlier filter statistics."""
        return {
            "objects_processed": self.objects_processed,
            "outliers_fixed": self.outliers_fixed,
        }


class Rotation90JumpFixPass(PostprocessingPass):
    """Fix 90° rotation jumps caused by minAreaRect w/h swapping.

    minAreaRect can return equivalent bounding boxes rotated by 90° with w/h swapped.
    This pass detects such jumps and corrects them by swapping w/h and adjusting rotation.
    """

    def __init__(self, jump_threshold_low: float = 70.0, jump_threshold_high: float = 110.0):
        """Initialize 90° jump fix pass.

        Args:
            jump_threshold_low: Lower bound for 90° jump detection in degrees (default: 70)
            jump_threshold_high: Upper bound for 90° jump detection in degrees (default: 110)
        """
        self.jump_threshold_low = np.radians(jump_threshold_low)
        self.jump_threshold_high = np.radians(jump_threshold_high)
        self.jumps_fixed = 0
        self.objects_processed = 0
        self.wh_swaps = 0

    def process(self, openlabel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fix 90° rotation jumps in object trajectories.

        Args:
            openlabel_data: Complete OpenLabel data structure

        Returns:
            Modified OpenLabel data with 90° jumps fixed
        """
        frames = openlabel_data.get("openlabel", {}).get("frames", {})

        # Build object-to-frames mapping
        object_frame_map = defaultdict(list)
        for frame_idx_str, frame_data in frames.items():
            frame_idx = int(frame_idx_str)
            frame_objects = frame_data.get("objects", {})
            for obj_id_str in frame_objects.keys():
                object_frame_map[obj_id_str].append(frame_idx)

        # Process each object
        for obj_id, frame_list in object_frame_map.items():
            if len(frame_list) < 2:
                continue

            self.objects_processed += 1
            frame_list_sorted = sorted(frame_list)
            self._fix_object_jumps(frames, obj_id, frame_list_sorted)

        logger.info(
            f"Rotation90JumpFix: Processed {self.objects_processed} objects, "
            f"fixed {self.jumps_fixed} jumps, {self.wh_swaps} w/h swaps"
        )

        return openlabel_data

    def _fix_object_jumps(
        self,
        frames: Dict[str, Any],
        obj_id: str,
        frame_list: List[int],
    ) -> None:
        """Fix 90° jumps for a single object.

        Args:
            frames: Frame data dictionary
            obj_id: Object ID string
            frame_list: Sorted list of frame indices for this object
        """
        prev_rotation = None

        for frame_idx in frame_list:
            frame_str = str(frame_idx)
            rbbox = frames[frame_str]["objects"][obj_id]["object_data"]["rbbox"][0]["val"]
            x, y, w, h, r = rbbox

            if prev_rotation is not None:
                # Calculate angle difference
                diff = normalize_angle_to_pi(r - prev_rotation)
                abs_diff = abs(diff)

                # Check if this looks like a 90° jump
                if self.jump_threshold_low < abs_diff < self.jump_threshold_high:
                    # Try adjusting rotation by ±90°
                    sign = -1 if diff > 0 else 1
                    alt_rotation = r + sign * (np.pi / 2)
                    alt_diff = normalize_angle_to_pi(alt_rotation - prev_rotation)

                    if abs(alt_diff) < abs_diff:
                        # Swapping makes it more continuous - apply correction
                        rbbox[2] = h  # Swap w and h
                        rbbox[3] = w
                        rbbox[4] = alt_rotation
                        r = alt_rotation  # Update for next iteration
                        self.jumps_fixed += 1
                        self.wh_swaps += 1
                        self._update_annotator(frames[frame_str]["objects"][obj_id])

            prev_rotation = r

    def _update_annotator(self, obj_data: Dict[str, Any]) -> None:
        """Update annotator field to indicate 90° jump fix was applied."""
        vec_list = obj_data["object_data"]["vec"]
        for vec_item in vec_list:
            if vec_item.get("name") == "annotator":
                if "markit_housekeeping(90fix)" not in vec_item["val"]:
                    vec_item["val"].insert(0, "markit_housekeeping(90fix)")
                return
        vec_list.insert(0, {"name": "annotator", "val": ["markit_housekeeping(90fix)"]})

    def get_statistics(self) -> Dict[str, Any]:
        """Get 90° jump fix statistics."""
        return {
            "objects_processed": self.objects_processed,
            "jumps_fixed": self.jumps_fixed,
            "wh_swaps": self.wh_swaps,
        }


class RotationTemporalSmoothingPass(PostprocessingPass):
    """Apply light temporal smoothing to rotation without recalculating from movement.

    Only smooths small jitter (<20°). Large intentional rotations are preserved.
    Does NOT use movement direction - respects the raw rotation values.
    """

    def __init__(self, smoothing_factor: float = 0.3, max_smooth_angle: float = 20.0):
        """Initialize rotation temporal smoothing pass.

        Args:
            smoothing_factor: EMA factor for smoothing (0-1, default: 0.3)
            max_smooth_angle: Maximum angle difference to smooth in degrees (default: 20)
        """
        self.smoothing_factor = smoothing_factor
        self.max_smooth_angle = np.radians(max_smooth_angle)
        self.rotations_smoothed = 0
        self.rotations_kept = 0
        self.objects_processed = 0

    def process(self, openlabel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply temporal smoothing to rotation values.

        Args:
            openlabel_data: Complete OpenLabel data structure

        Returns:
            Modified OpenLabel data with smoothed rotations
        """
        frames = openlabel_data.get("openlabel", {}).get("frames", {})

        # Build object-to-frames mapping
        object_frame_map = defaultdict(list)
        for frame_idx_str, frame_data in frames.items():
            frame_idx = int(frame_idx_str)
            frame_objects = frame_data.get("objects", {})
            for obj_id_str in frame_objects.keys():
                object_frame_map[obj_id_str].append(frame_idx)

        # Process each object
        for obj_id, frame_list in object_frame_map.items():
            if len(frame_list) < 2:
                continue

            self.objects_processed += 1
            frame_list_sorted = sorted(frame_list)
            self._smooth_object_rotation(frames, obj_id, frame_list_sorted)

        logger.info(
            f"RotationTemporalSmoothing: Processed {self.objects_processed} objects, "
            f"smoothed {self.rotations_smoothed} frames, kept {self.rotations_kept} unchanged"
        )

        return openlabel_data

    def _smooth_object_rotation(
        self,
        frames: Dict[str, Any],
        obj_id: str,
        frame_list: List[int],
    ) -> None:
        """Smooth rotation for a single object.

        Args:
            frames: Frame data dictionary
            obj_id: Object ID string
            frame_list: Sorted list of frame indices for this object
        """
        smoothed_rotation = None

        for frame_idx in frame_list:
            frame_str = str(frame_idx)
            rbbox = frames[frame_str]["objects"][obj_id]["object_data"]["rbbox"][0]["val"]
            current_rotation = rbbox[4]

            if smoothed_rotation is None:
                # First frame - initialize
                smoothed_rotation = current_rotation
            else:
                # Calculate difference
                diff = normalize_angle_to_pi(current_rotation - smoothed_rotation)

                if abs(diff) < self.max_smooth_angle:
                    # Small jitter - smooth it
                    smoothed_rotation = smoothed_rotation + self.smoothing_factor * diff
                    rbbox[4] = smoothed_rotation
                    self.rotations_smoothed += 1
                    self._update_annotator(frames[frame_str]["objects"][obj_id])
                else:
                    # Large change - accept it (might be real)
                    smoothed_rotation = current_rotation
                    self.rotations_kept += 1

    def _update_annotator(self, obj_data: Dict[str, Any]) -> None:
        """Update annotator field to indicate smoothing was applied."""
        vec_list = obj_data["object_data"]["vec"]
        for vec_item in vec_list:
            if vec_item.get("name") == "annotator":
                if "markit_housekeeping(rotsmooth)" not in vec_item["val"]:
                    vec_item["val"].insert(0, "markit_housekeeping(rotsmooth)")
                return
        vec_list.insert(0, {"name": "annotator", "val": ["markit_housekeeping(rotsmooth)"]})

    def get_statistics(self) -> Dict[str, Any]:
        """Get rotation smoothing statistics."""
        return {
            "objects_processed": self.objects_processed,
            "rotations_smoothed": self.rotations_smoothed,
            "rotations_kept": self.rotations_kept,
        }


class AngleNormalizationPass(PostprocessingPass):
    """Normalize all rotation angles to [0, 2π) for OpenLabel output.

    This is a MANDATORY final pass that ensures all rotation values in the
    OpenLabel output conform to the [0, 2π) range, regardless of what
    internal continuous angle representation was used during postprocessing.

    This pass should always be the LAST pass in the pipeline.
    """

    def __init__(self):
        """Initialize angle normalization pass."""
        self.angles_normalized = 0

    def process(self, openlabel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize all rotation angles to [0, 2π) range.

        Args:
            openlabel_data: OpenLabel structure with frame data

        Returns:
            Modified OpenLabel structure with normalized angles
        """
        frames = openlabel_data.get("openlabel", {}).get("frames", {})

        for frame_idx_str, frame_data in frames.items():
            frame_objects = frame_data.get("objects", {})

            for obj_id_str, obj_data in frame_objects.items():
                # Get rbbox data
                rbbox = obj_data["object_data"]["rbbox"][0]["val"]
                rotation = rbbox[4]

                # Normalize to [0, 2π) for OpenLabel output
                normalized_rotation = normalize_angle_to_2pi_range(rotation)

                # Update if changed
                if rotation != normalized_rotation:
                    rbbox[4] = normalized_rotation
                    self.angles_normalized += 1

        logger.info(
            f"AngleNormalization: Normalized {self.angles_normalized} angles to [0, 2π) range"
        )

        return openlabel_data

    def get_statistics(self) -> Dict[str, Any]:
        """Get angle normalization statistics.

        Returns:
            Dictionary with normalization statistics
        """
        return {"angles_normalized": self.angles_normalized}
