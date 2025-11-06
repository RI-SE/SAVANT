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
from ..utils import normalize_angle_to_pi, normalize_angle_to_2pi_range, rebase_angle_if_needed

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
                    gaps.append({
                        'start_frame': current_frame,
                        'end_frame': next_frame,
                        'gap_size': gap_size
                    })

            if gaps:
                self.gaps_detected[obj_id] = {
                    'frame_range': (frame_list_sorted[0], frame_list_sorted[-1]),
                    'total_frames': len(frame_list_sorted),
                    'gaps': gaps
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
        total_gaps = sum(len(info['gaps']) for info in self.gaps_detected.values())

        return {
            'objects_with_gaps': len(self.objects_with_gaps),
            'total_gaps_detected': total_gaps,
            'gap_details': self.gaps_detected
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
                        openlabel_data,
                        obj_id,
                        frame_before,
                        frame_after,
                        gap_size
                    )

        return openlabel_data

    def _fill_gap(self, openlabel_data: Dict[str, Any], obj_id: str,
                  frame_before: int, frame_after: int, gap_size: int) -> None:
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

        delta_x = x_after - x_before
        delta_y = y_after - y_before

        total_steps = gap_size + 1

        for step in range(1, gap_size + 1):
            interpolation_factor = step / total_steps

            x_interpolated = int(x_before + delta_x * interpolation_factor)
            y_interpolated = int(y_before + delta_y * interpolation_factor)

            missing_frame_idx = frame_before + step
            missing_frame_str = str(missing_frame_idx)

            if missing_frame_str not in frames:
                frames[missing_frame_str] = {"objects": {}}

            frames[missing_frame_str]["objects"][obj_id] = {
                "object_data": {
                    "rbbox": [{
                        "name": "shape",
                        "val": [x_interpolated, y_interpolated, w_before, h_before, r_before]
                    }],
                    "vec": [
                        {
                            "name": "annotator",
                            "val": ["markit_housekeeping(gap)"]
                        },
                        {
                            "name": "confidence",
                            "val": [0.6666]
                        }
                    ]
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
            'objects_processed': len(self.objects_processed),
            'gaps_filled': self.gaps_filled,
            'frames_added': self.frames_added
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
                    self.deletion_details.append({
                        'deleted_object': obj_to_delete,
                        'kept_object': obj_to_keep,
                        'frame_start': frames_list[0] if frames_list else None,
                        'frame_end': frames_list[-1] if frames_list else None
                    })

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

    def _are_duplicates(self, obj_a: str, obj_b: str,
                       object_frame_map: Dict[str, List[int]],
                       frames: Dict[str, Any]) -> bool:
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
                iou = self.iou_calculator.calculate_intersection_over_union(bbox_a, bbox_b)
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

            corners = np.array([
                [-hw, -hh],
                [hw, -hh],
                [hw, hh],
                [-hw, hh]
            ])

            rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            rotated_corners = corners @ rotation_matrix.T

            oriented_bbox = rotated_corners + np.array([x, y])

            return oriented_bbox.astype(np.float32)

        except (KeyError, IndexError, ValueError) as e:
            logger.debug(f"Failed to extract bbox: {e}")
            return None

    def _choose_object_to_delete(self, obj_a: str, obj_b: str,
                                 object_frame_map: Dict[str, List[int]],
                                 frames: Dict[str, Any]) -> str:
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

    def _calculate_average_confidence(self, obj_id: str,
                                     object_frame_map: Dict[str, List[int]],
                                     frames: Dict[str, Any]) -> float:
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
            'objects_deleted': self.objects_deleted,
            'duplicate_pairs_found': self.duplicate_pairs_found,
            'frames_modified': self.frames_modified
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
            cx, cy = first_rbbox[0], first_rbbox[1]
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

                logger.info(f"FirstDetection: obj {obj_id} base_angle={np.degrees(base_angle):.1f}° "
                           f"→ refined={np.degrees(refined_angle):.1f}° (movement={np.degrees(movement_dir):.1f}°)")
                self.objects_refined += 1
            else:
                # No significant movement detected - keep base angle
                logger.debug(f"FirstDetection: obj {obj_id} - no movement, keeping base angle")
                self.objects_kept_base += 1

            # Mark as refined
            self.refined_objects.add(obj_id)

        return openlabel_data

    def _calculate_movement_direction(self, obj_id_str: str, start_frame: int,
                                     frames_data: Dict[int, List[float]]) -> Optional[float]:
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
        lookahead_indices = frame_indices[start_idx_in_list + 1 : start_idx_in_list + 1 + self.lookahead_frames]

        if not lookahead_indices:
            return None

        start_cx, start_cy = frames_data[start_frame][0], frames_data[start_frame][1]

        # Check multiple future frames, use the one with most movement
        max_movement = 0.0
        best_direction = None

        for future_frame in lookahead_indices:
            future_cx, future_cy = frames_data[future_frame][0], frames_data[future_frame][1]

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
            'objects_refined': self.objects_refined,
            'objects_kept_base': self.objects_kept_base,
            'total_processed': self.objects_refined + self.objects_kept_base
        }


class RotationAdjustmentPass(PostprocessingPass):
    """Adjust rotation values based on movement direction with improved temporal smoothing."""

    def __init__(self, rotation_threshold: float = 0.1, min_movement_pixels: float = 5.0,
                 temporal_smoothing: float = 0.3):
        """Initialize rotation adjustment pass.

        Args:
            rotation_threshold: Minimum angle difference to trigger adjustment (radians, default: 0.1)
            min_movement_pixels: Minimum movement distance to consider for rotation calculation (default: 5.0)
            temporal_smoothing: Temporal smoothing factor (0-1, higher = more smoothing between frames, default: 0.3)
        """
        self.rotations_adjusted = 0
        self.rotations_kept = 0
        self.rotations_copied = 0
        self.objects_processed = 0
        self.rotation_threshold = rotation_threshold
        self.min_movement_pixels = min_movement_pixels
        self.temporal_smoothing = temporal_smoothing

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
                is_last_frame = (i == len(frame_list_sorted) - 1)

                if is_last_frame:
                    if last_valid_angle is not None:
                        current_frame_str = str(current_frame)
                        frame_obj_data = frames[current_frame_str]["objects"][obj_id]
                        rbbox = frame_obj_data["object_data"]["rbbox"][0]["val"]
                        r_current = rbbox[4]

                        if abs(last_valid_angle - r_current) > self.rotation_threshold:
                            self._apply_rotation_adjustment(frame_obj_data, last_valid_angle)
                            self.rotations_copied += 1
                    break

                r_new = self._calculate_smoothed_rotation(
                    frames, obj_id, current_frame, frame_list_sorted, i
                )

                current_frame_str = str(current_frame)
                frame_obj_data = frames[current_frame_str]["objects"][obj_id]
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
                        r_new = self._apply_temporal_smoothing(previous_smoothed_angle, r_new)

                    last_valid_angle = r_new
                    previous_smoothed_angle = r_new

                if abs(r_new - r_current) > self.rotation_threshold:
                    self._apply_rotation_adjustment(frame_obj_data, r_new)
                    self.rotations_adjusted += 1
                else:
                    self.rotations_kept += 1

        return openlabel_data

    def _apply_rotation_adjustment(self, frame_obj_data: Dict[str, Any], r_new: float) -> None:
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
                vec_item["val"].append("markit_housekeeping(rot)")
                annotator_found = True
            elif vec_item.get("name") == "confidence":
                vec_item["val"].append(0.8888)
                confidence_found = True

        if not annotator_found:
            vec_list.insert(0, {
                "name": "annotator",
                "val": ["markit_housekeeping(rot)"]
            })

        if not confidence_found:
            vec_list.append({
                "name": "confidence",
                "val": [0.8888]
            })

    def _apply_temporal_smoothing(self, prev_angle: float, curr_angle: float) -> float:
        """Apply temporal smoothing between consecutive frames using exponential moving average.

        Args:
            prev_angle: Previous frame's smoothed angle (radians, continuous)
            curr_angle: Current frame's calculated angle (radians)

        Returns:
            Temporally smoothed angle (radians, continuous)
        """
        # Handle angle wraparound: ensure angles are within π of each other
        angle_diff = curr_angle - prev_angle

        # Normalize to [-π, π] to get shortest rotation path
        angle_diff = normalize_angle_to_pi(angle_diff)

        # Apply exponential moving average
        smoothed = prev_angle + (1 - self.temporal_smoothing) * angle_diff

        # Rebase if needed (only if |angle| > 2π)
        smoothed = rebase_angle_if_needed(smoothed)

        return smoothed

    def _calculate_smoothed_rotation(self, frames: Dict[str, Any], obj_id: str,
                                     current_frame: int, frame_list_sorted: List[int],
                                     current_idx: int) -> Optional[float]:
        """Calculate smoothed rotation using bidirectional weighted average with movement threshold.

        This improved version:
        - Uses CORRECTED arctan2(delta_y, delta_x) argument order
        - Looks both backward (1-4 frames) and forward (1-8 frames)
        - Applies minimum movement threshold to filter noise
        - Handles angle wraparound with np.unwrap
        - Uses normalized weighting regardless of available frames
        - Validates against bbox aspect ratio

        Args:
            frames: Frame data
            obj_id: Object ID
            current_frame: Current frame index
            frame_list_sorted: Sorted list of frames for this object
            current_idx: Index in frame_list_sorted

        Returns:
            Smoothed rotation angle in radians, or None if insufficient movement
        """
        current_frame_str = str(current_frame)
        current_obj = frames[current_frame_str]["objects"][obj_id]
        current_rbbox = current_obj["object_data"]["rbbox"][0]["val"]
        x_current, y_current = current_rbbox[0], current_rbbox[1]
        w_current, h_current = current_rbbox[2], current_rbbox[3]

        angles = []
        weights = []
        distances = []

        # Look backward (1-4 frames) with lower weights
        for lookback in range(1, 5):
            if current_idx - lookback < 0:
                break

            past_frame = frame_list_sorted[current_idx - lookback]
            past_frame_str = str(past_frame)
            past_obj = frames[past_frame_str]["objects"][obj_id]
            past_rbbox = past_obj["object_data"]["rbbox"][0]["val"]
            x_past, y_past = past_rbbox[0], past_rbbox[1]

            delta_x = x_current - x_past
            delta_y = y_current - y_past
            distance = np.sqrt(delta_x**2 + delta_y**2)

            # Apply minimum movement threshold (Fix #2)
            if distance >= self.min_movement_pixels:
                # FIXED: Correct arctan2 argument order (Fix #1)
                angle = np.arctan2(delta_y, delta_x)
                angles.append(angle)
                weights.append(2.0 / lookback)  # Lower weight for past frames
                distances.append(distance)

        # Look forward (1-8 frames) with higher weights
        for lookahead in range(1, 9):
            if current_idx + lookahead >= len(frame_list_sorted):
                break

            future_frame = frame_list_sorted[current_idx + lookahead]
            future_frame_str = str(future_frame)
            future_obj = frames[future_frame_str]["objects"][obj_id]
            future_rbbox = future_obj["object_data"]["rbbox"][0]["val"]
            x_future, y_future = future_rbbox[0], future_rbbox[1]

            delta_x = x_future - x_current
            delta_y = y_future - y_current
            distance = np.sqrt(delta_x**2 + delta_y**2)

            # Apply minimum movement threshold (Fix #2)
            if distance >= self.min_movement_pixels:
                # FIXED: Correct arctan2 argument order (Fix #1)
                angle = np.arctan2(delta_y, delta_x)
                angles.append(angle)
                weights.append(9.0 - lookahead)  # Higher weight for near-future frames
                distances.append(distance)

        if not angles:
            return None

        # Calculate average movement distance to assess confidence
        avg_distance = sum(distances) / len(distances)

        # For very slow movement, don't adjust - insufficient data to determine direction
        # Adjusting on slow movement often makes things worse due to noise
        if avg_distance < 10.0:
            return None

        # Handle angle wraparound (Fix #4)
        angles_array = np.array(angles)
        angles_unwrapped = np.unwrap(angles_array)

        # Circular averaging with normalized weights
        weighted_sin = sum(np.sin(angle) * weight for angle, weight in zip(angles_unwrapped, weights))
        weighted_cos = sum(np.cos(angle) * weight for angle, weight in zip(angles_unwrapped, weights))
        weight_sum = sum(weights)

        avg_sin = weighted_sin / weight_sum
        avg_cos = weighted_cos / weight_sum

        # Movement direction in OpenLabel format (clockwise from horizontal right)
        # arctan2(delta_y, delta_x) with Y pointing down gives correct clockwise angle
        movement_direction = float(np.arctan2(avg_sin, avg_cos))

        # Determine correct rotation based on which axis should align with movement
        # New semantic format: width is always the long axis
        # - rotation = angle of the WIDTH axis (long axis)
        # - For vehicles, the long axis should align with movement direction

        aspect_ratio = max(w_current, h_current) / max(min(w_current, h_current), 1.0)

        if aspect_ratio > 1.5:  # Elongated object (likely vehicle)
            # Width (long axis) should align with movement direction
            correct_rotation = movement_direction
        else:
            # Not elongated (circular or square) - use movement direction
            correct_rotation = movement_direction

        # No normalization needed - angles are continuous
        return correct_rotation

    def get_statistics(self) -> Dict[str, Any]:
        """Get rotation adjustment statistics.

        Returns:
            Dictionary with rotation adjustment statistics
        """
        return {
            'objects_processed': self.objects_processed,
            'rotations_adjusted': self.rotations_adjusted,
            'rotations_kept': self.rotations_kept,
            'rotations_copied': self.rotations_copied
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

        if not hasattr(self, 'frame_width') or not hasattr(self, 'frame_height'):
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

                is_first_appearance = (i == 0)
                is_last_appearance = (i == len(frame_list_sorted) - 1)

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
                    vec_list.append({
                        "name": "suddenappear",
                        "val": sudden_appear_frames
                    })

                if sudden_disappear_frames:
                    vec_list.append({
                        "name": "suddendisappear",
                        "val": sudden_disappear_frames
                    })

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
            'objects_with_events': len(self.objects_with_events),
            'sudden_appear_count': self.sudden_appear_count,
            'sudden_disappear_count': self.sudden_disappear_count
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
                {
                    "frame_start": frame_start,
                    "frame_end": frame_end
                }
            ]
            self.intervals_added += 1

        return openlabel_data

    def get_statistics(self) -> Dict[str, Any]:
        """Get frame interval addition statistics.

        Returns:
            Dictionary with frame interval statistics
        """
        return {
            'intervals_added': self.intervals_added,
            'intervals_skipped_existing': self.intervals_skipped_existing,
            'intervals_skipped_no_frames': self.intervals_skipped_no_frames
        }


class StaticObjectRemovalPass(PostprocessingPass):
    """Remove DynamicObject instances that don't move beyond threshold.
       NOTE: This pass will remove e.g. parked cars, pedestrians standing still, etc. If this
       is not desired, do not use this pass or use --static-mark to mark instead of remove.
    """

    def __init__(self, static_threshold: int = 5, mark_only: bool = False):
        """Initialize static object removal pass.

        Args:
            static_threshold: Movement threshold in pixels (default: 5)
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
        if not hasattr(self, 'ontology_path') or not self.ontology_path:
            logger.warning("StaticObjectRemovalPass: Ontology path not set, skipping")
            return openlabel_data

        # Import ontology functions (done here to avoid circular imports)
        import sys
        import os
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from common.ontology import get_class_by_label

        # Check if ontology file exists
        if not os.path.exists(self.ontology_path):
            logger.warning(f"StaticObjectRemovalPass: Ontology file not found: {self.ontology_path}")
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
                class_info = get_class_by_label(self.ontology_path, obj_type, case_sensitive=False)

                # Skip if not found in ontology
                if class_info is None:
                    logger.debug(f"StaticObjectRemovalPass: Class '{obj_type}' not found in ontology, skipping object {obj_id}")
                    continue

                # Check if top-level class is DynamicObject
                top_level_class_name = class_info.get('top_level_class_name')
                if top_level_class_name != 'DynamicObject':
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
                        logger.debug(f"StaticObjectRemovalPass: Error extracting position for object {obj_id} in frame {frame_idx}: {e}")
                        continue

                # Check if we have position data
                if len(x_positions) == 0 or len(y_positions) == 0:
                    continue

                # Calculate max-min movement in each dimension
                delta_x = max(x_positions) - min(x_positions)
                delta_y = max(y_positions) - min(y_positions)

                # Check if both dimensions are below threshold
                if delta_x <= self.static_threshold and delta_y <= self.static_threshold:
                    objects_to_remove.append(obj_id)

                    # Store first frame for marking
                    first_frame = min(frame_list)

                    if self.mark_only:
                        self.marking_details.append({
                            'object_id': obj_id,
                            'type': obj_type,
                            'delta_x': delta_x,
                            'delta_y': delta_y,
                            'frame_count': len(frame_list),
                            'first_frame': first_frame
                        })
                        logger.info(
                            f"Marked static object {obj_id} (type: {obj_type}) - "
                            f"movement: dx={delta_x}px, dy={delta_y}px, frames={len(frame_list)}"
                        )
                    else:
                        self.removal_details.append({
                            'object_id': obj_id,
                            'type': obj_type,
                            'delta_x': delta_x,
                            'delta_y': delta_y,
                            'frame_count': len(frame_list)
                        })
                        logger.info(
                            f"Removed static object {obj_id} (type: {obj_type}) - "
                            f"movement: dx={delta_x}px, dy={delta_y}px, frames={len(frame_list)}"
                        )

            except Exception as e:
                logger.warning(f"StaticObjectRemovalPass: Error processing object {obj_id}: {e}")
                continue

        # Mark or remove objects
        if self.mark_only:
            # Mark objects by adding "staticdynamic" annotation
            for detail in self.marking_details:
                obj_id = detail['object_id']
                first_frame = detail['first_frame']

                if obj_id not in objects:
                    continue

                if "object_data" not in objects[obj_id]:
                    objects[obj_id]["object_data"] = {}

                if "vec" not in objects[obj_id]["object_data"]:
                    objects[obj_id]["object_data"]["vec"] = []

                vec_list = objects[obj_id]["object_data"]["vec"]
                vec_list.append({
                    "name": "staticdynamic",
                    "val": [first_frame]
                })

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
                'objects_checked': self.objects_checked,
                'objects_marked': self.objects_marked
            }
        else:
            return {
                'objects_checked': self.objects_checked,
                'objects_removed': self.objects_removed,
                'frames_modified': self.frames_modified
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

        logger.info(f"AngleNormalization: Normalized {self.angles_normalized} angles to [0, 2π) range")

        return openlabel_data

    def get_statistics(self) -> Dict[str, Any]:
        """Get angle normalization statistics.

        Returns:
            Dictionary with normalization statistics
        """
        return {
            'angles_normalized': self.angles_normalized
        }
