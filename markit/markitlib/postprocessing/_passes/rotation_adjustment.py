"""
RotationAdjustmentPass - Rotation adjustment postprocessing pass.
"""

from typing import Any, Dict, List, Optional
from collections import defaultdict
import numpy as np

from ..base import PostprocessingPass
from ...utils import (
    normalize_angle_to_pi,
    rebase_angle_if_needed,
)
from ._common import update_housekeeping_annotator


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
        self.first_frame_flips = 0
        self.objects_processed = 0
        self.rotation_threshold = rotation_threshold
        self.min_movement_pixels = min_movement_pixels
        self.min_total_movement = min_total_movement
        self.temporal_smoothing = temporal_smoothing
        self.max_rotation_change = max_rotation_change
        self.aspect_instability_window = aspect_instability_window

    def _check_first_frame_offset(
        self,
        frames: Dict[str, Any],
        obj_id: str,
        frame_list_sorted: List[int],
    ) -> bool:
        """Check if first frame angle is ~180° off from initial movement direction.

        When the first detection's angle is 180° off from the true heading,
        temporal smoothing causes a slow drift as it clamps the correction to
        max_rotation_change per frame. This method detects such cases by comparing
        the first frame's angle to movement direction from the first few frames.

        If detected, flips the first frame's angle by 180° so temporal smoothing
        starts from the correct orientation.

        Args:
            frames: Frame data
            obj_id: Object ID
            frame_list_sorted: Sorted list of frame indices

        Returns:
            True if first frame was flipped, False otherwise
        """
        if len(frame_list_sorted) < 3:
            return False

        # Get first non-gap-filled frame
        first_frame_str = str(frame_list_sorted[0])
        first_obj = frames[first_frame_str]["objects"][obj_id]

        if self._is_gap_filled(first_obj):
            return False

        first_rbbox = first_obj["object_data"]["rbbox"][0]["val"]
        x_first, y_first = first_rbbox[0], first_rbbox[1]
        w_first, h_first = first_rbbox[2], first_rbbox[3]
        r_first = first_rbbox[4]

        # Calculate initial movement direction from first few frames
        movement_vectors = []
        for i in range(1, min(5, len(frame_list_sorted))):
            frame_str = str(frame_list_sorted[i])
            frame_obj = frames[frame_str]["objects"][obj_id]

            if self._is_gap_filled(frame_obj):
                continue

            rbbox = frame_obj["object_data"]["rbbox"][0]["val"]
            x, y = rbbox[0], rbbox[1]

            dx = x - x_first
            dy = y - y_first
            distance = np.sqrt(dx**2 + dy**2)

            if distance >= self.min_movement_pixels:
                movement_vectors.append((dx, dy, distance))

        if not movement_vectors:
            return False

        # Calculate weighted average movement direction
        total_dx = sum(v[0] * v[2] for v in movement_vectors)
        total_dy = sum(v[1] * v[2] for v in movement_vectors)
        total_weight = sum(v[2] for v in movement_vectors)

        if total_weight < self.min_total_movement:
            return False

        movement_angle = np.arctan2(total_dy / total_weight, total_dx / total_weight)

        # Determine expected rotation based on aspect ratio
        aspect_ratio = max(w_first, h_first) / max(min(w_first, h_first), 1.0)
        if aspect_ratio > 1.5 and h_first > w_first:
            # Height is long axis, expected rotation is movement + 90°
            expected_rotation = movement_angle + np.pi / 2
        else:
            expected_rotation = movement_angle

        # Check if first frame angle is ~180° off
        diff = normalize_angle_to_pi(r_first - expected_rotation)
        if abs(abs(diff) - np.pi) < np.radians(30):  # Within 30° of 180° off
            # Flip by 180°
            new_angle = r_first + np.pi
            new_angle = rebase_angle_if_needed(new_angle)
            first_rbbox[4] = new_angle
            self.first_frame_flips += 1
            return True

        return False

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

            # Check if first frame angle is ~180° off from movement direction
            # If so, flip it before temporal smoothing starts
            self._check_first_frame_offset(frames, obj_id, frame_list_sorted)

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
        import re

        vec_list = frame_obj_data.get("object_data", {}).get("vec", [])
        for vec_item in vec_list:
            if vec_item.get("name") == "annotator":
                for val in vec_item.get("val", []):
                    # Check for gap tag in combined format: markit_housekeeping(gap,rot,...)
                    match = re.match(r"markit_housekeeping\(([^)]*)\)", val)
                    if match and "gap" in match.group(1).split(","):
                        return True
        return False

    def _apply_rotation_adjustment(
        self, frame_obj_data: Dict[str, Any], r_new: float
    ) -> None:
        """Apply rotation adjustment and update annotator/confidence.

        Sets the rotation to r_new and normalizes the bbox so that
        width >= height (the heading marker always points along the
        long axis of elongated objects).

        Args:
            frame_obj_data: Frame object data
            r_new: New rotation value
        """
        rbbox = frame_obj_data["object_data"]["rbbox"][0]["val"]

        # Rebase angle if needed (only if |angle| > 2π)
        adjusted_rotation = rebase_angle_if_needed(r_new)

        # Update rotation
        rbbox[4] = adjusted_rotation

        # Normalize: ensure width >= height so the heading marker
        # (drawn along the width axis) points along the long dimension.
        if rbbox[3] > rbbox[2]:  # h > w
            rbbox[2], rbbox[3] = rbbox[3], rbbox[2]  # swap w and h

        update_housekeeping_annotator(frame_obj_data, "rot")

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
        _, _ = current_rbbox[2], current_rbbox[3]

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

        # The heading marker is drawn along the width axis (direction θ).
        # Always set θ = movement_direction so the heading marker points
        # in the direction of travel. The caller will swap w/h if needed
        # to ensure the width axis is the long dimension.
        return movement_direction

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
            "first_frame_flips": self.first_frame_flips,
        }
