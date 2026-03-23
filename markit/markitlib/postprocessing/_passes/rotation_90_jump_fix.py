"""
Rotation90JumpFixPass - 90-degree rotation jump fix postprocessing pass.
"""

import logging
from typing import Any, Dict, List
from collections import defaultdict
import numpy as np

from ..base import PostprocessingPass
from ...utils import (
    normalize_angle_to_pi,
)
from ._common import update_housekeeping_annotator

logger = logging.getLogger(__name__)


class Rotation90JumpFixPass(PostprocessingPass):
    """Fix 90° and 180° rotation jumps caused by minAreaRect ambiguity.

    minAreaRect can return equivalent bounding boxes rotated by 90° with w/h swapped,
    or by 180° with the same w/h. This pass detects such jumps and corrects them:
    - 90° jumps: swap w/h and adjust rotation by ±90°
    - 180° jumps: adjust rotation by ±180° (no w/h swap needed)
    """

    def __init__(
        self,
        jump_threshold_low: float = 70.0,
        jump_threshold_high: float = 110.0,
        jump_180_threshold: float = 160.0,
    ):
        """Initialize rotation jump fix pass.

        Args:
            jump_threshold_low: Lower bound for 90° jump detection in degrees (default: 70)
            jump_threshold_high: Upper bound for 90° jump detection in degrees (default: 110)
            jump_180_threshold: Lower bound for 180° jump detection in degrees (default: 160)
        """
        self.jump_threshold_low = np.radians(jump_threshold_low)
        self.jump_threshold_high = np.radians(jump_threshold_high)
        self.jump_180_threshold = np.radians(jump_180_threshold)
        self.jumps_fixed = 0
        self.jumps_180_fixed = 0
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
            f"fixed {self.jumps_fixed} 90° jumps ({self.wh_swaps} w/h swaps), "
            f"{self.jumps_180_fixed} 180° jumps"
        )

        return openlabel_data

    def _fix_object_jumps(
        self,
        frames: Dict[str, Any],
        obj_id: str,
        frame_list: List[int],
    ) -> None:
        """Fix 90° and 180° jumps for a single object.

        For any jump > 70°, tries multiple corrections and picks the best:
        - ±90° with w/h swap (for minAreaRect w/h ambiguity)
        - ±180° without swap (for minAreaRect angle flip)

        Also handles the case where the first frame is the outlier by doing
        a second pass to check if F1 should be corrected based on F2+.

        Args:
            frames: Frame data dictionary
            obj_id: Object ID string
            frame_list: Sorted list of frame indices for this object
        """
        if len(frame_list) < 2:
            return

        # Forward pass: fix frames relative to previous frame
        prev_rotation = None
        for frame_idx in frame_list:
            frame_str = str(frame_idx)
            rbbox = frames[frame_str]["objects"][obj_id]["object_data"]["rbbox"][0]["val"]
            x, y, w, h, r = rbbox

            if prev_rotation is not None:
                r, w, h = self._try_fix_jump(
                    frames, obj_id, frame_str, rbbox, r, w, h, prev_rotation
                )

            prev_rotation = r

        # Second pass: check if first frame should be corrected
        # If F1→F2 still has a large jump after forward pass, F1 might be the outlier
        first_frame_str = str(frame_list[0])
        second_frame_str = str(frame_list[1])

        rbbox_first = frames[first_frame_str]["objects"][obj_id]["object_data"]["rbbox"][0]["val"]
        rbbox_second = frames[second_frame_str]["objects"][obj_id]["object_data"]["rbbox"][0]["val"]

        r_first = rbbox_first[4]
        r_second = rbbox_second[4]
        w_first, h_first = rbbox_first[2], rbbox_first[3]

        diff = normalize_angle_to_pi(r_first - r_second)
        abs_diff = abs(diff)

        if abs_diff > self.jump_threshold_low:
            # First frame might be wrong - try to fix it based on second frame
            sign = -1 if diff > 0 else 1

            # Try 90° correction (with w/h swap)
            r_90 = r_first + sign * (np.pi / 2)
            diff_90 = abs(normalize_angle_to_pi(r_90 - r_second))

            # Try 180° correction (no w/h swap)
            r_180 = r_first + sign * np.pi
            diff_180 = abs(normalize_angle_to_pi(r_180 - r_second))

            # Pick the best correction for first frame
            if diff_90 < abs_diff and diff_90 <= diff_180:
                rbbox_first[2] = h_first
                rbbox_first[3] = w_first
                rbbox_first[4] = r_90
                self.jumps_fixed += 1
                self.wh_swaps += 1
                update_housekeeping_annotator(frames[first_frame_str]["objects"][obj_id], "90fix")
            elif diff_180 < abs_diff:
                rbbox_first[4] = r_180
                self.jumps_180_fixed += 1
                update_housekeeping_annotator(frames[first_frame_str]["objects"][obj_id], "90fix")

    def _try_fix_jump(
        self,
        frames: Dict[str, Any],
        obj_id: str,
        frame_str: str,
        rbbox: List[float],
        r: float,
        w: float,
        h: float,
        prev_rotation: float,
    ) -> tuple:
        """Try to fix a rotation jump, return (new_r, new_w, new_h).

        Args:
            frames: Frame data dictionary
            obj_id: Object ID string
            frame_str: Current frame string
            rbbox: Current rbbox list (modified in place)
            r: Current rotation
            w: Current width
            h: Current height
            prev_rotation: Previous frame's rotation

        Returns:
            Tuple of (rotation, width, height) after any correction
        """
        diff = normalize_angle_to_pi(r - prev_rotation)
        abs_diff = abs(diff)

        if abs_diff <= self.jump_threshold_low:
            return r, w, h

        sign = -1 if diff > 0 else 1

        # Try 90° correction (with w/h swap)
        r_90 = r + sign * (np.pi / 2)
        diff_90 = abs(normalize_angle_to_pi(r_90 - prev_rotation))

        # Try 180° correction (no w/h swap)
        r_180 = r + sign * np.pi
        diff_180 = abs(normalize_angle_to_pi(r_180 - prev_rotation))

        # Pick the best correction
        if diff_90 < abs_diff and diff_90 <= diff_180:
            rbbox[2] = h
            rbbox[3] = w
            rbbox[4] = r_90
            self.jumps_fixed += 1
            self.wh_swaps += 1
            update_housekeeping_annotator(frames[frame_str]["objects"][obj_id], "90fix")
            return r_90, h, w
        elif diff_180 < abs_diff:
            rbbox[4] = r_180
            self.jumps_180_fixed += 1
            update_housekeeping_annotator(frames[frame_str]["objects"][obj_id], "90fix")
            return r_180, w, h

        return r, w, h

    def get_statistics(self) -> Dict[str, Any]:
        """Get rotation jump fix statistics."""
        return {
            "objects_processed": self.objects_processed,
            "jumps_90_fixed": self.jumps_fixed,
            "jumps_180_fixed": self.jumps_180_fixed,
            "wh_swaps": self.wh_swaps,
        }
