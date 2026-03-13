"""
FirstDetectionRefinementPass - First detection refinement postprocessing pass.
"""

import logging
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict
import numpy as np

from ..base import PostprocessingPass

logger = logging.getLogger(__name__)


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
