"""
RotationTemporalSmoothingPass - Rotation temporal smoothing postprocessing pass.
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
                    update_housekeeping_annotator(frames[frame_str]["objects"][obj_id], "rotsmooth")
                else:
                    # Large change - accept it (might be real)
                    smoothed_rotation = current_rotation
                    self.rotations_kept += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get rotation smoothing statistics."""
        return {
            "objects_processed": self.objects_processed,
            "rotations_smoothed": self.rotations_smoothed,
            "rotations_kept": self.rotations_kept,
        }
