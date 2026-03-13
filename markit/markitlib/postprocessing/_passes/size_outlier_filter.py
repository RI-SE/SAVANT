"""
SizeOutlierFilterPass - Size outlier filter postprocessing pass.
"""

import logging
from typing import Any, Dict, List
from collections import defaultdict
import numpy as np

from ..base import PostprocessingPass
from ._common import update_housekeeping_annotator

logger = logging.getLogger(__name__)


class SizeOutlierFilterPass(PostprocessingPass):
    """Detect and fix frames where object size is an outlier compared to its history.

    Uses MAD (Median Absolute Deviation) to identify outlier frames for each object
    individually. This catches motion streak artifacts (suddenly much larger area)
    without rejecting legitimately large vehicles like buses.

    Outlier frames have their size replaced with the median w/h from non-outlier
    frames. This handles cases where motion streaks persist over multiple consecutive
    frames better than neighbor-based interpolation.
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
        self.objects_all_outliers = 0

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
            + (f", {self.objects_all_outliers} objects had all frames as outliers"
               if self.objects_all_outliers > 0 else "")
        )

        return openlabel_data

    def _fix_object_outliers(
        self,
        frames: Dict[str, Any],
        obj_id: str,
        frame_list: List[int],
    ) -> None:
        """Fix size outliers for a single object.

        Uses median w/h from non-outlier frames instead of neighbor interpolation.
        This handles consecutive outlier frames (e.g., persistent motion streaks).

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

        # Identify outliers based on area
        outlier_indices = set()
        for i, s in enumerate(size_data):
            deviation = abs(s["area"] - median_area) / mad
            if deviation > self.outlier_threshold:
                outlier_indices.add(i)

        if not outlier_indices:
            return

        # Get non-outlier frames
        non_outlier_data = [s for i, s in enumerate(size_data) if i not in outlier_indices]

        if not non_outlier_data:
            # All frames are outliers - can't fix, flag object
            self.objects_all_outliers += 1
            logger.warning(
                f"SizeOutlierFilter: Object {obj_id} has all {len(size_data)} frames "
                "flagged as outliers - cannot fix"
            )
            return

        # Calculate median w and h from non-outlier frames
        median_w = float(np.median([s["w"] for s in non_outlier_data]))
        median_h = float(np.median([s["h"] for s in non_outlier_data]))

        # Fix outliers by replacing with median w/h
        for outlier_idx in outlier_indices:
            frame_idx = size_data[outlier_idx]["frame_idx"]
            frame_str = str(frame_idx)

            # Apply fix - use median w/h from non-outliers
            rbbox = frames[frame_str]["objects"][obj_id]["object_data"]["rbbox"][0]["val"]
            rbbox[2] = median_w
            rbbox[3] = median_h

            update_housekeeping_annotator(frames[frame_str]["objects"][obj_id], "outlier")
            self.outliers_fixed += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get size outlier filter statistics."""
        return {
            "objects_processed": self.objects_processed,
            "outliers_fixed": self.outliers_fixed,
            "objects_all_outliers": self.objects_all_outliers,
        }
