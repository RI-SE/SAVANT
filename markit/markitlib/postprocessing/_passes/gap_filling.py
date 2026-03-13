"""
GapFillingPass - Gap filling postprocessing pass.
"""

import logging
from typing import Any, Dict
from collections import defaultdict

from ..base import PostprocessingPass
from ...utils import (
    normalize_angle_to_pi,
)
from ._common import HOUSEKEEPING_CONFIDENCE

logger = logging.getLogger(__name__)


class GapFillingPass(PostprocessingPass):
    """Fill gaps in object ID frame sequences by interpolating positions."""

    def __init__(self, max_gap_size: int = 30):
        """Initialize gap filling pass.

        Args:
            max_gap_size: Maximum gap size (in frames) to fill. Gaps larger than
                this are left unfilled — they likely represent separate tracking
                segments rather than brief detection dropouts.
        """
        self.max_gap_size = max_gap_size
        self.gaps_filled = 0
        self.gaps_skipped = 0
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
                    if gap_size > self.max_gap_size:
                        logger.debug(
                            f"GapFilling: skipping {obj_id} gap frames "
                            f"{frame_before}->{frame_after} ({gap_size} frames "
                            f"> max {self.max_gap_size})"
                        )
                        self.gaps_skipped += 1
                        continue
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
                        {"name": "confidence", "val": [HOUSEKEEPING_CONFIDENCE]},
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
            "gaps_skipped": self.gaps_skipped,
            "frames_added": self.frames_added,
        }
