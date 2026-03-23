"""
FrameIntervalPass - Frame interval calculation postprocessing pass.
"""

from typing import Any, Dict
from collections import defaultdict

from ..base import PostprocessingPass


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
