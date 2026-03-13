"""
SuddenPass - Sudden event detection postprocessing pass.
"""

import logging
from typing import Any, Dict
from collections import defaultdict

from ..base import PostprocessingPass

logger = logging.getLogger(__name__)


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
