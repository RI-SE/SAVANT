"""
ShortDurationPass - Short duration object removal postprocessing pass.
"""

import logging
from typing import Any, Dict, List
from collections import defaultdict

from ..base import PostprocessingPass
from ._common import _get_object_source_engine

logger = logging.getLogger(__name__)


class ShortDurationPass(PostprocessingPass):
    """Delete short-lived DynamicObjects that appear for fewer than a minimum number of frames.

    Short-lived objects are typically noise from the optical flow engine: contours that
    briefly appear and disappear without corresponding to a real physical object. Deleting
    them before other passes reduces noise in the annotation and speeds up subsequent steps.

    When oflow_only=True (default), only optical-flow-origin objects are candidates, which
    avoids accidentally removing brief but legitimate YOLO detections (e.g. a vehicle
    visible for only a few frames at the edge of the scene).
    """

    def __init__(self, min_frames: int = 15, oflow_only: bool = True):
        """Initialize short duration pass.

        Args:
            min_frames: Objects with fewer than this many frames are deleted.
                At 30 fps, 15 frames ≈ 0.5 s — a reasonable minimum for a real
                object to be worth annotating. Noise tracks from optical flow
                typically last fewer than 10 frames.
            oflow_only: If True, only remove optical-flow-origin objects (default: True).
                Set to False to apply to all detection engines.
        """
        self.min_frames = min_frames
        self.oflow_only = oflow_only
        self.objects_checked = 0
        self.objects_removed = 0
        self.frames_modified = 0
        self.removal_details = []

    def process(self, openlabel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove objects that appear for fewer than min_frames frames.

        Args:
            openlabel_data: Complete OpenLabel data structure

        Returns:
            Modified OpenLabel data with short-duration objects removed
        """
        frames = openlabel_data.get("openlabel", {}).get("frames", {})
        objects = openlabel_data.get("openlabel", {}).get("objects", {})

        # Build object-to-frames mapping
        object_frame_map: Dict[str, List[int]] = defaultdict(list)
        for frame_idx_str, frame_data in frames.items():
            frame_idx = int(frame_idx_str)
            for obj_id_str in frame_data.get("objects", {}).keys():
                object_frame_map[obj_id_str].append(frame_idx)

        objects_to_remove = []

        for obj_id in list(objects.keys()):
            if self.oflow_only:
                source_engine = _get_object_source_engine(obj_id, object_frame_map, frames)
                if source_engine != "oflow":
                    continue

            self.objects_checked += 1
            frame_count = len(object_frame_map.get(obj_id, []))

            if frame_count < self.min_frames:
                objects_to_remove.append(obj_id)
                self.removal_details.append(
                    {
                        "object_id": obj_id,
                        "type": objects[obj_id].get("type", ""),
                        "frame_count": frame_count,
                    }
                )
                logger.info(
                    f"ShortDurationPass: Removing {obj_id} "
                    f"(type: {objects[obj_id].get('type', '')}, "
                    f"frames={frame_count} < {self.min_frames})"
                )

        for obj_id in objects_to_remove:
            if obj_id in objects:
                del objects[obj_id]
                self.objects_removed += 1

            for frame_idx_str, frame_data in frames.items():
                frame_objects = frame_data.get("objects", {})
                if obj_id in frame_objects:
                    del frame_objects[obj_id]
                    self.frames_modified += 1

        return openlabel_data

    def get_statistics(self) -> Dict[str, Any]:
        """Get short duration pass statistics.

        Returns:
            Dictionary with removal statistics
        """
        return {
            "objects_checked": self.objects_checked,
            "objects_removed": self.objects_removed,
            "frames_modified": self.frames_modified,
        }
