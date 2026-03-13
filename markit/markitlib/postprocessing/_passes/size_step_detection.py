"""
SizeStepDetectionPass - Size step detection postprocessing pass.
"""

import logging
from typing import Any, Dict, List
from collections import defaultdict
import numpy as np

from ..base import PostprocessingPass

logger = logging.getLogger(__name__)


class SizeStepDetectionPass(PostprocessingPass):
    """Detect persistent size changes (step changes) in object trajectories.

    Unlike outlier detection which catches single-frame spikes, this pass detects
    when an object's size changes persistently mid-trajectory (e.g., detection
    shifts to different part of object, or object changes orientation significantly).

    Transition frames are flagged with 'size_step' annotation for manual review.
    No automatic correction is applied since step changes may be legitimate.
    """

    def __init__(
        self,
        step_threshold: float = 0.3,
        min_segment_frames: int = 5,
        min_frames: int = 10,
    ):
        """Initialize size step detection pass.

        Args:
            step_threshold: Minimum relative change in median area to detect step (default: 0.3 = 30%)
            min_segment_frames: Minimum frames on each side of transition to consider (default: 5)
            min_frames: Minimum total frames for step detection (default: 10)
        """
        self.step_threshold = step_threshold
        self.min_segment_frames = min_segment_frames
        self.min_frames = min_frames
        self.objects_processed = 0
        self.steps_detected = 0
        self.objects_with_steps = set()

    def process(self, openlabel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect size step changes in object trajectories.

        Args:
            openlabel_data: Complete OpenLabel data structure

        Returns:
            Modified OpenLabel data with size_step annotations added
        """
        frames = openlabel_data.get("openlabel", {}).get("frames", {})
        objects = openlabel_data.get("openlabel", {}).get("objects", {})

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
            step_frames = self._detect_steps(frames, obj_id, frame_list_sorted)

            if step_frames:
                self.objects_with_steps.add(obj_id)
                self.steps_detected += len(step_frames)
                self._add_annotation(objects, obj_id, step_frames)

        logger.info(
            f"SizeStepDetection: Processed {self.objects_processed} objects, "
            f"detected {self.steps_detected} step changes in {len(self.objects_with_steps)} objects"
        )

        return openlabel_data

    def _detect_steps(
        self,
        frames: Dict[str, Any],
        obj_id: str,
        frame_list: List[int],
    ) -> List[int]:
        """Detect step changes in size for a single object.

        Uses a sliding window approach: at each potential transition point,
        compare median area of frames before vs after.

        Args:
            frames: Frame data dictionary
            obj_id: Object ID string
            frame_list: Sorted list of frame indices for this object

        Returns:
            List of frame indices where step changes were detected
        """
        # Collect area data
        areas = []
        for frame_idx in frame_list:
            frame_str = str(frame_idx)
            rbbox = frames[frame_str]["objects"][obj_id]["object_data"]["rbbox"][0]["val"]
            w, h = rbbox[2], rbbox[3]
            areas.append(w * h)

        n = len(areas)
        step_frames = []

        # Slide through potential transition points
        for i in range(self.min_segment_frames, n - self.min_segment_frames):
            # Get median of segment before and after this point
            before_median = np.median(areas[:i])
            after_median = np.median(areas[i:])

            # Calculate relative change
            if before_median > 0:
                relative_change = abs(after_median - before_median) / before_median

                if relative_change > self.step_threshold:
                    # Check if this is a local maximum of the change
                    # (to avoid flagging every frame in a transition region)
                    is_local_max = True

                    if i > self.min_segment_frames:
                        prev_before = np.median(areas[: i - 1])
                        prev_after = np.median(areas[i - 1 :])
                        prev_change = abs(prev_after - prev_before) / max(prev_before, 1)
                        if prev_change >= relative_change:
                            is_local_max = False

                    if i < n - self.min_segment_frames - 1:
                        next_before = np.median(areas[: i + 1])
                        next_after = np.median(areas[i + 1 :])
                        next_change = abs(next_after - next_before) / max(next_before, 1)
                        if next_change > relative_change:
                            is_local_max = False

                    if is_local_max:
                        step_frames.append(frame_list[i])

        return step_frames

    def _add_annotation(
        self,
        objects: Dict[str, Any],
        obj_id: str,
        step_frames: List[int],
    ) -> None:
        """Add size_step annotation to object.

        Args:
            objects: Objects dictionary from OpenLabel
            obj_id: Object ID string
            step_frames: List of frame indices with step changes
        """
        if obj_id not in objects:
            return

        if "object_data" not in objects[obj_id]:
            objects[obj_id]["object_data"] = {}

        if "vec" not in objects[obj_id]["object_data"]:
            objects[obj_id]["object_data"]["vec"] = []

        vec_list = objects[obj_id]["object_data"]["vec"]
        vec_list.append({"name": "size_step", "val": step_frames})

        logger.debug(
            f"SizeStepDetection: Object {obj_id} flagged with size_step at frames {step_frames}"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get size step detection statistics."""
        return {
            "objects_processed": self.objects_processed,
            "steps_detected": self.steps_detected,
            "objects_with_steps": len(self.objects_with_steps),
        }
