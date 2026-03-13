"""
GapDetectionPass - Gap detection postprocessing pass.
"""

import logging
from typing import Any, Dict
from collections import defaultdict

from ..base import PostprocessingPass

logger = logging.getLogger(__name__)


class GapDetectionPass(PostprocessingPass):
    """Detect gaps in object ID frame sequences."""

    def __init__(self):
        self.gaps_detected = {}
        self.objects_with_gaps = set()

    def process(self, openlabel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect gaps in object tracking sequences.

        Args:
            openlabel_data: Complete OpenLabel data structure

        Returns:
            Unmodified OpenLabel data (detection only, no fixes yet)
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
            gaps = []

            for i in range(len(frame_list_sorted) - 1):
                current_frame = frame_list_sorted[i]
                next_frame = frame_list_sorted[i + 1]
                gap_size = next_frame - current_frame - 1

                if gap_size > 0:
                    gaps.append(
                        {
                            "start_frame": current_frame,
                            "end_frame": next_frame,
                            "gap_size": gap_size,
                        }
                    )

            if gaps:
                self.gaps_detected[obj_id] = {
                    "frame_range": (frame_list_sorted[0], frame_list_sorted[-1]),
                    "total_frames": len(frame_list_sorted),
                    "gaps": gaps,
                }
                self.objects_with_gaps.add(obj_id)

                logger.warning(
                    f"Object ID {obj_id}: detected {len(gaps)} gap(s) in frame sequence "
                    f"[{frame_list_sorted[0]}-{frame_list_sorted[-1]}]"
                )
                for gap in gaps:
                    logger.warning(
                        f"  Gap: frames {gap['start_frame']} -> {gap['end_frame']} "
                        f"(missing {gap['gap_size']} frame(s))"
                    )

        return openlabel_data

    def get_statistics(self) -> Dict[str, Any]:
        """Get gap detection statistics.

        Returns:
            Dictionary with gap detection statistics
        """
        total_gaps = sum(len(info["gaps"]) for info in self.gaps_detected.values())

        return {
            "objects_with_gaps": len(self.objects_with_gaps),
            "total_gaps_detected": total_gaps,
            "gap_details": self.gaps_detected,
        }
