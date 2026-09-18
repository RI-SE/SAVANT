"""
PositionalJitterPass - Positional jitter detection/removal postprocessing pass.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from ..base import PostprocessingPass
from ._common import update_housekeeping_annotator

logger = logging.getLogger(__name__)


class PositionalJitterPass(PostprocessingPass):
    """Detect and remove runs of erratic positional jitter in object tracks.

    A real rigid object cannot repeatedly reverse its direction of travel
    frame to frame. This pass looks for short runs of consecutive frames
    where the bbox center's direction of travel changes sharply and
    repeatedly - a signature of tracking noise rather than real motion - and
    removes (or, with mark_only=True, tags) just that jittery frame run,
    preserving the rest of the trajectory.

    Requiring several consecutive sharp turns (not just one) avoids flagging
    legitimate large single jumps (e.g. a fast-moving object, or recovery
    after brief occlusion) and avoids the frame-edge confound, since
    partial-visibility bbox drift near a frame border is a consistent
    one-directional shift, not a reversal pattern.
    """

    def __init__(
        self,
        angle_threshold_deg: float = 100.0,
        min_run_length: int = 3,
        min_speed_px: float = 3.0,
        mark_only: bool = False,
    ):
        """Initialize positional jitter pass.

        Args:
            angle_threshold_deg: Turning angle (degrees) between consecutive
                velocity vectors above which a frame is considered a "sharp
                turn" (default: 100.0).
            min_run_length: Minimum number of consecutive sharp-turn frames
                required to call a run jitter, rather than a single
                plausible direction change (default: 3).
            min_speed_px: Minimum per-frame displacement in pixels for a
                velocity vector to be considered directional; slower
                transitions are ignored since their angle is unstable
                (default: 3.0).
            mark_only: If True, tag jitter frames instead of removing them
                (default: False).
        """
        self.angle_threshold_deg = angle_threshold_deg
        self.min_run_length = min_run_length
        self.min_speed_px = min_speed_px
        self.mark_only = mark_only
        self.objects_checked = 0
        self.objects_with_jitter = 0
        self.jitter_runs_found = 0
        self.frames_removed = 0
        self.frames_marked = 0
        self.decision_details: List[Dict[str, Any]] = []

    def process(self, openlabel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect and remove/mark positional jitter runs per object.

        Args:
            openlabel_data: Complete OpenLabel data structure

        Returns:
            Modified OpenLabel data with jitter frames removed or marked
        """
        frames = openlabel_data.get("openlabel", {}).get("frames", {})
        objects = openlabel_data.get("openlabel", {}).get("objects", {})

        object_frame_map: Dict[str, List[int]] = defaultdict(list)
        for frame_idx_str, frame_data in frames.items():
            frame_idx = int(frame_idx_str)
            for obj_id_str in frame_data.get("objects", {}).keys():
                object_frame_map[obj_id_str].append(frame_idx)

        for obj_id in list(objects.keys()):
            frame_list = sorted(object_frame_map.get(obj_id, []))
            # Need at least min_run_length interior corner positions.
            if len(frame_list) < self.min_run_length + 2:
                continue
            self.objects_checked += 1
            self._process_object(frames, obj_id, frame_list)

        logger.info(
            f"PositionalJitterPass: Checked {self.objects_checked} objects, "
            f"{self.objects_with_jitter} had jitter, "
            f"{self.jitter_runs_found} jitter run(s) found, "
            f"{self.frames_removed} frame(s) removed, "
            f"{self.frames_marked} frame(s) marked"
        )

        return openlabel_data

    def _process_object(
        self, frames: Dict[str, Any], obj_id: str, frame_list: List[int]
    ) -> None:
        """Detect and remove/mark jitter runs for a single object."""
        positions = []
        for frame_idx in frame_list:
            rbbox = frames[str(frame_idx)]["objects"][obj_id]["object_data"]["rbbox"][0]["val"]
            positions.append((frame_idx, rbbox[0], rbbox[1]))

        n = len(positions)
        # vel[k] = displacement vector from positions[k] to positions[k+1],
        # or None if the frames aren't temporally adjacent (a pre-existing
        # detection gap) or the movement is too small to have a stable
        # direction.
        vel: List[Optional[Tuple[float, float]]] = [None] * (n - 1)
        for k in range(n - 1):
            f0, x0, y0 = positions[k]
            f1, x1, y1 = positions[k + 1]
            if f1 - f0 != 1:
                continue
            dx, dy = x1 - x0, y1 - y0
            if math.hypot(dx, dy) < self.min_speed_px:
                continue
            vel[k] = (dx, dy)

        # A "corner" at position index c (1 <= c <= n-2) is a sharp turn if
        # the incoming vector vel[c-1] and outgoing vector vel[c] diverge.
        sharp_turn_positions = set()
        corner_angles: Dict[int, float] = {}
        for c in range(1, n - 1):
            v_in, v_out = vel[c - 1], vel[c]
            if v_in is None or v_out is None:
                continue
            angle = self._angle_between(v_in, v_out)
            if angle > self.angle_threshold_deg:
                sharp_turn_positions.add(c)
                corner_angles[c] = angle

        if not sharp_turn_positions:
            return

        # Group consecutive corner positions into runs.
        runs: List[List[int]] = []
        current_run: List[int] = []
        for c in sorted(sharp_turn_positions):
            if current_run and c == current_run[-1] + 1:
                current_run.append(c)
            else:
                if current_run:
                    runs.append(current_run)
                current_run = [c]
        if current_run:
            runs.append(current_run)

        jitter_frames: List[int] = []
        jitter_run_count = 0
        for run in runs:
            if len(run) >= self.min_run_length:
                frame_indices = [positions[c][0] for c in run]
                jitter_frames.extend(frame_indices)
                jitter_run_count += 1
                self.decision_details.append(
                    {
                        "object_id": obj_id,
                        "frame_indices": frame_indices,
                        "run_length": len(run),
                        "max_turn_angle_deg": max(corner_angles[c] for c in run),
                    }
                )

        if not jitter_frames:
            return

        self.objects_with_jitter += 1
        self.jitter_runs_found += jitter_run_count
        for frame_idx in jitter_frames:
            frame_str = str(frame_idx)
            frame_obj = frames[frame_str]["objects"].get(obj_id)
            if frame_obj is None:
                continue
            if self.mark_only:
                update_housekeeping_annotator(frame_obj, "jitter")
                self.frames_marked += 1
            else:
                del frames[frame_str]["objects"][obj_id]
                self.frames_removed += 1

        logger.info(
            f"PositionalJitterPass: {'Marked' if self.mark_only else 'Removed'} "
            f"{len(jitter_frames)} jitter frame(s) for object {obj_id} "
            f"across {jitter_run_count} run(s)"
        )

    @staticmethod
    def _angle_between(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
        """Return the angle in degrees between two 2D vectors, in [0, 180]."""
        mag1 = math.hypot(*v1)
        mag2 = math.hypot(*v2)
        if mag1 == 0 or mag2 == 0:
            return 0.0
        cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (mag1 * mag2)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        return math.degrees(math.acos(cos_angle))

    def get_statistics(self) -> Dict[str, Any]:
        """Get positional jitter pass statistics.

        Returns:
            Dictionary with detection/removal statistics
        """
        return {
            "objects_checked": self.objects_checked,
            "objects_with_jitter": self.objects_with_jitter,
            "jitter_runs_found": self.jitter_runs_found,
            "frames_removed": self.frames_removed,
            "frames_marked": self.frames_marked,
        }

    def get_decision_log(self) -> List[Dict[str, Any]]:
        """Return one record per removed or marked jitter run."""
        action = "mark_frames" if self.mark_only else "remove_frames"
        return [
            {
                "action": action,
                "object_id": detail["object_id"],
                "frame_indices": detail["frame_indices"],
                "reason": f"{detail['run_length']} consecutive sharp turns "
                f"(max {detail['max_turn_angle_deg']:.1f} deg > "
                f"{self.angle_threshold_deg} deg threshold)",
                "details": {
                    "run_length": detail["run_length"],
                    "max_turn_angle_deg": detail["max_turn_angle_deg"],
                    "angle_threshold_deg": self.angle_threshold_deg,
                },
            }
            for detail in self.decision_details
        ]
