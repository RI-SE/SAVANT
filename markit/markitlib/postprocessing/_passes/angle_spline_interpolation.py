"""
AngleSplineInterpolationPass - Derive bounding box angles from spline-fitted trajectories.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.interpolate import splev, splprep

from ..base import PostprocessingPass
from ._common import update_housekeeping_annotator

logger = logging.getLogger(__name__)


class AngleSplineInterpolationPass(PostprocessingPass):
    """Set bounding box angles parallel to a cubic spline fitted to the trajectory.

    For each tracked object the pass fits a parametric cubic spline through
    the (x, y) centre coordinates and then orients bounding boxes so that
    their rotation equals the tangent angle of the spline.

    Consecutive duplicate positions (vehicle stopped) are removed before
    fitting; their angles are forward-filled from the last moving position.
    """

    def __init__(self, smoothing_factor: float = 0.0):
        """Initialize the pass.

        Args:
            smoothing_factor: The ``s`` parameter passed to
                ``scipy.interpolate.splprep``.  ``0`` interpolates exactly;
                larger values produce smoother curves.
        """
        self.smoothing_factor = smoothing_factor

        # Statistics
        self.objects_processed = 0
        self.objects_skipped = 0
        self.angles_updated = 0

    # ------------------------------------------------------------------
    # Public interface (PostprocessingPass)
    # ------------------------------------------------------------------

    def process(self, openlabel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fit splines and update bounding box angles."""
        frames = openlabel_data.get("openlabel", {}).get("frames", {})

        object_frame_map: Dict[str, List[int]] = defaultdict(list)
        for frame_idx_str, frame_data in frames.items():
            frame_idx = int(frame_idx_str)
            for obj_id in frame_data.get("objects", {}):
                object_frame_map[obj_id].append(frame_idx)

        for obj_id, frame_list in object_frame_map.items():
            frame_list_sorted = sorted(frame_list)
            self._process_object(frames, obj_id, frame_list_sorted)

        logger.info(
            f"AngleSplineInterpolation: processed {self.objects_processed} objects, "
            f"skipped {self.objects_skipped}, updated {self.angles_updated} angles"
        )
        return openlabel_data

    def get_statistics(self) -> Dict[str, Any]:
        """Return processing statistics."""
        return {
            "objects_processed": self.objects_processed,
            "objects_skipped": self.objects_skipped,
            "angles_updated": self.angles_updated,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate_positions(
        xs: List[float],
        ys: List[float],
    ) -> Tuple[List[float], List[float], List[int]]:
        """Remove consecutive duplicate (x, y) positions.

        Returns the deduplicated x/y lists and a mapping from each original
        index to the index of the unique point it corresponds to (for
        forward-filling angles later).
        """
        unique_xs: List[float] = []
        unique_ys: List[float] = []
        orig_to_unique: List[int] = []

        for i, (x, y) in enumerate(zip(xs, ys)):
            if not unique_xs or (x != unique_xs[-1] or y != unique_ys[-1]):
                unique_xs.append(x)
                unique_ys.append(y)
            orig_to_unique.append(len(unique_xs) - 1)

        return unique_xs, unique_ys, orig_to_unique

    def _process_object(
        self,
        frames: Dict[str, Any],
        obj_id: str,
        frame_list: List[int],
    ) -> None:
        """Fit a spline and update angles for a single object."""
        # Collect centre coordinates in frame order
        xs: List[float] = []
        ys: List[float] = []
        for frame_idx in frame_list:
            rbbox = frames[str(frame_idx)]["objects"][obj_id][
                "object_data"
            ]["rbbox"][0]["val"]
            xs.append(rbbox[0])
            ys.append(rbbox[1])

        unique_xs, unique_ys, orig_to_unique = self._deduplicate_positions(xs, ys)

        # Cubic spline needs at least k+1 = 4 unique points
        if len(unique_xs) < 4:
            self.objects_skipped += 1
            return

        self.objects_processed += 1

        # Fit parametric cubic spline
        tck, u = splprep(
            [unique_xs, unique_ys], s=self.smoothing_factor, k=3
        )

        # Evaluate first derivative → tangent vector
        dx, dy = splev(u, tck, der=1)
        unique_angles = np.arctan2(dy, dx).tolist()

        # Map angles back to every original frame (forward-fill for duplicates)
        for i, frame_idx in enumerate(frame_list):
            angle = unique_angles[orig_to_unique[i]]
            frame_str = str(frame_idx)
            obj_data = frames[frame_str]["objects"][obj_id]
            rbbox = obj_data["object_data"]["rbbox"][0]["val"]

            rbbox[4] = angle

            # Normalise so width >= height (heading along long axis)
            if rbbox[3] > rbbox[2]:
                rbbox[2], rbbox[3] = rbbox[3], rbbox[2]

            update_housekeeping_annotator(obj_data, "spline")
            self.angles_updated += 1
