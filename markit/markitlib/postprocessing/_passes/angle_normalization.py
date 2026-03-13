"""
AngleNormalizationPass - Angle normalization postprocessing pass.
"""

import logging
from typing import Any, Dict

from ..base import PostprocessingPass
from ...utils import (
    normalize_angle_to_2pi_range,
)

logger = logging.getLogger(__name__)


class AngleNormalizationPass(PostprocessingPass):
    """Normalize all rotation angles to [0, 2π) for OpenLabel output.

    This is a MANDATORY final pass that ensures all rotation values in the
    OpenLabel output conform to the [0, 2π) range, regardless of what
    internal continuous angle representation was used during postprocessing.

    This pass should always be the LAST pass in the pipeline.
    """

    def __init__(self):
        """Initialize angle normalization pass."""
        self.angles_normalized = 0

    def process(self, openlabel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize all rotation angles to [0, 2π) range.

        Args:
            openlabel_data: OpenLabel structure with frame data

        Returns:
            Modified OpenLabel structure with normalized angles
        """
        frames = openlabel_data.get("openlabel", {}).get("frames", {})

        for frame_idx_str, frame_data in frames.items():
            frame_objects = frame_data.get("objects", {})

            for obj_id_str, obj_data in frame_objects.items():
                # Get rbbox data
                rbbox = obj_data["object_data"]["rbbox"][0]["val"]
                rotation = rbbox[4]

                # Normalize to [0, 2π) for OpenLabel output
                normalized_rotation = normalize_angle_to_2pi_range(rotation)

                # Update if changed
                if rotation != normalized_rotation:
                    rbbox[4] = normalized_rotation
                    self.angles_normalized += 1

        logger.info(
            f"AngleNormalization: Normalized {self.angles_normalized} angles to [0, 2π) range"
        )

        return openlabel_data

    def get_statistics(self) -> Dict[str, Any]:
        """Get angle normalization statistics.

        Returns:
            Dictionary with normalization statistics
        """
        return {"angles_normalized": self.angles_normalized}
