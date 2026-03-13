"""
BboxSmoothingPass - Bounding box smoothing postprocessing pass.
"""

import logging
from typing import Any, Dict, List
from collections import defaultdict

from ..base import PostprocessingPass
from ._common import update_housekeeping_annotator

logger = logging.getLogger(__name__)


class BboxSmoothingPass(PostprocessingPass):
    """Apply temporal smoothing to bbox parameters using bidirectional EMA.

    Smooths size (w, h) always. Position (x, y) smoothing is velocity-adaptive:
    low-velocity objects get more smoothing (OF centroid noise is worst when
    stationary), high-velocity objects get less (preserve responsive tracking).
    Bidirectional EMA (forward + backward + average) eliminates lag.
    """

    def __init__(
        self,
        smoothing_factor: float = 0.7,
        edge_margin: int = 100,
        edge_size_mode: str = "freeze",
        smooth_position: bool = True,
        min_velocity: float = 2.0,
        max_velocity: float = 20.0,
    ):
        """Initialize bbox smoothing pass.

        Args:
            smoothing_factor: Base EMA retention factor (0-1). Represents how much of the
                previous smoothed value to keep. Higher = more smoothing/stability.
                With factor 0.7: new_smoothed = 0.7 * old_smoothed + 0.3 * raw_value
                Default 0.7 provides good noise rejection while tracking real movement.
            edge_margin: Pixels from frame edge for special handling (default: 100)
            edge_size_mode: How to handle size near edges - "freeze" or "normal" (default: "freeze")
            smooth_position: Whether to smooth position (x, y) in addition to size.
                Uses velocity-adaptive factor to avoid over-smoothing fast objects.
            min_velocity: Below this velocity (px/frame), use maximum position smoothing.
            max_velocity: Above this velocity (px/frame), use minimum position smoothing.
        """
        self.smoothing_factor = smoothing_factor
        self.edge_margin = edge_margin
        self.edge_size_mode = edge_size_mode
        self.smooth_position = smooth_position
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity

        # Statistics
        self.objects_smoothed = 0
        self.frames_smoothed = 0
        self.edge_frames_handled = 0

    def process(self, openlabel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply temporal smoothing to bbox parameters.

        Args:
            openlabel_data: Complete OpenLabel data structure

        Returns:
            Modified OpenLabel data with smoothed bbox parameters
        """
        frames = openlabel_data.get("openlabel", {}).get("frames", {})

        if not hasattr(self, "frame_width") or not hasattr(self, "frame_height"):
            logger.warning("BboxSmoothingPass: Video properties not set, skipping")
            return openlabel_data

        # Build object-to-frames mapping
        object_frame_map = defaultdict(list)
        for frame_idx_str, frame_data in frames.items():
            frame_idx = int(frame_idx_str)
            frame_objects = frame_data.get("objects", {})
            for obj_id_str in frame_objects.keys():
                object_frame_map[obj_id_str].append(frame_idx)

        # Process each object
        for obj_id, frame_list in object_frame_map.items():
            if len(frame_list) < 2:
                continue

            frame_list_sorted = sorted(frame_list)
            self._smooth_object_trajectory(frames, obj_id, frame_list_sorted)
            self.objects_smoothed += 1

        pos_status = "enabled" if self.smooth_position else "disabled"
        logger.info(
            f"BboxSmoothing: Smoothed {self.objects_smoothed} objects, "
            f"{self.frames_smoothed} frames, {self.edge_frames_handled} edge frames handled "
            f"(position smoothing: {pos_status})"
        )

        return openlabel_data

    def _velocity_adaptive_factor(self, velocity: float) -> float:
        """Calculate position smoothing factor based on velocity.

        Low velocity → high smoothing (stationary/slow objects benefit most from denoising).
        High velocity → low smoothing (preserve responsive tracking for fast objects).

        Args:
            velocity: Object velocity in pixels/frame.

        Returns:
            Smoothing factor in [factor*0.5, factor*1.0] range.
        """
        if velocity <= self.min_velocity:
            return self.smoothing_factor  # Full smoothing
        if velocity >= self.max_velocity:
            return self.smoothing_factor * 0.5  # Reduced smoothing
        # Linear interpolation between full and reduced
        t = (velocity - self.min_velocity) / (self.max_velocity - self.min_velocity)
        return self.smoothing_factor * (1.0 - 0.5 * t)

    def _smooth_object_trajectory(
        self,
        frames: Dict[str, Any],
        obj_id: str,
        frame_list: List[int],
    ) -> None:
        """Smooth bbox parameters of a single object using bidirectional EMA.

        Size (w, h) is always smoothed with a fixed factor. Position (x, y) is
        optionally smoothed with a velocity-adaptive factor.

        Bidirectional smoothing eliminates lag by:
        1. Forward pass: EMA from start to end
        2. Backward pass: EMA from end to start
        3. Average the two passes

        Args:
            frames: Frame data dictionary
            obj_id: Object ID string
            frame_list: Sorted list of frame indices for this object
        """
        # Collect raw values
        raw_values = []
        for frame_idx in frame_list:
            frame_str = str(frame_idx)
            rbbox = frames[frame_str]["objects"][obj_id]["object_data"]["rbbox"][0]["val"]
            x, y, w, h, r = rbbox
            is_near_edge = self._is_near_edge(x, y)
            raw_values.append({
                "frame_idx": frame_idx,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "r": r,
                "is_near_edge": is_near_edge,
            })

        n = len(raw_values)
        if n == 0:
            return

        factor = self.smoothing_factor

        # --- Size smoothing (bidirectional EMA, fixed factor) ---
        forward_w = [0.0] * n
        forward_h = [0.0] * n
        forward_w[0] = raw_values[0]["w"]
        forward_h[0] = raw_values[0]["h"]

        for i in range(1, n):
            w, h = raw_values[i]["w"], raw_values[i]["h"]
            forward_w[i] = factor * forward_w[i-1] + (1 - factor) * w
            forward_h[i] = factor * forward_h[i-1] + (1 - factor) * h

        backward_w = [0.0] * n
        backward_h = [0.0] * n
        backward_w[n-1] = raw_values[n-1]["w"]
        backward_h[n-1] = raw_values[n-1]["h"]

        for i in range(n-2, -1, -1):
            w, h = raw_values[i]["w"], raw_values[i]["h"]
            backward_w[i] = factor * backward_w[i+1] + (1 - factor) * w
            backward_h[i] = factor * backward_h[i+1] + (1 - factor) * h

        # --- Position smoothing (bidirectional EMA, velocity-adaptive factor) ---
        forward_x = [0.0] * n
        forward_y = [0.0] * n
        backward_x = [0.0] * n
        backward_y = [0.0] * n

        if self.smooth_position:
            # Calculate per-frame velocities for adaptive factor
            velocities = [0.0] * n
            for i in range(1, n):
                dx = raw_values[i]["x"] - raw_values[i-1]["x"]
                dy = raw_values[i]["y"] - raw_values[i-1]["y"]
                velocities[i] = (dx**2 + dy**2) ** 0.5

            # Forward pass
            forward_x[0] = raw_values[0]["x"]
            forward_y[0] = raw_values[0]["y"]
            for i in range(1, n):
                pf = self._velocity_adaptive_factor(velocities[i])
                forward_x[i] = pf * forward_x[i-1] + (1 - pf) * raw_values[i]["x"]
                forward_y[i] = pf * forward_y[i-1] + (1 - pf) * raw_values[i]["y"]

            # Backward pass
            backward_x[n-1] = raw_values[n-1]["x"]
            backward_y[n-1] = raw_values[n-1]["y"]
            for i in range(n-2, -1, -1):
                pf = self._velocity_adaptive_factor(velocities[i+1])
                backward_x[i] = pf * backward_x[i+1] + (1 - pf) * raw_values[i]["x"]
                backward_y[i] = pf * backward_y[i+1] + (1 - pf) * raw_values[i]["y"]

        # Interior sizes for edge handling
        interior_sizes = []
        for i in range(n):
            avg_w = (forward_w[i] + backward_w[i]) / 2
            avg_h = (forward_h[i] + backward_h[i]) / 2
            if not raw_values[i]["is_near_edge"]:
                interior_sizes.append((i, avg_w, avg_h))

        # Apply smoothed values
        for i in range(n):
            frame_idx = raw_values[i]["frame_idx"]
            frame_str = str(frame_idx)
            rbbox = frames[frame_str]["objects"][obj_id]["object_data"]["rbbox"][0]["val"]
            is_near_edge = raw_values[i]["is_near_edge"]

            original_x = raw_values[i]["x"]
            original_y = raw_values[i]["y"]
            original_w = raw_values[i]["w"]
            original_h = raw_values[i]["h"]

            # Average bidirectional smoothing for size
            smoothed_w = (forward_w[i] + backward_w[i]) / 2
            smoothed_h = (forward_h[i] + backward_h[i]) / 2

            # Handle size near edges - use nearest interior size
            if is_near_edge and self.edge_size_mode == "freeze" and interior_sizes:
                nearest = min(interior_sizes, key=lambda x: abs(x[0] - i))
                smoothed_w, smoothed_h = nearest[1], nearest[2]
                self.edge_frames_handled += 1

            # Position smoothing (average bidirectional)
            smoothed_x = original_x
            smoothed_y = original_y
            if self.smooth_position:
                smoothed_x = (forward_x[i] + backward_x[i]) / 2
                smoothed_y = (forward_y[i] + backward_y[i]) / 2

            # Only update and tag if values actually changed (> 0.1 pixel threshold)
            changed = (
                abs(smoothed_w - original_w) > 0.1
                or abs(smoothed_h - original_h) > 0.1
                or abs(smoothed_x - original_x) > 0.1
                or abs(smoothed_y - original_y) > 0.1
            )
            if changed:
                rbbox[0] = smoothed_x
                rbbox[1] = smoothed_y
                rbbox[2] = smoothed_w
                rbbox[3] = smoothed_h
                update_housekeeping_annotator(frames[frame_str]["objects"][obj_id], "smooth")
                self.frames_smoothed += 1

    def _is_near_edge(self, x: float, y: float) -> bool:
        """Check if position is near frame edge.

        Args:
            x: Center x coordinate
            y: Center y coordinate

        Returns:
            True if position is within edge_margin of any frame edge
        """
        return (
            x < self.edge_margin
            or x > self.frame_width - self.edge_margin
            or y < self.edge_margin
            or y > self.frame_height - self.edge_margin
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get bbox smoothing statistics.

        Returns:
            Dictionary with smoothing statistics
        """
        return {
            "objects_smoothed": self.objects_smoothed,
            "frames_smoothed": self.frames_smoothed,
            "edge_frames_handled": self.edge_frames_handled,
        }
