from typing import Union, List, Dict, Tuple
import math
import numpy as np
from scipy.interpolate import splev, splprep


class InterpolationService:
    @staticmethod
    def interpolate_center_trajectory(
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        num_frames: int,
    ) -> List[Tuple[float, float]]:
        if num_frames <= 0:
            raise ValueError("Number of frames must be positive")

        interpolation_factors = np.linspace(0, 1, num_frames + 2)[1:-1]
        x_positions = (
            start_point[0] + (end_point[0] - start_point[0]) * interpolation_factors
        )
        y_positions = (
            start_point[1] + (end_point[1] - start_point[1]) * interpolation_factors
        )

        return list(zip(x_positions, y_positions))

    @staticmethod
    def interpolate_annotations(
        start_bbox: Union[Dict, object],
        end_bbox: Union[Dict, object],
        num_frames: int,
    ) -> List[Dict]:
        if num_frames < 0:
            raise ValueError("Number of frames must be non-negative")
        if num_frames == 0:
            return []

        start_dict = start_bbox if isinstance(start_bbox, dict) else start_bbox.__dict__
        end_dict = end_bbox if isinstance(end_bbox, dict) else end_bbox.__dict__
        start_center = (start_dict.get("x_center", 0), start_dict.get("y_center", 0))
        end_center = (end_dict.get("x_center", 0), end_dict.get("y_center", 0))
        centers_interpolated = InterpolationService.interpolate_center_trajectory(
            start_center, end_center, num_frames
        )
        properties = ["width", "height", "rotation"]
        interpolated_properties = {prop: [] for prop in properties}
        for prop in properties:
            start_value = start_dict.get(prop, 0)
            end_value = end_dict.get(prop, 0)
            if prop == "rotation":
                rotation_difference = (
                    (end_value - start_value + math.pi) % (2 * math.pi) - math.pi
                )
                # Increment num_frames by 2 to account for start
                # and end points, then exclude them via the [1:-1].
                interpolation_factors = np.linspace(0, 1, num_frames + 2)[1:-1]
                interpolated_properties[prop] = [
                    (start_value + rotation_difference * factor) % (2 * math.pi)
                    for factor in interpolation_factors
                ]
            else:
                interpolation_factors = np.linspace(0, 1, num_frames + 2)[1:-1]
                interpolated_properties[prop] = [
                    start_value + (end_value - start_value) * factor
                    for factor in interpolation_factors
                ]
        interpolated_bboxes = []
        for i in range(num_frames):
            x_center, y_center = centers_interpolated[i]
            bbox = {
                "x_center": x_center,
                "y_center": y_center,
                "width": interpolated_properties["width"][i],
                "height": interpolated_properties["height"][i],
                "rotation": interpolated_properties["rotation"][i],
            }
            interpolated_bboxes.append(bbox)
        return interpolated_bboxes

    @staticmethod
    def _deduplicate_positions(
        xs: List[float], ys: List[float]
    ) -> Tuple[List[float], List[float], List[int]]:
        """Remove consecutive duplicate (x, y) positions.

        Returns deduplicated x/y lists and a mapping from each original
        index to the corresponding unique-point index (for forward-filling).
        """
        unique_xs: List[float] = []
        unique_ys: List[float] = []
        orig_to_unique: List[int] = []

        for x, y in zip(xs, ys):
            if not unique_xs or (x != unique_xs[-1] or y != unique_ys[-1]):
                unique_xs.append(x)
                unique_ys.append(y)
            orig_to_unique.append(len(unique_xs) - 1)

        return unique_xs, unique_ys, orig_to_unique

    @staticmethod
    def spline_interpolate_angles(
        positions: List[Tuple[float, float]],
        smoothing_factor: float = 0.0,
    ) -> List[float]:
        """Compute heading angles from a spline fit to (x, y) positions.

        Consecutive duplicate positions are deduplicated before fitting;
        the resulting angles are forward-filled back to the original length.

        Returns a list of angles in radians, one per input position.
        Raises ValueError if fewer than 4 unique positions are available.
        """
        if len(positions) < 1:
            raise ValueError("At least one position is required.")

        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        unique_xs, unique_ys, orig_to_unique = (
            InterpolationService._deduplicate_positions(xs, ys)
        )

        if len(unique_xs) < 4:
            raise ValueError(
                f"Need at least 4 unique positions for cubic spline, "
                f"got {len(unique_xs)}."
            )

        tck, u = splprep(
            [unique_xs, unique_ys], s=smoothing_factor, k=3
        )
        dx, dy = splev(u, tck, der=1)
        unique_angles = np.arctan2(dy, dx).tolist()

        # Forward-fill angles for deduplicated positions
        return [unique_angles[orig_to_unique[i]] for i in range(len(positions))]
