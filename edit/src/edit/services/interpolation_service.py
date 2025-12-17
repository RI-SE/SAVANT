from typing import Union, List, Dict, Tuple
import numpy as np


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
                rotation_difference = ((end_value - start_value + 180) % 360) - 180
                # Increment num_frames by 2 to account for start
                # and end points, then exclude them via the [1:-1].
                interpolation_factors = np.linspace(0, 1, num_frames + 2)[1:-1]
                interpolated_properties[prop] = [
                    (start_value + rotation_difference * factor) % 360
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
