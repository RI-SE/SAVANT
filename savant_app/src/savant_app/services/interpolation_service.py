import numpy as np
from typing import Dict, List, Union


class InterpolationService:
    @staticmethod
    def interpolate_center_trajectory(
        start_point: tuple[float, float],
        end_point: tuple[float, float],
        # control_points: List[tuple[float, float]],
        num_frames: int,
    ) -> List[tuple[float, float]]:
        """
        Interpolate center trajectory using cubic Bezier spline for 2D points.

        Args:
            start_point: Starting (x,y) coordinate
            end_point: Ending (x,y) coordinate
            num_frames: Number of frames to interpolate (must be >= 1)

        Raises:
            ValueError: If num_frames is less than 1
        """
        if num_frames < 1:
            raise ValueError("num_frames must be >= 1")
        x_values = np.linspace(start_point[0], end_point[0], num_frames)
        y_values = np.linspace(start_point[1], end_point[1], num_frames)
        return list(zip(x_values, y_values))

    @staticmethod
    def interpolate_annotations(
        start_bbox: Union[Dict, object],
        end_bbox: Union[Dict, object],
        num_frames: int,
        # control_centers: List[tuple[float, float]]
    ) -> List[Dict]:
        """Interpolate bounding boxes with spline interpolation for center points"""
        # Ensure bbox inputs are dictionaries
        start_dict = start_bbox if isinstance(start_bbox, dict) else start_bbox.__dict__
        end_dict = end_bbox if isinstance(end_bbox, dict) else end_bbox.__dict__

        # Safely access center coordinates
        start_center = (start_dict.get("x_center", 0), start_dict.get("y_center", 0))
        end_center = (end_dict.get("x_center", 0), end_dict.get("y_center", 0))

        centers = InterpolationService.interpolate_center_trajectory(
            start_center, end_center, num_frames
        )

        # Interpolate other properties linearly
        properties = ["width", "height", "rotation"]
        interpolated_values = {prop: [] for prop in properties}

        for prop in properties:
            start_val = start_dict.get(prop, 0)
            end_val = end_dict.get(prop, 0)

            if prop == "rotation":
                # Handle angle wrapping for rotation
                diff = ((end_val - start_val + 180) % 360) - 180
                interpolated_values[prop] = [
                    (start_val + diff * t) % 360 for t in np.linspace(0, 1, num_frames)
                ]
            else:
                interpolated_values[prop] = np.linspace(
                    start_val, end_val, num_frames
                ).tolist()

        # Combine into bbox dictionaries
        interpolated = []
        for i in range(num_frames):
            x_center, y_center = centers[i]
            bbox = {
                "x_center": x_center,
                "y_center": y_center,
                "width": interpolated_values["width"][i],
                "height": interpolated_values["height"][i],
                "rotation": interpolated_values["rotation"][i],
            }
            interpolated.append(bbox)

        return interpolated
