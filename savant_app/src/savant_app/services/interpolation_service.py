import numpy as np
from typing import Dict, List, Union

class CubicBezier:
    """Custom implementation of Cubic Bezier interpolation"""
    def __init__(self, P0, P1, P2, P3):
        self.P0 = P0
        self.P1 = P1
        self.P2 = P2
        self.P3 = P3
        
    def __call__(self, t):
        """Evaluate the Bezier curve at parameter t"""
        t = np.asarray(t)
        return (1-t)**3 * self.P0 + 3*(1-t)**2*t * self.P1 + 3*(1-t)*t**2 * self.P2 + t**3 * self.P3

class InterpolationService:
    @staticmethod
    def interpolate_center_trajectory(
        start_point: tuple[float, float],
        end_point: tuple[float, float],
        control_points: List[tuple[float, float]],
        num_frames: int
    ) -> List[tuple[float, float]]:
        """
        Interpolate center trajectory using cubic Bezier spline for 2D points.
        """
        if len(control_points) == 0:
            # Linear interpolation
            x_values = np.linspace(start_point[0], end_point[0], num_frames)
            y_values = np.linspace(start_point[1], end_point[1], num_frames)
            return list(zip(x_values, y_values))
        else:
            # Convert points to numpy array for vector operations
            points = np.array([start_point] + control_points + [end_point])
            
            # Calculate Bezier curve
            curve_x = []
            curve_y = []
            for t in np.linspace(0, 1, num_frames):
                # Cubic Bezier formula
                mt = 1 - t
                x = (mt**3 * points[0][0] + 
                     3 * mt**2 * t * points[1][0] + 
                     3 * mt * t**2 * points[2][0] + 
                     t**3 * points[3][0])
                y = (mt**3 * points[0][1] + 
                     3 * mt**2 * t * points[1][1] + 
                     3 * mt * t**2 * points[2][1] + 
                     t**3 * points[3][1])
                curve_x.append(x)
                curve_y.append(y)
            
            return list(zip(curve_x, curve_y))

    @staticmethod
    def interpolate_annotations(
        start_bbox: Union[Dict, object], 
        end_bbox: Union[Dict, object], 
        num_frames: int, 
        control_centers: List[tuple[float, float]]
    ) -> List[Dict]:
        """Interpolate bounding boxes with spline interpolation for center points"""
        # Ensure bbox inputs are dictionaries
        start_dict = start_bbox if isinstance(start_bbox, dict) else start_bbox.__dict__
        end_dict = end_bbox if isinstance(end_bbox, dict) else end_bbox.__dict__
        
        # Create frame numbers for interpolation points
        frames = list(range(1, num_frames + 1))
        
        # Safely access center coordinates
        start_center = (start_dict.get('x_center', 0), start_dict.get('y_center', 0))
        end_center = (end_dict.get('x_center', 0), end_dict.get('y_center', 0))
        print(start_center, end_center, control_centers)
        
        # Interpolate center trajectory using spline
        centers = InterpolationService.interpolate_center_trajectory(
            start_center, end_center, control_centers, num_frames
        )
        
        # Interpolate other properties linearly
        properties = ['width', 'height', 'rotation']
        interpolated_values = {prop: [] for prop in properties}
        
        for prop in properties:
            start_val = start_dict.get(prop, 0)
            end_val = end_dict.get(prop, 0)
            interpolated_values[prop] = np.linspace(
                start_val, end_val, num_frames
            ).tolist()
        
        # Combine into bbox dictionaries
        interpolated = []
        for i in range(num_frames):
            x_center, y_center = centers[i]
            bbox = {
                'x_center': x_center,
                'y_center': y_center,
                'width': interpolated_values['width'][i],
                'height': interpolated_values['height'][i],
                'rotation': interpolated_values['rotation'][i]
            }
            interpolated.append(bbox)
            
        return interpolated
    
    @staticmethod
    def interpolate_bboxes(start_bbox: Union[Dict, object], 
                          end_bbox: Union[Dict, object], 
                          num_frames: int, 
                          control_points: Dict[str, List[float]]) -> List[Dict]:
        """Interpolate bounding boxes between start and end frames"""
        interpolated = []
        
        # Ensure bbox inputs are dictionaries
        start_dict = start_bbox if isinstance(start_bbox, dict) else start_bbox.__dict__
        end_dict = end_bbox if isinstance(end_bbox, dict) else end_bbox.__dict__
        
        # Create frame numbers for interpolation points
        frames = list(range(1, num_frames + 1))
        
        # Interpolate each property separately
        properties = ['center_x', 'center_y', 'width', 'height', 'rotation']
        interpolated_values = {prop: [] for prop in properties}
        
        for prop in properties:
            start_val = start_dict.get(prop, 0)
            end_val = end_dict.get(prop, 0)
            ctrl_pts = control_points.get(prop, [])
            interpolated_values[prop] = InterpolationService.interpolate_property(
                start_val, end_val, ctrl_pts, frames
            )
        
        # Combine into bbox dictionaries
        for i in range(num_frames):
            bbox = {
                'x_center': interpolated_values['x_center'][i],
                'y_center': interpolated_values['y_center'][i],
                'width': interpolated_values['width'][i],
                'height': interpolated_values['height'][i],
                'rotation': interpolated_values['rotation'][i]
            }
            interpolated.append(bbox)
            
        return interpolated
