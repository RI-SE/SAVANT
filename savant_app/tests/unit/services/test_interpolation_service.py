import pytest
import numpy as np
from savant_app.services.interpolation_service import InterpolationService


@pytest.fixture
def sample_bbox():
    """Fixture providing sample bounding box data"""
    return {
        "x_center": 100,
        "y_center": 200,
        "width": 50,
        "height": 30,
        "rotation": 0
    }


class TestInterpolateCenterTrajectory:
    """Tests for interpolate_center_trajectory method"""
    
    @pytest.mark.parametrize("start,end,frames,expected_length", [
        ((0, 0), (10, 10), 5, 5),  # Basic case
        ((5, 5), (5, 5), 3, 3),    # Same start/end
        ((0, 0), (100, 50), 10, 10) # Larger range
    ])
    def test_interpolate_center_trajectory_basic(self, start, end, frames, expected_length):
        """Test basic linear interpolation cases"""
        result = InterpolationService.interpolate_center_trajectory(start, end, frames)
        assert len(result) == expected_length
        assert result[0] == pytest.approx(start)
        assert result[-1] == pytest.approx(end)
        
        # Verify linear progression
        x_values = [p[0] for p in result]
        y_values = [p[1] for p in result]
        assert np.allclose(np.diff(x_values), (end[0]-start[0])/(frames-1))
        assert np.allclose(np.diff(y_values), (end[1]-start[1])/(frames-1))

    def test_interpolate_center_trajectory_edge_cases(self):
        """Test edge case handling"""
        # Zero frames
        with pytest.raises(ValueError):
            InterpolationService.interpolate_center_trajectory((0,0), (10,10), 0)
            
        # Single frame
        result = InterpolationService.interpolate_center_trajectory((0,0), (10,10), 1)
        assert len(result) == 1
        assert result[0] == pytest.approx((0,0))


class TestInterpolateAnnotations:
    """Tests for interpolate_annotations method"""
    
    def test_interpolate_annotations_basic(self, sample_bbox):
        """Test basic bbox interpolation"""
        start_bbox = sample_bbox
        end_bbox = {
            "x_center": 200,
            "y_center": 300,
            "width": 60,
            "height": 40,
            "rotation": 45
        }
        num_frames = 5
        
        result = InterpolationService.interpolate_annotations(start_bbox, end_bbox, num_frames)
        
        # Verify output structure
        assert len(result) == num_frames
        assert all(set(bbox.keys()) == {"x_center", "y_center", "width", "height", "rotation"} 
                  for bbox in result)
                  
        # Verify start/end values
        assert result[0]["x_center"] == pytest.approx(start_bbox["x_center"])
        assert result[-1]["x_center"] == pytest.approx(end_bbox["x_center"])
        
        # Verify linear interpolation of properties
        widths = [bbox["width"] for bbox in result]
        assert np.allclose(np.diff(widths), (end_bbox["width"]-start_bbox["width"])/(num_frames-1))

    def test_interpolate_annotations_missing_properties(self):
        """Test handling of missing properties"""
        start_bbox = {"x_center": 0, "y_center": 0}  # Missing width/height/rotation
        end_bbox = {"x_center": 10, "y_center": 10, "width": 20}
        
        result = InterpolationService.interpolate_annotations(start_bbox, end_bbox, 3)
        
        # Missing properties should default to 0
        assert result[0]["width"] == 0
        assert result[0]["height"] == 0
        assert result[0]["rotation"] == 0
        assert result[-1]["width"] == pytest.approx(20)

    def test_interpolate_annotations_rotation_wrapping(self):
        """Test rotation interpolation handles angle wrapping"""
        start_bbox = {"x_center": 0, "y_center": 0, "rotation": 350}
        end_bbox = {"x_center": 0, "y_center": 0, "rotation": 10}
        
        result = InterpolationService.interpolate_annotations(start_bbox, end_bbox, 5)
        
        # Rotation should interpolate through 0 rather than backwards
        rotations = [bbox["rotation"] for bbox in result]
        assert rotations == pytest.approx([350, 355, 0, 5, 10])
