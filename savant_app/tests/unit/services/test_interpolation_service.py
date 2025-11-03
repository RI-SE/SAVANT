import pytest
import numpy as np
from savant_app.services.interpolation_service import InterpolationService


@pytest.fixture
def sample_bbox():
    return {"x_center": 100, "y_center": 200, "width": 50, "height": 30, "rotation": 0}


class TestInterpolateCenterTrajectory:
    @pytest.mark.parametrize(
        "start,end,num_frames,expected_length",
        [
            ((0, 0), (10, 10), 5, 5),
            ((5, 5), (5, 5), 3, 3),
            ((0, 0), (100, 50), 10, 10),
        ],
    )
    def test_basic_interpolation(self, start, end, num_frames, expected_length):
        result = InterpolationService.interpolate_center_trajectory(
            start, end, num_frames
        )
        assert len(result) == expected_length

        x_vals = [p[0] for p in result]
        y_vals = [p[1] for p in result]

        if num_frames > 1:
            x_diffs = np.diff(x_vals)
            y_diffs = np.diff(y_vals)
            expected_x_diff = (end[0] - start[0]) / (num_frames + 1)
            expected_y_diff = (end[1] - start[1]) / (num_frames + 1)
            assert np.allclose(x_diffs, expected_x_diff)
            assert np.allclose(y_diffs, expected_y_diff)

    def test_edge_cases(self):
        with pytest.raises(ValueError):
            InterpolationService.interpolate_center_trajectory((0, 0), (10, 10), 0)

        result = InterpolationService.interpolate_center_trajectory((0, 0), (10, 10), 1)
        assert len(result) == 1
        x, y = result[0]
        assert 0 < x < 10
        assert 0 < y < 10


class TestInterpolateAnnotations:
    def test_basic_interpolation(self, sample_bbox):
        start_bbox = sample_bbox
        end_bbox = {
            "x_center": 200,
            "y_center": 300,
            "width": 60,
            "height": 40,
            "rotation": 45,
        }
        num_frames = 5

        result = InterpolationService.interpolate_annotations(
            start_bbox, end_bbox, num_frames
        )

        assert len(result) == num_frames
        assert all(
            set(b.keys()) == {"x_center", "y_center", "width", "height", "rotation"}
            for b in result
        )

        widths = [b["width"] for b in result]
        if num_frames > 1:
            width_diffs = np.diff(widths)
            expected_diff = (end_bbox["width"] - start_bbox["width"]) / (num_frames + 1)
            assert np.allclose(width_diffs, expected_diff)

    def test_missing_properties(self):
        start_bbox = {"x_center": 0, "y_center": 0}  # missing width/height/rotation
        end_bbox = {"x_center": 10, "y_center": 10, "width": 20}
        num_frames = 3

        result = InterpolationService.interpolate_annotations(
            start_bbox, end_bbox, num_frames
        )

        expected_first_width = 0 + (20 - 0) * (1 / (num_frames + 1))
        expected_last_width = 0 + (20 - 0) * (num_frames / (num_frames + 1))

        assert result[0]["width"] == pytest.approx(expected_first_width)
        assert result[-1]["width"] == pytest.approx(expected_last_width)

        # height and rotation default to 0
        assert all(b["height"] == 0 for b in result)
        assert all(b["rotation"] == 0 for b in result)

    def test_rotation_wrapping(self):
        start_bbox = {"x_center": 0, "y_center": 0, "rotation": 350}
        end_bbox = {"x_center": 0, "y_center": 0, "rotation": 10}
        num_frames = 5

        result = InterpolationService.interpolate_annotations(
            start_bbox, end_bbox, num_frames
        )

        rotations = [b["rotation"] for b in result]
        rotation_diff = ((10 - 350 + 180) % 360) - 180
        expected_rotations = [
            (350 + rotation_diff * factor) % 360
            for factor in np.linspace(0, 1, num_frames + 2)[1:-1]
        ]

        assert np.allclose(rotations, expected_rotations)

    def test_zero_frames(self, sample_bbox):
        result = InterpolationService.interpolate_annotations(
            sample_bbox, sample_bbox, 0
        )
        assert result == []

    def test_single_frame(self, sample_bbox):
        result = InterpolationService.interpolate_annotations(
            sample_bbox, sample_bbox, 1
        )
        assert len(result) == 1
        bbox = result[0]
        assert all(
            k in bbox for k in ["x_center", "y_center", "width", "height", "rotation"]
        )
