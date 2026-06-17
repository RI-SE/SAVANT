import pytest
import numpy as np
from edit.services.interpolation_service import InterpolationService


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
        """Shortest-path interpolation in radians: values close in angle must not rotate far."""
        import math
        # Simulate the real bug: 7.596 rad ≈ 75°, 1.397 rad ≈ 80° — only ~5° apart.
        # Before the fix the service produced a ~355° backward rotation.
        start_rot = 7.59592698725086
        end_rot = 1.3966128312587314
        start_bbox = {"x_center": 0, "y_center": 0, "rotation": start_rot}
        end_bbox = {"x_center": 0, "y_center": 0, "rotation": end_rot}
        num_frames = 5

        result = InterpolationService.interpolate_annotations(
            start_bbox, end_bbox, num_frames
        )

        rotations = [b["rotation"] for b in result]
        # Each interpolated rotation should stay close to the start/end (~75-80°).
        for r in rotations:
            r_deg = math.degrees(r)
            # Must be near 75°-80°, not spinning through 355° the wrong way.
            assert 70 <= r_deg <= 85, f"Unexpected rotation {r_deg:.1f}° (expected ~75-80°)"

        # Also verify the cross-zero case: 350° → 10° in radians (≈ +20° forward, not -340°).
        start_rad = math.radians(350)
        end_rad = math.radians(10)
        start_bbox2 = {"x_center": 0, "y_center": 0, "rotation": start_rad}
        end_bbox2 = {"x_center": 0, "y_center": 0, "rotation": end_rad}
        result2 = InterpolationService.interpolate_annotations(start_bbox2, end_bbox2, 5)
        for b in result2:
            r_deg = math.degrees(b["rotation"])
            # Should pass through ~350-360/0-10°, not backwards through ~170-350°.
            assert r_deg > 345 or r_deg < 15, f"Unexpected rotation {r_deg:.1f}°"

    def test_rotation_no_wrap(self):
        """Simple case: both angles in mid-range, no wrapping needed."""
        import math
        start_angle = math.pi / 4
        end_angle = math.pi / 2
        start_bbox = {"x_center": 0, "y_center": 0, "rotation": start_angle}
        end_bbox = {"x_center": 0, "y_center": 0, "rotation": end_angle}
        num_frames = 3

        result = InterpolationService.interpolate_annotations(
            start_bbox, end_bbox, num_frames
        )

        rotations = [b["rotation"] for b in result]
        diff = end_angle - start_angle
        expected = [
            start_angle + diff * f
            for f in np.linspace(0, 1, num_frames + 2)[1:-1]
        ]
        assert np.allclose(rotations, expected, atol=1e-9)

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
