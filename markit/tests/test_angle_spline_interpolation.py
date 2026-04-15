"""
Tests for AngleSplineInterpolationPass.
"""

import math

import numpy as np
import pytest

from markit.markitlib.postprocessing import AngleSplineInterpolationPass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_openlabel(trajectories):
    """Build a minimal OpenLabel structure from trajectory descriptions.

    Args:
        trajectories: dict mapping obj_id → list of (frame_idx, x, y, w, h, r)
            tuples sorted by frame_idx.

    Returns:
        OpenLabel-style dict ready for pass.process().
    """
    frames = {}
    objects = {}
    for obj_id, points in trajectories.items():
        objects[obj_id] = {"name": obj_id, "type": "car"}
        for frame_idx, x, y, w, h, r in points:
            frame_str = str(frame_idx)
            if frame_str not in frames:
                frames[frame_str] = {"objects": {}}
            frames[frame_str]["objects"][obj_id] = {
                "object_data": {
                    "rbbox": [{"name": "shape", "val": [x, y, w, h, r]}],
                    "vec": [
                        {"name": "annotator", "val": ["yolo_obb_v8"]},
                        {"name": "confidence", "val": [0.9]},
                    ],
                }
            }
    return {"openlabel": {"frames": frames, "objects": objects}}


def _get_angles(openlabel_data, obj_id):
    """Extract the angle sequence for an object, sorted by frame index."""
    frames = openlabel_data["openlabel"]["frames"]
    result = []
    for frame_str in sorted(frames, key=lambda s: int(s)):
        objs = frames[frame_str].get("objects", {})
        if obj_id in objs:
            rbbox = objs[obj_id]["object_data"]["rbbox"][0]["val"]
            result.append((int(frame_str), rbbox[4]))
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAngleSplineInterpolationPass:
    """Tests for the spline-based angle interpolation pass."""

    def test_straight_line_horizontal(self):
        """Object moving right along x-axis → all angles ≈ 0."""
        traj = {
            "obj_1": [
                (i, 100.0 + i * 10.0, 200.0, 80, 40, 1.5)
                for i in range(10)
            ]
        }
        data = _make_openlabel(traj)

        p = AngleSplineInterpolationPass(smoothing_factor=0.0)
        result = p.process(data)

        for frame_idx, angle in _get_angles(result, "obj_1"):
            assert abs(angle) < 0.05, (
                f"Frame {frame_idx}: expected ≈0, got {angle:.4f}"
            )

        assert p.objects_processed == 1
        assert p.objects_skipped == 0
        assert p.angles_updated == 10

    def test_straight_line_diagonal(self):
        """Object moving at 45° → all angles ≈ π/4."""
        traj = {
            "obj_1": [
                (i, 100.0 + i * 10.0, 100.0 + i * 10.0, 80, 40, 0.0)
                for i in range(10)
            ]
        }
        data = _make_openlabel(traj)

        p = AngleSplineInterpolationPass(smoothing_factor=0.0)
        result = p.process(data)

        for frame_idx, angle in _get_angles(result, "obj_1"):
            assert abs(angle - math.pi / 4) < 0.05, (
                f"Frame {frame_idx}: expected ≈π/4, got {angle:.4f}"
            )

    def test_curved_trajectory(self):
        """Quarter-circle arc: angle should rotate from 0 to ~π/2."""
        n = 20
        t = np.linspace(0, math.pi / 2, n)
        radius = 200.0
        xs = (radius * np.cos(t) + 300.0).tolist()
        ys = (radius * np.sin(t) + 300.0).tolist()

        traj = {
            "obj_1": [
                (i, xs[i], ys[i], 80, 40, 0.0)
                for i in range(n)
            ]
        }
        data = _make_openlabel(traj)

        p = AngleSplineInterpolationPass(smoothing_factor=0.0)
        result = p.process(data)

        angles = _get_angles(result, "obj_1")
        # First angle should be near π/2 (tangent to cos at t=0 is -sin → pointing up)
        # Actually for parametric circle (cos t, sin t): tangent = (-sin t, cos t)
        # At t=0: tangent is (0, 1) → angle = π/2
        # At t=π/2: tangent is (-1, 0) → angle = π (or -π, equivalent)
        first_angle = angles[0][1]
        last_angle = angles[-1][1]

        assert abs(first_angle - math.pi / 2) < 0.15, (
            f"First angle should be ≈π/2, got {first_angle:.4f}"
        )
        # atan2 may return -π or π; both represent the same direction
        assert abs(abs(last_angle) - math.pi) < 0.15, (
            f"Last angle should be ≈±π, got {last_angle:.4f}"
        )

        # Angles should be monotonically increasing (≈π/2 → ≈π)
        # Use angular difference to handle the ±π wrap-around
        for i in range(1, len(angles)):
            diff = angles[i][1] - angles[i - 1][1]
            # Normalize to [-π, π]
            diff = (diff + math.pi) % (2 * math.pi) - math.pi
            assert diff >= -0.1, (
                f"Angles should increase: frame {angles[i][0]} "
                f"({angles[i][1]:.4f}) vs frame {angles[i-1][0]} "
                f"({angles[i-1][1]:.4f}), diff={diff:.4f}"
            )

    def test_stationary_frames_forward_fill(self):
        """Repeated positions should get forward-filled angles."""
        # Moving right, then stopped for 4 frames, then moving right again
        traj_points = []
        # Moving phase 1: frames 0-4
        for i in range(5):
            traj_points.append((i, 100.0 + i * 20.0, 200.0, 80, 40, 0.0))
        # Stationary phase: frames 5-8 (same position as frame 4)
        for i in range(5, 9):
            traj_points.append((i, 180.0, 200.0, 80, 40, 0.0))
        # Moving phase 2: frames 9-14
        for i in range(9, 15):
            traj_points.append(
                (i, 180.0 + (i - 8) * 20.0, 200.0, 80, 40, 0.0)
            )

        traj = {"obj_1": traj_points}
        data = _make_openlabel(traj)

        p = AngleSplineInterpolationPass(smoothing_factor=0.0)
        result = p.process(data)

        angles = _get_angles(result, "obj_1")
        angle_dict = dict(angles)

        # All angles should be approximately 0 (moving right along x-axis)
        for frame_idx, angle in angles:
            assert abs(angle) < 0.15, (
                f"Frame {frame_idx}: expected ≈0 (rightward), got {angle:.4f}"
            )

        # Stationary frames should have the same angle as the last moving frame
        # before the stop (forward-fill)
        last_moving_angle = angle_dict[4]
        for i in range(5, 9):
            assert angle_dict[i] == pytest.approx(last_moving_angle, abs=1e-10), (
                f"Stationary frame {i}: expected forward-filled angle "
                f"{last_moving_angle:.6f}, got {angle_dict[i]:.6f}"
            )

    def test_too_few_points_skipped(self):
        """Object with <4 frames should be skipped."""
        traj = {
            "obj_1": [
                (0, 100.0, 200.0, 80, 40, 1.0),
                (1, 110.0, 200.0, 80, 40, 1.0),
                (2, 120.0, 200.0, 80, 40, 1.0),
            ]
        }
        data = _make_openlabel(traj)

        p = AngleSplineInterpolationPass(smoothing_factor=0.0)
        result = p.process(data)

        # Angles should remain unchanged
        for frame_idx, angle in _get_angles(result, "obj_1"):
            assert angle == 1.0, f"Frame {frame_idx}: angle should be unchanged"

        assert p.objects_skipped == 1
        assert p.objects_processed == 0
        assert p.angles_updated == 0

    def test_single_frame_skipped(self):
        """Single-frame object should be skipped."""
        traj = {"obj_1": [(0, 100.0, 200.0, 80, 40, 0.5)]}
        data = _make_openlabel(traj)

        p = AngleSplineInterpolationPass(smoothing_factor=0.0)
        result = p.process(data)

        angles = _get_angles(result, "obj_1")
        assert angles[0][1] == 0.5
        assert p.objects_skipped == 1

    def test_all_positions_identical_skipped(self):
        """Object where all positions are the same should be skipped."""
        traj = {
            "obj_1": [
                (i, 100.0, 200.0, 80, 40, 0.0)
                for i in range(10)
            ]
        }
        data = _make_openlabel(traj)

        p = AngleSplineInterpolationPass(smoothing_factor=0.0)
        p.process(data)

        # Only 1 unique point after dedup → skipped
        assert p.objects_skipped == 1
        assert p.objects_processed == 0

    def test_width_height_normalization(self):
        """After angle update, width should be >= height."""
        # h > w initially — should be swapped
        traj = {
            "obj_1": [
                (i, 100.0 + i * 10.0, 200.0, 30, 80, 0.0)
                for i in range(10)
            ]
        }
        data = _make_openlabel(traj)

        p = AngleSplineInterpolationPass(smoothing_factor=0.0)
        result = p.process(data)

        frames = result["openlabel"]["frames"]
        for frame_str, frame_data in frames.items():
            rbbox = frame_data["objects"]["obj_1"]["object_data"]["rbbox"][0]["val"]
            assert rbbox[2] >= rbbox[3], (
                f"Frame {frame_str}: width ({rbbox[2]}) should be >= "
                f"height ({rbbox[3]})"
            )

    def test_housekeeping_tag_added(self):
        """Processed frames should have 'spline' housekeeping tag."""
        traj = {
            "obj_1": [
                (i, 100.0 + i * 10.0, 200.0, 80, 40, 0.0)
                for i in range(10)
            ]
        }
        data = _make_openlabel(traj)

        p = AngleSplineInterpolationPass(smoothing_factor=0.0)
        result = p.process(data)

        frames = result["openlabel"]["frames"]
        for frame_str, frame_data in frames.items():
            obj = frame_data["objects"]["obj_1"]
            vec_list = obj["object_data"]["vec"]
            annotator_vals = None
            for vec_item in vec_list:
                if vec_item.get("name") == "annotator":
                    annotator_vals = vec_item["val"]
                    break
            assert annotator_vals is not None
            assert any("spline" in v for v in annotator_vals), (
                f"Frame {frame_str}: missing 'spline' housekeeping tag"
            )

    def test_statistics(self):
        """get_statistics() should return expected keys and values."""
        traj = {
            "obj_1": [
                (i, 100.0 + i * 10.0, 200.0, 80, 40, 0.0)
                for i in range(10)
            ],
            "obj_2": [
                (0, 50.0, 50.0, 40, 20, 0.0),
                (1, 60.0, 50.0, 40, 20, 0.0),
            ],
        }
        data = _make_openlabel(traj)

        p = AngleSplineInterpolationPass(smoothing_factor=0.0)
        p.process(data)
        stats = p.get_statistics()

        assert stats["objects_processed"] == 1
        assert stats["objects_skipped"] == 1
        assert stats["angles_updated"] == 10

    def test_smoothing_factor_effect(self):
        """Higher smoothing should produce less variation in angles on noisy data."""
        np.random.seed(42)
        n = 30
        xs = np.linspace(100, 400, n)
        ys = 200.0 + np.random.normal(0, 5, n)  # noisy y

        traj = {
            "obj_1": [
                (i, float(xs[i]), float(ys[i]), 80, 40, 0.0)
                for i in range(n)
            ]
        }

        # Exact interpolation (s=0)
        data_exact = _make_openlabel(traj)
        p_exact = AngleSplineInterpolationPass(smoothing_factor=0.0)
        result_exact = p_exact.process(data_exact)
        angles_exact = [a for _, a in _get_angles(result_exact, "obj_1")]

        # Smooth interpolation (s=large)
        data_smooth = _make_openlabel(traj)
        p_smooth = AngleSplineInterpolationPass(smoothing_factor=100.0)
        result_smooth = p_smooth.process(data_smooth)
        angles_smooth = [a for _, a in _get_angles(result_smooth, "obj_1")]

        # Variance of angles should be smaller with more smoothing
        var_exact = np.var(angles_exact)
        var_smooth = np.var(angles_smooth)
        assert var_smooth < var_exact, (
            f"Smooth variance ({var_smooth:.6f}) should be less than "
            f"exact variance ({var_exact:.6f})"
        )

    def test_multiple_objects(self):
        """Multiple objects should be processed independently."""
        traj = {
            "obj_1": [
                (i, 100.0 + i * 10.0, 200.0, 80, 40, 0.5)
                for i in range(10)
            ],
            "obj_2": [
                (i, 300.0, 100.0 + i * 10.0, 80, 40, 0.5)
                for i in range(10)
            ],
        }
        data = _make_openlabel(traj)

        p = AngleSplineInterpolationPass(smoothing_factor=0.0)
        result = p.process(data)

        # obj_1 moves right → angle ≈ 0
        for _, angle in _get_angles(result, "obj_1"):
            assert abs(angle) < 0.05

        # obj_2 moves down → angle ≈ π/2
        for _, angle in _get_angles(result, "obj_2"):
            assert abs(angle - math.pi / 2) < 0.05

        assert p.objects_processed == 2


class TestDeduplicatePositions:
    """Unit tests for the static _deduplicate_positions helper."""

    def test_no_duplicates(self):
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [10.0, 20.0, 30.0, 40.0]
        ux, uy, mapping = AngleSplineInterpolationPass._deduplicate_positions(
            xs, ys
        )
        assert ux == xs
        assert uy == ys
        assert mapping == [0, 1, 2, 3]

    def test_consecutive_duplicates(self):
        xs = [1.0, 1.0, 1.0, 2.0, 3.0, 3.0]
        ys = [10.0, 10.0, 10.0, 20.0, 30.0, 30.0]
        ux, uy, mapping = AngleSplineInterpolationPass._deduplicate_positions(
            xs, ys
        )
        assert ux == [1.0, 2.0, 3.0]
        assert uy == [10.0, 20.0, 30.0]
        assert mapping == [0, 0, 0, 1, 2, 2]

    def test_non_consecutive_duplicates_kept(self):
        """Non-consecutive duplicates should not be merged."""
        xs = [1.0, 2.0, 1.0, 3.0]
        ys = [10.0, 20.0, 10.0, 30.0]
        ux, uy, mapping = AngleSplineInterpolationPass._deduplicate_positions(
            xs, ys
        )
        assert ux == [1.0, 2.0, 1.0, 3.0]
        assert uy == [10.0, 20.0, 10.0, 30.0]
        assert mapping == [0, 1, 2, 3]

    def test_all_same(self):
        xs = [5.0, 5.0, 5.0]
        ys = [5.0, 5.0, 5.0]
        ux, uy, mapping = AngleSplineInterpolationPass._deduplicate_positions(
            xs, ys
        )
        assert ux == [5.0]
        assert uy == [5.0]
        assert mapping == [0, 0, 0]
