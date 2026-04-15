"""Tests for spline-based angle interpolation."""

import math
from collections import deque
from unittest.mock import MagicMock

import numpy as np
import pytest

from edit.models.OpenLabel import (
    AnnotatorData,
    FrameLevelObject,
    FrameObjects,
    GeometryData,
    ObjectData,
    ObjectMetadata,
    OpenLabel,
    OpenLabelMetadata,
    RotatedBBox,
)
from edit.services.annotation_service import AnnotationService
from edit.services.interpolation_service import InterpolationService
from edit.services.project_state import ProjectState


# ── InterpolationService.spline_interpolate_angles ────────────────────


class TestSplineInterpolateAngles:
    """Tests for the pure-math spline angle method."""

    def test_straight_line_horizontal(self):
        """Moving right → all angles ≈ 0."""
        positions = [(i * 10.0, 0.0) for i in range(10)]
        angles = InterpolationService.spline_interpolate_angles(positions, 0.0)
        assert len(angles) == 10
        for a in angles:
            assert abs(a) < 0.1

    def test_straight_line_vertical(self):
        """Moving up → all angles ≈ π/2."""
        positions = [(0.0, i * 10.0) for i in range(10)]
        angles = InterpolationService.spline_interpolate_angles(positions, 0.0)
        assert len(angles) == 10
        for a in angles:
            assert abs(a - math.pi / 2) < 0.1

    def test_straight_line_diagonal(self):
        """Moving at 45° → all angles ≈ π/4."""
        positions = [(i * 10.0, i * 10.0) for i in range(10)]
        angles = InterpolationService.spline_interpolate_angles(positions, 0.0)
        assert len(angles) == 10
        for a in angles:
            assert abs(a - math.pi / 4) < 0.1

    def test_curved_trajectory(self):
        """Quarter-circle arc: angles rotate from ~π/2 to ~π."""
        n = 20
        t = np.linspace(0, math.pi / 2, n)
        positions = [(float(np.cos(ti)), float(np.sin(ti))) for ti in t]
        angles = InterpolationService.spline_interpolate_angles(positions, 0.0)
        assert len(angles) == n
        # Tangent at t=0: dx/dt=-sin(0)=0, dy/dt=cos(0)=1 → atan2(1,0) = π/2
        assert abs(angles[0] - math.pi / 2) < 0.15
        # Tangent at t=π/2: dx/dt=-sin(π/2)=-1, dy/dt=cos(π/2)=0 → atan2(0,-1)=π
        assert abs(abs(angles[-1]) - math.pi) < 0.15

    def test_stationary_frames_forward_fill(self):
        """Repeated positions in the middle get forward-filled angles."""
        positions = [
            (0.0, 0.0),
            (10.0, 0.0),
            (20.0, 0.0),
            (20.0, 0.0),  # stationary
            (20.0, 0.0),  # stationary
            (30.0, 0.0),
            (40.0, 0.0),
        ]
        angles = InterpolationService.spline_interpolate_angles(positions, 0.0)
        assert len(angles) == 7
        # Stationary frames should have the same angle as last moving frame
        assert angles[3] == angles[2]
        assert angles[4] == angles[2]

    def test_too_few_unique_points_raises(self):
        positions = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
        with pytest.raises(ValueError, match="at least 4 unique"):
            InterpolationService.spline_interpolate_angles(positions, 0.0)

    def test_all_same_position_raises(self):
        positions = [(5.0, 5.0)] * 10
        with pytest.raises(ValueError, match="at least 4 unique"):
            InterpolationService.spline_interpolate_angles(positions, 0.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            InterpolationService.spline_interpolate_angles([], 0.0)

    def test_single_point_raises(self):
        with pytest.raises(ValueError):
            InterpolationService.spline_interpolate_angles([(1.0, 2.0)], 0.0)

    def test_smoothing_factor_effect(self):
        """Larger smoothing → reduced angle variation for zigzag."""
        positions = [
            (0.0, 0.0),
            (10.0, 10.0),
            (20.0, 0.0),
            (30.0, 10.0),
            (40.0, 0.0),
            (50.0, 10.0),
            (60.0, 0.0),
        ]
        angles_exact = InterpolationService.spline_interpolate_angles(
            positions, 0.0
        )
        angles_smooth = InterpolationService.spline_interpolate_angles(
            positions, 50.0
        )
        assert np.var(angles_smooth) < np.var(angles_exact)

    def test_output_length_matches_input(self):
        positions = [(float(i), float(i * 2)) for i in range(20)]
        angles = InterpolationService.spline_interpolate_angles(positions, 0.0)
        assert len(angles) == len(positions)

    def test_angles_are_finite(self):
        positions = [(float(i), float(i ** 2 * 0.01)) for i in range(10)]
        angles = InterpolationService.spline_interpolate_angles(positions, 0.0)
        for a in angles:
            assert math.isfinite(a)


class TestDeduplicatePositions:
    """Tests for the deduplication helper."""

    def test_no_duplicates(self):
        xs = [0.0, 1.0, 2.0, 3.0]
        ys = [0.0, 1.0, 2.0, 3.0]
        ux, uy, mapping = InterpolationService._deduplicate_positions(xs, ys)
        assert ux == xs
        assert uy == ys
        assert mapping == [0, 1, 2, 3]

    def test_consecutive_duplicates(self):
        xs = [0.0, 1.0, 1.0, 1.0, 2.0]
        ys = [0.0, 1.0, 1.0, 1.0, 2.0]
        ux, uy, mapping = InterpolationService._deduplicate_positions(xs, ys)
        assert ux == [0.0, 1.0, 2.0]
        assert uy == [0.0, 1.0, 2.0]
        assert mapping == [0, 1, 1, 1, 2]

    def test_non_consecutive_duplicates_preserved(self):
        xs = [0.0, 1.0, 0.0, 1.0]
        ys = [0.0, 1.0, 0.0, 1.0]
        ux, uy, mapping = InterpolationService._deduplicate_positions(xs, ys)
        assert ux == [0.0, 1.0, 0.0, 1.0]
        assert len(mapping) == 4

    def test_all_same(self):
        xs = [5.0, 5.0, 5.0]
        ys = [5.0, 5.0, 5.0]
        ux, uy, mapping = InterpolationService._deduplicate_positions(xs, ys)
        assert ux == [5.0]
        assert uy == [5.0]
        assert mapping == [0, 0, 0]


# ── AnnotationService.apply_spline_angle_interpolation ────────────────


def _make_frame_object(x, y, w=20.0, h=10.0, rotation=0.0):
    """Build a valid FrameLevelObject with a RotatedBBox."""
    bbox = RotatedBBox(
        x_center=x, y_center=y, width=w, height=h, rotation=rotation
    )
    return FrameLevelObject(
        object_data=ObjectData(
            rbbox=[GeometryData(val=bbox)],
            vec=[
                AnnotatorData(name="annotator", val=deque(["test"])),
            ],
        )
    )


def _build_annotation_service(frame_specs):
    """Build an AnnotationService from a list of (frame_num, obj_id, x, y, rot).

    Creates a valid OpenLabel model and ProjectState.
    """
    frames = {}
    obj_ids = set()
    for frame_num, obj_id, x, y, rot in frame_specs:
        fkey = str(frame_num)
        if fkey not in frames:
            frames[fkey] = FrameObjects(objects={})
        frames[fkey].objects[obj_id] = _make_frame_object(x, y, rotation=rot)
        obj_ids.add(obj_id)

    objects = {
        oid: ObjectMetadata(name=f"Object-{oid}", type="car")
        for oid in obj_ids
    }

    config = OpenLabel(
        metadata=OpenLabelMetadata(schema_version="1.0.0"),
        ontologies={},
        objects=objects,
        frames=frames,
    )

    state = MagicMock(spec=ProjectState)
    state.annotation_config = config
    state.interpolation_metadata = set()

    return AnnotationService(state)


class TestApplySplineAngleInterpolation:
    """Tests for AnnotationService.apply_spline_angle_interpolation."""

    def test_full_trajectory_horizontal(self):
        """All frames updated with horizontal trajectory → angles ≈ 0."""
        specs = [(i, "obj1", i * 10.0, 0.0, 1.5) for i in range(10)]
        service = _build_annotation_service(specs)

        frames = service.apply_spline_angle_interpolation(
            "obj1", 0.0, "tester"
        )
        assert len(frames) == 10
        for f in frames:
            bbox = service.get_bbox(f, "obj1")
            assert abs(bbox.rotation) < 0.1

    def test_frame_range_only_updates_range(self):
        """Only frames in [5, 14] updated; others untouched."""
        specs = [(i, "obj1", i * 10.0, 0.0, 1.5) for i in range(20)]
        service = _build_annotation_service(specs)

        frames = service.apply_spline_angle_interpolation(
            "obj1", 0.0, "tester", start_frame=5, end_frame=14
        )
        assert len(frames) == 10
        assert all(5 <= f <= 14 for f in frames)

        # Frame 0 should keep original angle
        bbox_0 = service.get_bbox(0, "obj1")
        assert abs(bbox_0.rotation - 1.5) < 0.01

    def test_nonexistent_object_raises(self):
        specs = [(0, "obj1", 0.0, 0.0, 0.0)]
        service = _build_annotation_service(specs)
        with pytest.raises(Exception):
            service.apply_spline_angle_interpolation(
                "nonexistent", 0.0, "tester"
            )
