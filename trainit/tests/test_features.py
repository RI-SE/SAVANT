"""Per-box features for the box-uncertainty study (savant_common.features)."""

import math
from pathlib import Path

import numpy as np
import pytest

from savant_common.features import (
    _angle_std,
    flow_features,
    iter_gray_frames,
    patch_features,
    position_features,
    track_features,
    track_index,
)

VIDEO = Path(__file__).parents[2] / "edit" / "tests" / "assets" / "Kraklanda_short.mp4"


def test_position_features():
    assert position_features([50, 25, 10, 10, 0], 100, 50) == {"radial": 0.0, "edge_dist": 25.0}
    assert position_features([0, 0, 10, 10, 0], 100, 50)["radial"] == pytest.approx(1.0)


def test_track_features_constant_motion():
    frames = np.arange(20)
    boxes = np.array([[10 + 2 * f, 50, 40, 20, 0.1] for f in frames], dtype=float)
    features = track_features(frames, boxes, 10)
    assert features["speed"] == pytest.approx(2.0)
    assert features["jitter"] == pytest.approx(0.0, abs=1e-9)
    assert features["w_std"] == 0 and features["r_std"] == pytest.approx(0.0, abs=1e-6)
    assert features["track_age"] == 10 and features["track_left"] == 9


def test_single_box_track_has_nan_motion():
    features = track_features(np.array([5]), np.array([[1, 1, 2, 2, 0.0]]), 5)
    assert math.isnan(features["speed"]) and features["track_age"] == 0


def test_angle_std_is_modulo_pi():
    assert _angle_std(np.array([0.01, math.pi - 0.01])) == pytest.approx(0.01, abs=1e-3)


def test_track_index():
    openlabel = {"frames": {"3": {"objects": {"1": {"object_data": {"rbbox": [{"name": "shape", "val": [1, 2, 3, 4, 0]}]}}}},
                            "1": {"objects": {"1": {"object_data": {"rbbox": [{"name": "shape", "val": [0, 2, 3, 4, 0]}]}}}}}}
    frames, boxes = track_index(openlabel)["1"]
    assert list(frames) == [1, 3] and boxes[1][0] == 1


def test_patch_features_flat_vs_edges():
    flat = np.full((100, 100), 128, np.uint8)
    edges = flat.copy()
    edges[:, 50:] = 255
    box = [50, 50, 40, 40, 0.0]
    assert patch_features(flat, box)["edge_density"] == 0 and patch_features(flat, box)["contrast"] == 0
    assert patch_features(edges, box)["edge_density"] > 0 and patch_features(edges, box)["sharpness"] > 0
    assert math.isnan(patch_features(flat, [500, 500, 10, 10, 0])["brightness"])


def test_flow_features_detect_shift():
    rng = np.random.default_rng(0)
    image = (rng.random((120, 120)) * 255).astype(np.uint8)
    image = np.array(__import__("cv2").GaussianBlur(image, (7, 7), 2))
    shifted = np.roll(image, 2, axis=1)
    assert flow_features(image, shifted, [60, 60, 50, 50, 0])["flow_mag"] > 1.0
    assert math.isnan(flow_features(None, shifted, [60, 60, 50, 50, 0])["flow_mag"])


@pytest.mark.skipif(not VIDEO.exists(), reason="test video missing")
def test_iter_gray_frames_yields_wanted_frames_with_previous():
    got = [(f, prev is not None) for f, prev, _ in iter_gray_frames(VIDEO, [5, 0, 6, 100000])]
    assert got == [(0, False), (5, True), (6, True)]
