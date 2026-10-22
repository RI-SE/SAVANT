"""Error model fitting (trainit.fit_error_model) on synthetic data."""

import math

import numpy as np
import pytest

from trainit.fit_error_model import (
    expected_calibration_error,
    fit_sigma,
    isotonic,
    match_image,
    residual,
    rotated_iou,
    val_pairs,
)


def test_residual_uses_equivalent_box_form():
    # Same rectangle written as (w, h, r) and (h, w, r + pi/2): no residual
    assert residual([10, 10, 20, 8, 0.1 + math.pi / 2], [10, 10, 8, 20, 0.1]) == pytest.approx([0, 0, 0, 0, 0])
    assert residual([11, 10, 20, 8, 0.0], [10, 10, 20, 8, math.pi - 0.05])[4] == pytest.approx(0.05)


def test_rotated_iou_and_matching():
    box = [50, 50, 20, 10, 0.3]
    assert rotated_iou(box, box) == pytest.approx(1.0, abs=1e-3)
    assert rotated_iou(box, [200, 200, 20, 10, 0]) == 0.0
    matches, scored = match_image([(0, 0.9, box), (0, 0.4, [52, 50, 20, 10, 0.3]), (1, 0.8, box)], [(0, box)])
    assert len(matches) == 1 and scored == [(0.9, True), (0.8, False), (0.4, False)]


def test_fit_sigma_recovers_size_dependence():
    rng = np.random.default_rng(0)
    sizes = rng.uniform(10, 100, 5000)
    residuals = rng.normal(0, 1, (5000, 5)) * (0.5 + 0.02 * sizes)[:, None]
    a, b = fit_sigma(residuals, sizes)["cx"]
    assert a == pytest.approx(0.5, abs=0.1) and b == pytest.approx(0.02, abs=0.003)


def test_isotonic_is_monotone_and_calibrates():
    rng = np.random.default_rng(1)
    scores = rng.uniform(0, 1, 4000)
    hits = (rng.uniform(0, 1, 4000) < scores ** 2).astype(float)  # overconfident raw score
    x, y = isotonic(scores, hits)
    assert all(b >= a for a, b in zip(y, y[1:])) and all(b >= a for a, b in zip(x, x[1:]))
    assert expected_calibration_error(scores, hits) > 0.1


def test_val_pairs_follow_yolo_layout(tmp_path):
    (tmp_path / "images" / "val").mkdir(parents=True)
    (tmp_path / "images" / "val" / "f1.jpg").write_bytes(b"")
    (tmp_path / "data.yaml").write_text(f"path: {tmp_path}\ntrain: images/train\nval: images/val\n")
    assert val_pairs(tmp_path / "data.yaml") == [(tmp_path / "images/val/f1.jpg", tmp_path / "labels/val/f1.txt")]
