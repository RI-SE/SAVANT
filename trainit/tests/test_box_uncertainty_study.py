"""Learned box-uncertainty study on synthetic residuals (trainit.box_uncertainty_study)."""

import csv
import json

import numpy as np
import pytest

pytest.importorskip("sklearn")

from trainit.box_uncertainty_study import (  # noqa: E402
    ONE_SIGMA,
    folds,
    held_out_bias,
    metrics,
    run,
    scale_for_coverage,
)
from trainit.build_residuals import COLUMNS, FEATURE_COLUMNS  # noqa: E402


def _table(path, recording, rng, n=1500):
    """Residuals whose spread depends on contrast, not on the baseline sigma (which is constant)."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS + FEATURE_COLUMNS)
        writer.writeheader()
        for i in range(n):
            contrast = rng.uniform(5, 60)
            true_sigma = 6.0 - 0.08 * contrast
            row = dict.fromkeys(COLUMNS + FEATURE_COLUMNS, "nan")
            row.update({"recording": recording, "frame": i, "auto_id": i, "gt_id": f"{recording}-{i // 10}",
                        "auto_class": "Car", "source": "markit_yolo", "origin": "detected", "score": 0.8,
                        "gap_distance": ""})
            row.update({f"res_{c}": rng.normal(0, true_sigma) for c in ("cx", "cy", "w", "h")})
            row["res_r"] = rng.normal(0, 0.02)
            row.update({f"sigma_{c}": 2.0 for c in ("cx", "cy", "w", "h")})
            row["sigma_r"] = 0.05
            row.update({"contrast": contrast, "brightness": rng.uniform(50, 200), "size": rng.uniform(40, 120),
                        "w": 60.0, "h": 30.0})
            writer.writerow(row)


def test_scale_and_metrics():
    rng = np.random.default_rng(0)
    residual = rng.normal(0, 3, 20000)
    assert scale_for_coverage(np.abs(residual) / 1.0) == pytest.approx(3.0, rel=0.05)
    m = metrics(residual, np.full_like(residual, 3.0))
    assert m["cov1"] == pytest.approx(ONE_SIGMA, abs=0.02) and m["cov2"] == pytest.approx(0.954, abs=0.01)


def test_folds_by_object_for_one_recording():
    rows = [{"recording": "a", "gt_id": str(i // 3)} for i in range(60)]
    fold, kind = folds(rows)
    assert kind == "object" and len(set(fold)) == 5
    assert all(len({fold[i] for i in range(60) if rows[i]["gt_id"] == g}) == 1 for g in {r["gt_id"] for r in rows})


def test_rf_learns_contrast_dependence(tmp_path):
    rng = np.random.default_rng(1)
    paths = [tmp_path / f"{name}.csv" for name in ("a", "b", "c")]
    for path in paths:
        _table(path, path.stem, rng)
    report = run(paths, tmp_path / "out")
    cx = report["components"]["cx"]
    assert report["folds"] == "recording" and report["n_folds"] == 3
    assert cx["pooled"]["rf"]["nll"] < cx["pooled"]["baseline_scaled"]["nll"]
    assert cx["pooled"]["rf"]["spearman"] > 0.2
    assert cx["pooled"]["rf"]["cov1"] == pytest.approx(ONE_SIGMA, abs=0.05)
    assert max(cx["importance"], key=cx["importance"].get) == "contrast"
    assert json.loads((tmp_path / "out" / "report.json").read_text())["n_rows"] == 4500
    assert (tmp_path / "out" / "importance.png").exists()


def test_held_out_bias_is_removed():
    """Loose boxes: w is 10 % too large; the bias is fitted on the other folds and explains most of the error."""
    rng = np.random.default_rng(2)
    rows = []
    for i in range(600):
        w, h = rng.uniform(40, 200), rng.uniform(20, 80)
        res = {f"res_{c}": rng.normal(0, 1.0) for c in ("cx", "cy", "w", "h", "r")}
        res["res_w"] += 0.1 * (w * h) ** 0.5
        rows.append({"recording": "a", "gt_id": str(i // 6), "auto_class": "Car", "w": w, "h": h, **res})
    fold, _ = folds(rows)
    bias = held_out_bias(rows, fold)
    res_w = np.array([r["res_w"] for r in rows])
    assert np.std(res_w - bias[:, 2]) == pytest.approx(1.0, abs=0.1)
    assert np.abs(bias[:, 0]).max() < 0.5
