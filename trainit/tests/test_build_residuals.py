"""Residual table of a markit run against key-frame-reviewed ground truth (trainit.build_residuals)."""

import csv

import pytest

from trainit.build_residuals import COLUMNS, gt_frames, main, residual_rows


def _box(cx, cy, w=40.0, h=20.0, r=0.0, vec=None):
    return {"object_data": {"rbbox": [{"name": "shape", "val": [cx, cy, w, h, r]}], "vec": vec or []}}


def _files():
    gt = {"objects": {"10": {"type": "Car"}, "11": {"type": "Van"}}, "frames": {
        "0": {"frame_properties": {"review": {"annotator": "alice"}}, "objects": {"10": _box(100, 100), "11": _box(300, 300)}},
        "1": {"objects": {"10": _box(105, 100)}},
    }}
    detected = [{"name": "annotator", "val": ["markit_yolo"]}, {"name": "confidence", "val": [0.8]},
                {"name": "origin", "val": ["detected"]}]
    auto = {"objects": {"1": {"type": "Car"}, "2": {"type": "Car"}}, "frames": {
        "0": {"objects": {"1": _box(102, 99, vec=detected),
                          "2": _box(301, 300, 42, vec=detected + [{"name": "sigma_px", "val": [1, 1, 2, 2, 0.05]}])}},
        "1": {"objects": {"1": _box(105, 100, vec=detected)}},
    }}
    return gt, auto


def test_only_reviewed_frames_unless_all_frames():
    gt, _ = _files()
    assert gt_frames(gt, all_frames=False) == ["0"]
    assert gt_frames(gt, all_frames=True) == ["0", "1"]


def test_rows_have_residuals_and_baseline_sigma():
    gt, auto = _files()
    rows = {r["auto_id"]: r for r in residual_rows(gt, auto, "rec")}
    assert set(rows) == {"1", "2"}
    assert rows["1"]["res_cx"] == pytest.approx(2.0) and rows["1"]["res_cy"] == pytest.approx(-1.0)
    assert rows["1"]["gt_class"] == "Car" and rows["2"]["gt_class"] == "Van"  # class-agnostic matching
    assert rows["1"]["sigma_cx"] == pytest.approx(0.02 * (40 * 20) ** 0.5, abs=1e-3)  # default localization
    assert rows["2"]["sigma_w"] == 2 and rows["2"]["res_w"] == pytest.approx(2.0)  # sigma_px from the file
    assert rows["1"]["origin"] == "detected" and rows["1"]["score"] == pytest.approx(0.8)


def test_cli_writes_csv(tmp_path):
    import json
    gt, auto = _files()
    (tmp_path / "gt.json").write_text(json.dumps({"openlabel": gt}))
    (tmp_path / "auto.json").write_text(json.dumps({"openlabel": auto}))
    out = tmp_path / "res.csv"
    assert main(["--gt", str(tmp_path / "gt.json"), "--auto", str(tmp_path / "auto.json"), "--output", str(out)]) == 0
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 2 and list(rows[0]) == COLUMNS and rows[0]["recording"] == "gt"
