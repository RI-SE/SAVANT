"""Per-box sigma_px / p_exist (savant_common.quality) as markit writes them."""

import json

import pytest

from savant_common.quality import (
    DEFAULT_LOCALIZATION,
    MODEL_SCHEMA,
    P_EXIST_ENTRY,
    SIGMA_ENTRY,
    apply_box_quality,
    box_sigma,
    calibrated_score,
    error_model_path,
    load_error_model,
    run_quality,
    sha256_file,
)

LOCALIZATION = {"by_class": {"_all": {"sigma": {c: [1.0, 0.0] for c in ("cx", "cy", "w", "h", "r")}},
                             "car": {"sigma": {c: [0.0, 0.1] for c in ("cx", "cy", "w", "h", "r")}}}}
QUALITY = {"localization": LOCALIZATION}
CALIBRATION = {"x": [0.2, 0.8], "y": [0.1, 0.9]}


def _box(annotators, origins=None, confidence=None, gap=None, size=(40.0, 10.0)):
    vec = [{"name": "annotator", "val": annotators}]
    if confidence is not None:
        vec.append({"name": "confidence", "val": confidence})
    if origins is not None:
        vec.append({"name": "origin", "val": origins})
    if gap is not None:
        vec.append({"name": "gap_distance", "val": [gap]})
    return {"object_data": {"rbbox": [{"name": "shape", "val": [50.0, 50.0, *size, 0.0]}], "vec": vec}}


def _vec(box):
    return {e["name"]: e["val"] for e in box["object_data"]["vec"]}


def test_sigma_by_class_and_origin():
    # size = sqrt(40*10) = 20 -> car: 0.1*20 = 2
    assert box_sigma(LOCALIZATION, "car", 40, 10, "detected", None) == [2.0] * 5
    assert box_sigma(LOCALIZATION, "bus", 40, 10, "detected", None) == [1.0] * 5  # pooled
    assert box_sigma(LOCALIZATION, "car", 40, 10, "reviewed", None) == [1.0] * 5
    assert box_sigma(LOCALIZATION, "car", 40, 10, "interpolated", 5) == [3.0] * 5
    assert box_sigma(LOCALIZATION, "car", 40, 10, "manual_tracked", None) == [5.0] * 5  # unknown gap = 15


def test_calibrated_score_is_clamped_and_interpolated():
    assert calibrated_score(CALIBRATION, 0.0) == 0.1
    assert calibrated_score(CALIBRATION, 0.5) == pytest.approx(0.5)
    assert calibrated_score(CALIBRATION, 1.0) == 0.9


def test_apply_box_quality_per_source_and_origin():
    boxes = {"1": _box(["markit_yolo"], ["detected"], [0.5]),
             "2": _box(["markit_aruco"], ["detected"], [1.0]),
             "3": _box(["markit_oflow"], ["detected"], [0.5]),
             "4": _box(["markit_housekeeping(gap)"], ["interpolated"], [0.8888], gap=2),
             "5": _box(["alice", "markit_yolo"], confidence=[1.0, 0.3])}
    openlabel = {"objects": {k: {"type": "car"} for k in boxes}, "frames": {"0": {"objects": boxes}}}

    done = apply_box_quality(openlabel, QUALITY, CALIBRATION)

    assert done == {"boxes": 4, "default_sigma": 1}
    assert _vec(boxes["1"])[SIGMA_ENTRY] == [2.0] * 5 and _vec(boxes["1"])[P_EXIST_ENTRY] == [0.5]
    assert SIGMA_ENTRY not in _vec(boxes["2"])  # marker
    assert _vec(boxes["3"])[SIGMA_ENTRY][0] == pytest.approx(0.4) and P_EXIST_ENTRY not in _vec(boxes["3"])
    assert _vec(boxes["4"])[SIGMA_ENTRY] == [2.4] * 5 and P_EXIST_ENTRY not in _vec(boxes["4"])
    assert _vec(boxes["5"])[SIGMA_ENTRY] == [1.0] * 5 and _vec(boxes["5"])[P_EXIST_ENTRY] == [1.0]


def test_only_origins_leaves_other_boxes_alone():
    boxes = {"1": _box(["markit_yolo"], ["detected"], [0.5]), "2": _box(["bob", "markit_yolo"], ["manual", "detected"])}
    openlabel = {"objects": {}, "frames": {"0": {"objects": boxes}}}
    apply_box_quality(openlabel, QUALITY, only_origins=["manual"])
    assert SIGMA_ENTRY not in _vec(boxes["1"]) and SIGMA_ENTRY in _vec(boxes["2"])


def test_error_model_must_match_the_weights(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"weights")
    model = {"schema": MODEL_SCHEMA, "model": {"sha256": sha256_file(weights)}, "localization": LOCALIZATION}
    error_model_path(weights).write_text(json.dumps(model))
    assert load_error_model(weights)["localization"] == LOCALIZATION
    assert run_quality(load_error_model(weights), weights)["error_model"]["basis"] == "measured"

    weights.write_bytes(b"other weights")
    assert load_error_model(weights) is None
    assert run_quality(None)["localization"] == DEFAULT_LOCALIZATION
