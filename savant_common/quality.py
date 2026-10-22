"""Per-box quality (scenario-uncertainty SPEC P3/P4/P7): sigma_px and p_exist from a trainit error model.

Shared by markit (writes them) and edit (recomputes them for boxes a person changed); standard library only,
so edit does not need scenario-uncertainty. Quality values are plain Q dicts (SPEC §3).
"""

import bisect
import hashlib
import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from savant_common.openlabel import (
    GAP_DISTANCE_ENTRY,
    HOUSEKEEPING_ANNOTATOR,
    HUMAN_ORIGINS,
    ORIGIN_ENTRY,
    UNKNOWN_ORIGIN,
    Origin,
    aligned_origins,
    source_confidence,
)

logger = logging.getLogger(__name__)

QUALITY_SPEC = "scenario-uncertainty/0.1"
MODEL_SCHEMA = "scenario-uncertainty/model/1"
ERROR_MODEL_FILE = "error_model.json"
SIGMA_ENTRY = "sigma_px"
P_EXIST_ENTRY = "p_exist"
SIGMA_COMPONENTS = ("cx", "cy", "w", "h", "r")  # px, px, px, px, rad
ALL_CLASSES = "_all"  # pooled coefficients, used for classes without their own
MODEL_FORM = "sigma = a + b*size_px"
SIZE_PX = "sqrt(w*h)"

# SPEC P7 default [a, b] per component for detected boxes, until a measured error model exists
DEFAULT_LOCALIZATION = {"by_class": {ALL_CLASSES: {"sigma": {
    "cx": [0.0, 0.02], "cy": [0.0, 0.02], "w": [0.0, 0.04], "h": [0.0, 0.04], "r": [0.05, 0.0]}}},
    "model_form": MODEL_FORM, "size_px": SIZE_PX}
# SPEC P7 origin rule: confirmed by a person = factor x detected; synthesized = detected x (1 + growth*k)
ORIGIN_RULE = {"human_factor": 0.5, "gap_growth": 0.1, "unknown_gap": 15}

# Synthesized between real boxes, by markit or by edit's spline
INTERPOLATED_ORIGINS = (Origin.INTERPOLATED, Origin.MANUAL_INTERPOLATED)
# A human drew, changed or confirmed the box itself; edit's spline and tracker do not count
REVIEWED_ORIGINS = (Origin.MANUAL, Origin.REVIEWED)
SYNTHESIZED_ORIGINS = (Origin.INTERPOLATED, Origin.TRACKED, Origin.MANUAL_INTERPOLATED, Origin.MANUAL_TRACKED)
# Origins whose score is the detector's own, so the score calibration applies
SCORED_ORIGINS = (Origin.DETECTED, Origin.MERGED)

# Detector the error model describes, and other markit engines (these get the default sigma)
MODEL_SOURCES = ("markit_yolo",)
DEFAULT_SOURCES = ("markit_oflow",)
# Marker boxes: their accuracy is not a detector's box error
SKIPPED_SOURCES = ("markit_aruco",)


def q(value, unit: str, stat: str, level: str, basis: str, **extra) -> dict:
    """A Q dict (SPEC §3); optional n/source/note only when given."""
    out = {"value": value, "unit": unit, "stat": stat, "level": level, "basis": basis}
    out.update({key: val for key, val in extra.items() if val is not None})
    return out


def origin_shares(counts: Mapping[str, int], source: str = None) -> dict:
    """share_interpolated and share_reviewed (SPEC P3.1) of a set of boxes, from box counts per origin."""
    n = sum(counts.values())

    def share(origins: Iterable[str]) -> dict:
        if not n:
            return q(None, "1", "ratio", "none", "unknown", n=0, source=source)
        value = round(sum(counts.get(origin, 0) for origin in origins) / n, 4)
        return q(value, "1", "ratio", "none", "measured", n=n, source=source)

    return {"share_interpolated": share(INTERPOLATED_ORIGINS), "share_reviewed": share(REVIEWED_ORIGINS)}


def count_origins(openlabel: dict) -> Counter:
    """Boxes per newest origin in an OpenLABEL dict ({"frames": ...})."""
    counts = Counter()
    for frame in (openlabel.get("frames") or {}).values():
        for frame_obj in (frame.get("objects") or {}).values():
            history = _origin_history(_vec_values((frame_obj.get("object_data") or {}).get("vec") or []))
            counts[history[0] if history else UNKNOWN_ORIGIN] += 1
    return counts


def _vec_values(vec: list) -> dict:
    return {entry.get("name"): list(entry.get("val") or []) for entry in vec}


def _origin_history(values: dict) -> list:
    """Origin history of a box (inferred for older files); empty without annotator and origin entries."""
    return aligned_origins(values.get("annotator"), values.get(ORIGIN_ENTRY)) or values.get(ORIGIN_ENTRY) or []


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def error_model_path(weights) -> Path:
    """error_model.json next to the weights file."""
    return Path(weights).with_name(ERROR_MODEL_FILE)


def load_error_model(weights) -> Optional[dict]:
    """The error model fitted for these weights, or None (missing, other schema or other weights: warns)."""
    path = error_model_path(weights)
    if not path.exists():
        return None
    model = json.loads(path.read_text())
    if model.get("schema") != MODEL_SCHEMA:
        logger.warning(f"{path}: unsupported schema {model.get('schema')!r}, expected {MODEL_SCHEMA}; ignored")
        return None
    if (model.get("model") or {}).get("sha256") != sha256_file(weights):
        logger.warning(f"{path} was fitted for other weights than {weights}; ignored")
        return None
    return model


def run_quality(model: Optional[dict], weights=None) -> dict:
    """metadata.quality (SPEC P3.1) of a markit run, before the box pass adds defaults_used and shares.

    Carries the localization coefficients and the origin rule, so edit can recompute boxes without the model file.
    """
    if model is None:
        return {"spec": QUALITY_SPEC, "error_model": None, "score_calibration": None,
                "localization": DEFAULT_LOCALIZATION, "origin_rule": ORIGIN_RULE}
    calibration = model.get("score_calibration")
    return {
        "spec": QUALITY_SPEC,
        "error_model": {"file": ERROR_MODEL_FILE, "sha256": sha256_file(error_model_path(weights)),
                        "basis": "measured"},
        "score_calibration": None if not calibration else {
            "file": ERROR_MODEL_FILE, "method": calibration.get("method"), "ece": calibration.get("ece")},
        "localization": model["localization"],
        "origin_rule": ORIGIN_RULE,
    }


def box_sigma(localization: dict, class_name: str, width: float, height: float, origin: str,
              gap_distance: Optional[int]) -> List[float]:
    """[σcx, σcy, σw, σh, σr] of one box, with the SPEC P7 origin rule applied."""
    by_class = localization["by_class"]
    coefficients = (by_class.get(class_name) or by_class[ALL_CLASSES])["sigma"]
    size = math.sqrt(max(width, 0.0) * max(height, 0.0))
    factor = 1.0
    if origin in SYNTHESIZED_ORIGINS:
        k = ORIGIN_RULE["unknown_gap"] if gap_distance is None else gap_distance
        factor = 1.0 + ORIGIN_RULE["gap_growth"] * k
    elif origin in REVIEWED_ORIGINS:
        factor = ORIGIN_RULE["human_factor"]
    return [round((a + b * size) * factor, 4) for a, b in (coefficients[c] for c in SIGMA_COMPONENTS)]


def calibrated_score(calibration: dict, score: float) -> float:
    """P(true positive) for a raw score: piecewise-linear through the calibration knots, clamped at the ends."""
    xs, ys = calibration["x"], calibration["y"]
    i = bisect.bisect_left(xs, score)
    if i == 0:
        return float(ys[0])
    if i == len(xs):
        return float(ys[-1])
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    return float(y0 + (y1 - y0) * (score - x0) / (x1 - x0)) if x1 > x0 else float(y1)


def apply_box_quality(openlabel: dict, quality: dict, calibration: Optional[dict] = None,
                      only_origins: Optional[Iterable[str]] = None) -> Dict[str, int]:
    """Write sigma_px (and p_exist) into every box of an OpenLABEL dict ({"objects", "frames"}).

    only_origins restricts the pass to boxes with these newest origins (edit: the human ones).
    Returns {"boxes": boxes given a value, "default_sigma": those of them with the SPEC P7 default}.
    """
    objects = openlabel.get("objects") or {}
    only = None if only_origins is None else set(only_origins)
    done = {"boxes": 0, "default_sigma": 0}
    for frame in (openlabel.get("frames") or {}).values():
        for obj_id, frame_obj in (frame.get("objects") or {}).items():
            data = frame_obj.get("object_data") or {}
            used = _apply_to_box(data, (objects.get(obj_id) or {}).get("type"), quality, calibration, only)
            if used is not None:
                done["boxes"] += 1
                done["default_sigma"] += used is DEFAULT_LOCALIZATION
    return done


def _apply_to_box(data: dict, class_name: Optional[str], quality: dict, calibration: Optional[dict],
                  only: Optional[set]) -> Optional[dict]:
    """Set sigma_px/p_exist on one box; the localization used, or None if the box was left alone."""
    vec = data.setdefault("vec", [])
    values = _vec_values(vec)
    shape = next((g.get("val") for g in data.get("rbbox") or [] if g.get("name") == "shape"), None)
    annotators = values.get("annotator") or []
    history = _origin_history(values)
    source = box_source(annotators)
    if not shape or not history or source in SKIPPED_SOURCES or (only is not None and history[0] not in only):
        return None
    origin = history[0]
    localization = DEFAULT_LOCALIZATION if source in DEFAULT_SOURCES else quality["localization"]
    gap = (values.get(GAP_DISTANCE_ENTRY) or [None])[0]
    _set_entry(vec, SIGMA_ENTRY, box_sigma(localization, class_name, shape[2], shape[3], origin, gap))
    if origin in HUMAN_ORIGINS:  # a person placed or confirmed it
        _set_entry(vec, P_EXIST_ENTRY, [1.0])
    elif origin in SCORED_ORIGINS and calibration and source in MODEL_SOURCES:
        score = source_confidence(annotators, values.get("confidence"))
        if score is not None:
            _set_entry(vec, P_EXIST_ENTRY, [round(calibrated_score(calibration, score), 4)])
    return localization


def box_source(annotators: List[str]) -> Optional[str]:
    """The oldest annotator that is not a housekeeping step: who made the box first (None if unknown)."""
    for annotator in reversed(list(annotators)):
        if not str(annotator).startswith(HOUSEKEEPING_ANNOTATOR):
            return str(annotator)
    return None


def _set_entry(vec: list, name: str, val: list) -> None:
    for entry in vec:
        if entry.get("name") == name:
            entry["val"] = val
            return
    vec.append({"name": name, "val": val})
