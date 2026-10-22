#!/usr/bin/env python3
"""
fit_error_model - Error model of trained YOLO OBB weights (scenario-uncertainty `model` profile, SPEC P4).

Runs the weights on the dataset's val split and writes error_model.json next to them:
- localization: residuals of matched predictions (same class, rotated IoU > 0.5) against ground truth,
  fitted per class and pooled as sigma = a + b*size_px, size_px = sqrt(w*h);
- score_calibration: isotonic fit of P(true positive | raw score), with the ECE of the raw score.

The val split must not overlap the training data (trainit's chunked split), or the errors are underestimated.
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from savant_common.quality import (
    ALL_CLASSES,
    ERROR_MODEL_FILE,
    MODEL_FORM,
    MODEL_SCHEMA,
    SIGMA_COMPONENTS,
    SIZE_PX,
    error_model_path,
    q,
    sha256_file,
)
from trainit import __version__

logger = logging.getLogger(__name__)

IOU_MATCH = 0.5
PREDICT_CONF = 0.01  # low, so the calibration sees the whole score range
MIN_CLASS_MATCHES = 30  # fewer matches: the class uses the pooled coefficients
ECE_BINS = 10
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
GAUSSIAN_ABS_TO_SIGMA = math.sqrt(math.pi / 2)  # E|r| = sigma * sqrt(2/pi)

Box = List[float]  # [cx, cy, w, h, r] in px and rad


def val_pairs(data_yaml) -> List[Tuple[Path, Path]]:
    """(image, label) paths of the val split; labels follow the YOLO images/ -> labels/ convention."""
    config = yaml.safe_load(Path(data_yaml).read_text())
    root = Path(config.get("path") or Path(data_yaml).parent)
    val_dir = root / config["val"]
    pairs = []
    for image in sorted(p for p in val_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
        parts = list(image.parts)
        parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
        pairs.append((image, Path(*parts).with_suffix(".txt")))
    return pairs


def read_labels(label_path: Path, width: int, height: int) -> List[Tuple[int, Box]]:
    """Ground-truth boxes of a YOLO OBB label file (class x1 y1 ... x4 y4, normalised)."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        values = line.split()
        if len(values) != 9:
            continue
        corners = np.array(values[1:], dtype=np.float32).reshape(4, 2) * [width, height]
        (cx, cy), (w, h), angle = cv2.minAreaRect(corners.astype(np.float32))
        boxes.append((int(values[0]), [cx, cy, w, h, math.radians(angle)]))
    return boxes


def rotated_iou(a: Box, b: Box) -> float:
    rect_a = ((a[0], a[1]), (a[2], a[3]), math.degrees(a[4]))
    rect_b = ((b[0], b[1]), (b[2], b[3]), math.degrees(b[4]))
    status, points = cv2.rotatedRectangleIntersection(rect_a, rect_b)
    if status == cv2.INTERSECT_NONE or points is None:
        return 0.0
    inter = cv2.contourArea(cv2.convexHull(points))
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def residual(pred: Box, truth: Box) -> Box:
    """pred - truth, with pred written in the equivalent form (w/h swapped, angle + pi/2) closest in angle."""
    best = None
    for w, h, r in ((pred[2], pred[3], pred[4]), (pred[3], pred[2], pred[4] + math.pi / 2)):
        dr = (r - truth[4] + math.pi / 2) % math.pi - math.pi / 2
        if best is None or abs(dr) < abs(best[4]):
            best = [pred[0] - truth[0], pred[1] - truth[1], w - truth[2], h - truth[3], dr]
    return best


def match_image(preds: Sequence[Tuple[int, float, Box]], truths: Sequence[Tuple[int, Box]]):
    """Greedy matching by score; (matches [(class, pred, truth)], [(score, is_true_positive)])."""
    used, matches, scored = set(), [], []
    for cls, score, box in sorted(preds, key=lambda p: -p[1]):
        best, best_iou = None, IOU_MATCH
        for i, (truth_cls, truth) in enumerate(truths):
            if i in used or truth_cls != cls:
                continue
            iou = rotated_iou(box, truth)
            if iou > best_iou:
                best, best_iou = i, iou
        scored.append((score, best is not None))
        if best is not None:
            used.add(best)
            matches.append((cls, box, truths[best][1]))
    return matches, scored


def fit_sigma(residuals: np.ndarray, sizes: np.ndarray) -> Dict[str, List[float]]:
    """[a, b] per component for sigma = a + b*size, fitted to |r|*sqrt(pi/2); a, b >= 0."""
    design = np.column_stack([np.ones_like(sizes), sizes])
    coefficients = {}
    for i, component in enumerate(SIGMA_COMPONENTS):
        target = np.abs(residuals[:, i]) * GAUSSIAN_ABS_TO_SIGMA
        (a, b), *_ = np.linalg.lstsq(design, target, rcond=None)
        if b < 0:  # no growth with size: constant RMS
            a, b = math.sqrt(float(np.mean(residuals[:, i] ** 2))), 0.0
        elif a < 0:  # proportional to size
            a, b = 0.0, float(target @ sizes / (sizes @ sizes))
        coefficients[component] = [round(float(a), 5), round(float(b), 6)]
    return coefficients


def isotonic(scores: np.ndarray, hits: np.ndarray) -> Tuple[List[float], List[float]]:
    """Pool-adjacent-violators fit of hits on scores; knots at the block mean scores."""
    order = np.argsort(scores)
    blocks = []  # [sum_hits, count, sum_scores]
    for score, hit in zip(scores[order], hits[order]):
        blocks.append([float(hit), 1, float(score)])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]:
            hit_sum, count, score_sum = blocks.pop()
            blocks[-1][0] += hit_sum
            blocks[-1][1] += count
            blocks[-1][2] += score_sum
    return ([round(b[2] / b[1], 5) for b in blocks], [round(b[0] / b[1], 5) for b in blocks])


def expected_calibration_error(probabilities: np.ndarray, hits: np.ndarray, bins: int = ECE_BINS) -> float:
    """Count-weighted mean |accuracy - confidence| over equal-width bins."""
    index = np.minimum((probabilities * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        in_bin = index == b
        if in_bin.any():
            total += in_bin.sum() * abs(hits[in_bin].mean() - probabilities[in_bin].mean())
    return float(total / len(probabilities))


def localization(matches: list, names: Dict[int, str]) -> dict:
    """Per-class and pooled coefficients from all matches."""
    by_class_rows: Dict[str, list] = {}
    for cls, pred, truth in matches:
        by_class_rows.setdefault(names.get(cls, str(cls)), []).append((residual(pred, truth), truth))
    groups = {ALL_CLASSES: [row for rows in by_class_rows.values() for row in rows]}
    groups.update({name: rows for name, rows in by_class_rows.items() if len(rows) >= MIN_CLASS_MATCHES})
    by_class = {}
    for name, rows in groups.items():
        residuals = np.array([r for r, _ in rows], dtype=float)
        sizes = np.array([math.sqrt(t[2] * t[3]) for _, t in rows], dtype=float)
        by_class[name] = {"sigma": fit_sigma(residuals, sizes), "n": len(rows)}
    return {"by_class": by_class, "model_form": MODEL_FORM, "size_px": SIZE_PX, "iou_match": IOU_MATCH}


def score_calibration(scored: list) -> dict:
    scores = np.array([s for s, _ in scored], dtype=float)
    hits = np.array([h for _, h in scored], dtype=float)
    x, y = isotonic(scores, hits)
    ece = expected_calibration_error(scores, hits)
    return {"method": "isotonic", "x": x, "y": y, "n": len(scored), "predict_conf": PREDICT_CONF,
            "ece": q(round(ece, 4), "1", "value", "none", "measured", n=len(scored),
                     note="ECE of the raw score on the val split")}


def collect(weights, pairs: List[Tuple[Path, Path]], imgsz: Optional[int], device: Optional[str]):
    """Run the weights on the val images; (matches, scored, class names)."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    labels = {str(image): label for image, label in pairs}
    matches, scored = [], []
    predict_args = {"conf": PREDICT_CONF, "stream": True, "verbose": False}
    predict_args.update({k: v for k, v in (("imgsz", imgsz), ("device", device)) if v is not None})
    for result in model.predict(source=[str(image) for image, _ in pairs], **predict_args):
        height, width = result.orig_shape
        truths = read_labels(labels[str(result.path)], width, height)
        obb = result.obb
        preds = [] if obb is None else [
            (int(c), float(s), [float(v) for v in box])
            for c, s, box in zip(obb.cls.cpu().numpy(), obb.conf.cpu().numpy(), obb.xywhr.cpu().numpy())]
        image_matches, image_scored = match_image(preds, truths)
        matches += image_matches
        scored += image_scored
    return matches, scored, dict(model.names)


def fit(weights, data_yaml, imgsz: Optional[int] = None, device: Optional[str] = None,
        metrics: Optional[dict] = None) -> dict:
    """The error model document of these weights on the val split of data_yaml."""
    pairs = val_pairs(data_yaml)
    if not pairs:
        raise ValueError(f"No val images found for {data_yaml}")
    matches, scored, names = collect(weights, pairs, imgsz, device)
    if not matches:
        raise ValueError("No prediction matched the ground truth; cannot fit an error model")
    logger.info(f"Error model: {len(pairs)} val images, {len(matches)} matched boxes, {len(scored)} predictions")
    return {
        "schema": MODEL_SCHEMA,
        "tool": f"fit_error_model/{__version__}",
        "model": {"file": Path(weights).name, "sha256": sha256_file(weights)},
        "dataset": {"yaml": str(data_yaml), "split": "val", "val_frames": len(pairs)},
        "localization": localization(matches, names),
        "score_calibration": score_calibration(scored),
        "metrics": metrics or {},
    }


def write(document: dict, weights) -> Path:
    path = error_model_path(weights)
    path.write_text(json.dumps(document, indent=2))
    logger.info(f"Error model written to {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Fit {ERROR_MODEL_FILE} of YOLO OBB weights on a val split")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--weights", required=True, help="Trained weights (.pt); the model is written next to them")
    parser.add_argument("--data", required=True, help="Dataset YAML with a val split not used for training")
    parser.add_argument("--imgsz", type=int, default=None, help="Inference image size (default: the model's)")
    parser.add_argument("--device", default=None, help="Device, e.g. 0 or cpu")
    parser.add_argument("--provenance", default=None,
                        help="Path to provenance chain file (will be created if not exists)")
    return parser


def record_provenance(args: argparse.Namespace, started_at: str, model_path: Path) -> None:
    """Append this fit to the --provenance chain; the error model is a scenario-uncertainty `model` output."""
    from datetime import datetime, timezone

    from scenario_uncertainty import provenance

    from savant_common.provenance import arguments_string, load_or_create_chain

    chain = load_or_create_chain(args.provenance, entity_id="savant_error_model", initial_source=args.weights,
                                 description="SAVANT trainit error model")
    chain.add(started_at=started_at, ended_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              tool_name="fit_error_model", tool_version=__version__, operation="error model fit",
              inputs=[args.weights, args.data], input_formats=["PT", "YAML"], outputs=[str(model_path)],
              output_formats=[provenance.output_format("model", 1)],
              arguments=arguments_string(build_parser(), args), capture_agent=True, agent_type="automated",
              capture_environment=True)
    chain.save(args.provenance)


def main():
    from datetime import datetime, timezone

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = build_parser().parse_args()
    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        path = write(fit(args.weights, args.data, args.imgsz, args.device), args.weights)
    except (OSError, ValueError, KeyError) as e:
        logger.error(f"Error model fit failed: {e}")
        sys.exit(1)
    if args.provenance:
        record_provenance(args, started_at, path)


if __name__ == "__main__":
    main()
