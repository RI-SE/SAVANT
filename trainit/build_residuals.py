#!/usr/bin/env python3
"""
build_residuals - Box errors of a markit run against human ground truth, one row per matched box (CSV).

Input for the learned box-uncertainty study (development.md, "Learned per-box uncertainty"):
- GT: an OpenLABEL file corrected in edit. Only frames confirmed in edit's key-frame review
  (frame_properties.review) are used, since only there every box was checked; --all-frames uses every GT frame
  (older files without review marks; their unchanged boxes bias the errors towards 0).
- AUTO: markit output for the same video, made with a detector that was not trained on this recording.

Boxes are matched per frame, class-agnostic, greedily by rotated IoU (class errors are a separate question).
The residual convention (auto - GT, w/h swap and angle wrap) is that of fit_error_model.residual.
The baseline sigma is the auto box's sigma_px, or else box_sigma() from the run's metadata.quality.
Feature columns (savant_common.features) are added for each auto box; image and optical-flow features need --video.
"""

import argparse
import csv
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from savant_common.features import (
    BOX_FEATURES,
    FLOW_FEATURES,
    PATCH_FEATURES,
    POSITION_FEATURES,
    TRACK_FEATURES,
    box_features,
    flow_features,
    iter_gray_frames,
    patch_features,
    position_features,
    track_features,
    track_index,
    video_size,
)
from savant_common.openlabel import FRAME_REVIEW_KEY, GAP_DISTANCE_ENTRY, source_confidence
from savant_common.quality import (
    DEFAULT_LOCALIZATION,
    SIGMA_COMPONENTS,
    SIGMA_ENTRY,
    _origin_history,
    _vec_values,
    box_sigma,
    box_source,
)
from trainit.fit_error_model import Box, residual, rotated_iou

logger = logging.getLogger(__name__)

IOU_MATCH = 0.5
GEOMETRY = ("cx", "cy", "w", "h", "r")
COLUMNS = (
    ["recording", "frame", "auto_id", "gt_id", "auto_class", "gt_class", "source", "origin", "gap_distance",
     "score", "iou"]
    + list(GEOMETRY)
    + [f"res_{c}" for c in SIGMA_COMPONENTS]
    + [f"sigma_{c}" for c in SIGMA_COMPONENTS]
)
FEATURE_COLUMNS = list(BOX_FEATURES + POSITION_FEATURES + TRACK_FEATURES + PATCH_FEATURES + FLOW_FEATURES)


def load(path) -> dict:
    """The openlabel section of an OpenLABEL file."""
    return json.loads(Path(path).read_text())["openlabel"]


def gt_frames(gt: dict, all_frames: bool) -> List[str]:
    """Frame keys usable as ground truth: confirmed in key-frame review, or all with --all-frames."""
    frames = gt.get("frames") or {}
    keys = [k for k, f in frames.items()
            if all_frames or ((f.get("frame_properties") or {}).get(FRAME_REVIEW_KEY))]
    return sorted(keys, key=int)


def frame_boxes(openlabel: dict, frame_key: str) -> List[Tuple[str, dict, Box]]:
    """(object id, object_data, [cx, cy, w, h, r]) of every box in a frame."""
    frame = (openlabel.get("frames") or {}).get(frame_key) or {}
    boxes = []
    for obj_id, frame_obj in (frame.get("objects") or {}).items():
        data = frame_obj.get("object_data") or {}
        shape = next((g.get("val") for g in data.get("rbbox") or [] if g.get("name") == "shape"), None)
        if shape and len(shape) >= 5:
            boxes.append((obj_id, data, [float(v) for v in shape[:5]]))
    return boxes


def greedy_match(autos: List[Tuple[str, dict, Box]], truths: List[Tuple[str, dict, Box]]):
    """Pairs (auto, truth, iou), highest IoU first, each box used once, IoU >= IOU_MATCH."""
    candidates = sorted(
        ((rotated_iou(a[2], t[2]), i, j) for i, a in enumerate(autos) for j, t in enumerate(truths)),
        reverse=True)
    used_a, used_t, pairs = set(), set(), []
    for iou, i, j in candidates:
        if iou < IOU_MATCH:
            break
        if i in used_a or j in used_t:
            continue
        used_a.add(i)
        used_t.add(j)
        pairs.append((autos[i], truths[j], iou))
    return pairs


def baseline_sigma(data: dict, class_name: Optional[str], box: Box, localization: dict) -> Optional[List[float]]:
    """The box's sigma_px, or box_sigma() from the run's localization; None for boxes without origin."""
    values = _vec_values(data.get("vec") or [])
    if values.get(SIGMA_ENTRY):
        return [float(v) for v in values[SIGMA_ENTRY]]
    history = _origin_history(values)
    if not history:
        return None
    gap = (values.get(GAP_DISTANCE_ENTRY) or [None])[0]
    return box_sigma(localization, class_name, box[2], box[3], history[0], gap)


def residual_rows(gt: dict, auto: dict, recording: str, all_frames: bool = False) -> Iterator[Dict]:
    """One row (COLUMNS) per matched auto/GT box pair in the GT frames."""
    localization = ((auto.get("metadata") or {}).get("quality") or {}).get("localization") or DEFAULT_LOCALIZATION
    gt_objects, auto_objects = gt.get("objects") or {}, auto.get("objects") or {}
    for frame_key in gt_frames(gt, all_frames):
        pairs = greedy_match(frame_boxes(auto, frame_key), frame_boxes(gt, frame_key))
        for (auto_id, data, box), (gt_id, _, truth), iou in pairs:
            auto_class = (auto_objects.get(auto_id) or {}).get("type")
            sigma = baseline_sigma(data, auto_class, box, localization)
            if sigma is None:
                continue
            values = _vec_values(data.get("vec") or [])
            annotators = values.get("annotator") or []
            row = {
                "recording": recording, "frame": int(frame_key), "auto_id": auto_id, "gt_id": gt_id,
                "auto_class": auto_class, "gt_class": (gt_objects.get(gt_id) or {}).get("type"),
                "source": box_source(annotators), "origin": _origin_history(values)[0],
                "gap_distance": (values.get(GAP_DISTANCE_ENTRY) or [""])[0],
                "score": source_confidence(annotators, values.get("confidence")), "iou": round(iou, 4),
            }
            row.update({c: round(v, 4) for c, v in zip(GEOMETRY, box)})
            row.update({f"res_{c}": round(v, 5) for c, v in zip(SIGMA_COMPONENTS, residual(box, truth))})
            row.update({f"sigma_{c}": v for c, v in zip(SIGMA_COMPONENTS, sigma)})
            yield row


def add_features(rows: List[Dict], auto: dict, video=None, image_size: Optional[Tuple[int, int]] = None) -> None:
    """Add the FEATURE_COLUMNS of each row's auto box in place (NaN where an input is missing)."""
    tracks = track_index(auto)
    size = video_size(video) if video else image_size
    for row in rows:
        box = [row[c] for c in GEOMETRY]
        row.update(box_features(box))
        row.update(position_features(box, *size) if size else dict.fromkeys(POSITION_FEATURES, math.nan))
        frames, boxes = tracks[row["auto_id"]]
        row.update(track_features(frames, boxes, row["frame"]))
        row.update(dict.fromkeys(PATCH_FEATURES + FLOW_FEATURES, math.nan))
    if not video:
        return
    by_frame = defaultdict(list)
    for row in rows:
        by_frame[row["frame"]].append(row)
    for frame, previous, gray in iter_gray_frames(video, list(by_frame)):
        for row in by_frame[frame]:
            box = [row[c] for c in GEOMETRY]
            row.update(patch_features(gray, box))
            row.update(flow_features(previous, gray, box))


def write_csv(rows: List[Dict], path: Path, columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt", required=True, help="OpenLABEL ground truth (corrected in edit)")
    parser.add_argument("--auto", required=True, help="OpenLABEL markit output for the same video")
    parser.add_argument("--output", required=True, help="CSV file to write")
    parser.add_argument("--recording", help="Recording name for grouping in the study (default: GT file stem)")
    parser.add_argument("--video", help="The video both files annotate; adds image and optical-flow features")
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"),
                        help="Image size for position features when no --video is given")
    parser.add_argument("--all-frames", action="store_true",
                        help="Use every GT frame, not only frames confirmed in key-frame review")
    return parser


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    gt, auto = load(args.gt), load(args.auto)
    if not gt_frames(gt, args.all_frames):
        logger.error(f"{args.gt}: no frames confirmed in key-frame review (use --all-frames for older files)")
        return 1
    rows = list(residual_rows(gt, auto, args.recording or Path(args.gt).stem, args.all_frames))
    add_features(rows, auto, args.video, args.image_size)
    write_csv(rows, Path(args.output), COLUMNS + FEATURE_COLUMNS)
    logger.info(f"{len(rows)} matched boxes in {len(gt_frames(gt, args.all_frames))} GT frames -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
