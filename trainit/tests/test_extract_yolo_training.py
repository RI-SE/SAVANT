"""Tests for converting annotation boxes to YOLO OBB label corners."""

import math

import cv2
import numpy as np
import pytest

from trainit.extract_yolo_training import YOLODatasetExtractor

W, H = 200, 100
to_corners = YOLODatasetExtractor._bbox_to_corners


def _pixels(corners):
    return (np.array(corners, np.float32).reshape(4, 2) * np.float32([W, H])).astype(np.float32)


def _trained_box(pts):
    """Ultralytics fits a min-area rectangle to the 4 label points."""
    return cv2.boxPoints(cv2.minAreaRect(pts))


def test_inside_box_is_unchanged():
    pts = _pixels(to_corners(None, 100, 50, 40, 20, 0.3, W, H))
    assert math.isclose(cv2.contourArea(pts), 800, rel_tol=1e-3)


def test_edge_crossing_box_matches_visible_part_better_than_corner_clamping():
    from savant_common.geometry import BBoxOverlapCalculator, bbox_to_corners

    box = bbox_to_corners(190, 50, 40, 20, 0.4).astype(np.float32)
    image = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    visible = cv2.intersectConvexConvex(box, image)[1].reshape(-1, 2)
    pts = _pixels(to_corners(None, 190, 50, 40, 20, 0.4, W, H))
    clamped = np.clip(box, [0, 0], [W, H]).astype(np.float32)

    assert pts.min() >= 0 and (pts[:, 0] <= W).all() and (pts[:, 1] <= H).all()
    iou = BBoxOverlapCalculator.calculate_intersection_over_union
    assert iou(_trained_box(pts), visible) > iou(_trained_box(clamped), visible)


@pytest.mark.parametrize("cx, cy", [(-100, 50), (100, 500)])
def test_box_fully_outside_is_dropped(cx, cy):
    assert to_corners(None, cx, cy, 40, 20, 0.0, W, H) is None


def test_provenance_records_origins_of_exported_labels(tmp_path):
    import json

    import cv2
    import numpy as np

    from trainit.extract_yolo_training import ExtractionConfig, build_parser

    data = tmp_path / "in"
    data.mkdir()
    writer = cv2.VideoWriter(str(data / "clip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    for _ in range(3):
        writer.write(np.zeros((64, 64, 3), np.uint8))
    writer.release()

    def box(*annotators):
        return {"object_data": {"rbbox": [{"name": "shape", "val": [32, 32, 10, 10, 0]}],
                                "vec": [{"name": "annotator", "val": list(annotators)}]}}
    openlabel = {"metadata": {"schema_version": "1.1", "tagged_file": "clip.mp4"}, "ontologies": {},
                 "objects": {"1": {"name": "a", "type": "DynamicObject"}, "2": {"name": "b", "type": "DynamicObject"}},
                 "frames": {"0": {"objects": {"1": box("alice", "markit_yolo"),
                                              "2": box("markit_housekeeping(gap)", "markit_yolo")}},
                            "1": {"objects": {"1": box("markit_yolo")}}}}
    (data / "clip.json").write_text(json.dumps({"openlabel": openlabel}))
    chain = tmp_path / "prov.json"

    args = build_parser().parse_args(["-i", str(data), "-o", str(tmp_path / "out"), "--interval-frames", "2",
                                      "--provenance", str(chain)])
    YOLODatasetExtractor(ExtractionConfig(args)).run()

    activity = next(iter(json.loads(chain.read_text())["activity"].values()))
    # Frame 0 only (interval 2): one manual and one interpolated label
    assert activity["savant:origin_counts"] == {"interpolated": 1, "manual": 1}
    assert activity["su:share_reviewed"]["value"] == 0.5
    assert "--provenance" in activity["dataprov:arguments"]
