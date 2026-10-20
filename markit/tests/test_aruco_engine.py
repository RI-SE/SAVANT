"""Tests for ArUcoEngine detections on a synthetic marker."""

import cv2
import numpy as np

from markit.markitlib.processing.engines import ArUcoEngine


def test_detection_reports_marker_size(tmp_path):
    csv_path = tmp_path / "Test_markers.csv"
    csv_path.write_text("point_name,latitude,longitude,altitude\naruco_3a,57.0,12.0,10.0\n")
    side = 120
    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 3, side
    )
    frame = np.full((400, 400), 255, np.uint8)
    frame[140:140 + side, 140:140 + side] = marker
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    detections = ArUcoEngine(str(csv_path), class_id=0).process_frame(frame)

    assert len(detections) == 1
    assert abs(detections[0].width - side) < 5
    assert abs(detections[0].height - side) < 5
