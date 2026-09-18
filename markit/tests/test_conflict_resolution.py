"""
Unit tests for DetectionConflictResolver's structured decision log.
"""

import numpy as np

from markit.markitlib import DetectionResult
from markit.markitlib.config import ConflictResolutionConfig
from markit.markitlib.processing import DetectionConflictResolver


def _make_detection(object_id, source_engine, bbox):
    return DetectionResult(
        object_id=object_id,
        class_id=1,
        confidence=0.9,
        oriented_bbox=bbox,
        center=(150.0, 125.0),
        angle=0.0,
        source_engine=source_engine,
        width=100.0,
        height=50.0,
    )


class TestDetectionConflictResolverDecisionLog:
    """Tests for DetectionConflictResolver.get_decision_log()."""

    def test_no_conflict_no_decision_log(self, sample_obb_bbox):
        """Non-overlapping detections produce no decision log entries."""
        far_bbox = sample_obb_bbox + 1000
        resolver = DetectionConflictResolver(ConflictResolutionConfig(iou_threshold=0.3))
        results = [
            _make_detection("obj_yolo_1", "yolo", sample_obb_bbox),
            _make_detection("obj_oflow_1", "optical_flow", far_bbox),
        ]

        resolver.resolve_conflicts(results, frame_idx=5)

        assert resolver.get_decision_log() == []

    def test_conflict_recorded_in_decision_log(self, sample_obb_bbox):
        """A dropped optical-flow detection is recorded with IoU and engines."""
        resolver = DetectionConflictResolver(ConflictResolutionConfig(iou_threshold=0.3))
        results = [
            _make_detection("obj_yolo_1", "yolo", sample_obb_bbox),
            _make_detection("obj_oflow_1", "optical_flow", sample_obb_bbox),
        ]

        resolver.resolve_conflicts(results, frame_idx=5)
        log = resolver.get_decision_log()

        assert len(log) == 1
        record = log[0]
        assert record["stage"] == "detection"
        assert record["source"] == "conflict_resolution"
        assert record["action"] == "drop_detection"
        assert record["object_id"] == "obj_oflow_1"
        assert record["frame"] == 5
        assert record["details"]["winning_object_id"] == "obj_yolo_1"
        assert record["details"]["winning_engine"] == "yolo"
        assert record["details"]["losing_engine"] == "optical_flow"
        assert np.isclose(record["details"]["iou"], 1.0)
        assert record["details"]["iou_threshold"] == 0.3

    def test_decision_log_recorded_without_verbose_logging(self, sample_obb_bbox):
        """Structured records are collected even when enable_logging is off."""
        resolver = DetectionConflictResolver(
            ConflictResolutionConfig(iou_threshold=0.3, enable_logging=False)
        )
        results = [
            _make_detection("obj_yolo_1", "yolo", sample_obb_bbox),
            _make_detection("obj_oflow_1", "optical_flow", sample_obb_bbox),
        ]

        resolver.resolve_conflicts(results, frame_idx=5)

        assert len(resolver.get_decision_log()) == 1
