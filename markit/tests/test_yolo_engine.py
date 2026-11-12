"""YOLOEngine calls Ultralytics with markit's own tracker, not the library default."""

from unittest.mock import MagicMock

import numpy as np
from markit.markitlib.processing import engines
from markit.markitlib.processing.engines import YOLO_TRACKER, YOLOEngine


def _engine(monkeypatch):
    model = MagicMock()
    model.track.return_value = []
    monkeypatch.setattr(engines, "YOLO", lambda _weights: model)
    return YOLOEngine("weights.pt"), model


def test_track_uses_botsort(monkeypatch):
    engine, model = _engine(monkeypatch)
    engine.process_frame(np.zeros((8, 8, 3), dtype=np.uint8))
    assert YOLO_TRACKER == "botsort.yaml"
    assert model.track.call_args.kwargs["tracker"] == YOLO_TRACKER


def test_tracker_in_model_info(monkeypatch):
    engine, _ = _engine(monkeypatch)
    assert engine.get_model_info()["yolo_tracker"] == YOLO_TRACKER
