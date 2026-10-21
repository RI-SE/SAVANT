"""edit records how each box came about (per-box origin, aligned with the annotator history)."""

import shutil
from pathlib import Path

from src.edit.services.annotation_service import AnnotationService
from src.edit.services.project_state import ProjectState

ASSET = Path(__file__).parent.parent.parent / "assets" / "Kraklanda_short.json"


def _service(tmp_path):
    target = tmp_path / ASSET.name
    shutil.copy2(ASSET, target)
    state = ProjectState()
    state.load_openlabel_config(str(target))
    return AnnotationService(state), state.annotation_config


def _vec(openlabel, frame, object_id):
    entries = openlabel.frames[str(frame)].objects[object_id].object_data.vec
    return {e.name: list(e.val) for e in entries}


def _first_box(openlabel):
    """First box, given a detector history as markit writes it (the fixture has confidence only)."""
    from collections import deque

    from src.edit.models.OpenLabel import AnnotatorData, ConfidenceData

    frame = sorted(openlabel.frames, key=int)[0]
    object_id = next(iter(openlabel.frames[frame].objects))
    openlabel.frames[frame].objects[object_id].object_data.vec = [
        AnnotatorData(val=deque(["markit_yolo"])), ConfidenceData(val=deque([0.42]))]
    return int(frame), object_id


def test_drawn_box_is_manual(tmp_path):
    _, openlabel = _service(tmp_path)
    openlabel.append_object_bbox(frame_id=999, bbox_coordinates=[10, 10, 5, 5, 0], obj_id="9", annotator="alice")
    assert _vec(openlabel, 999, "9")["origin"] == ["manual"]


def test_editing_a_detected_box_prepends_manual(tmp_path):
    service, openlabel = _service(tmp_path)
    frame, object_id = _first_box(openlabel)
    service.move_resize_bbox(frame, object_id, delta_x=2.0, annotator="alice")
    vec = _vec(openlabel, frame, object_id)
    assert vec["annotator"][0] == "alice"
    assert vec["origin"][0] == "manual"
    assert len(vec["origin"]) == len(vec["annotator"])


def test_review_after_own_edit_stays_manual(tmp_path):
    service, openlabel = _service(tmp_path)
    frame, object_id = _first_box(openlabel)
    service.move_resize_bbox(frame, object_id, delta_x=2.0, annotator="alice")
    service.mark_confidence_resolved(frame, object_id, "alice")
    assert _vec(openlabel, frame, object_id)["origin"][0] == "manual"
    service.mark_confidence_resolved(frame, object_id, "bob")
    assert _vec(openlabel, frame, object_id)["origin"][:2] == ["reviewed", "manual"]


def test_interpolated_boxes_are_manual_interpolated(tmp_path):
    service, openlabel = _service(tmp_path)
    object_id = next(iter(openlabel.objects))
    frames = sorted(int(f) for f, fo in openlabel.frames.items() if object_id in fo.objects)
    start, end = frames[0], frames[-1]
    # Clear the frames in between so interpolation fills them
    for f in range(start + 1, end):
        openlabel.frames.get(str(f)) and openlabel.frames[str(f)].objects.pop(object_id, None)
    if end - start < 2:
        return
    service.interpolate_annotations(object_id, start, end, "alice")
    assert _vec(openlabel, start + 1, object_id)["origin"] == ["manual_interpolated"]
