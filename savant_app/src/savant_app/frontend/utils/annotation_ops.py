# savant_app/frontend/utils/annotation_ops.py
from dataclasses import asdict
from savant_app.frontend.states.annotation_state import AnnotationMode, AnnotationState
from .render import refresh_frame


def wire(mw):
    """
    Connect all annotation-related signals. Safe to call once in MainWindow.__init__.
    """
    if hasattr(mw.sidebar, "start_bbox_drawing"):
        mw.sidebar.start_bbox_drawing.connect(
            lambda object_type: on_new_object_bbox(mw, object_type))
    if hasattr(mw.sidebar, "add_new_bbox_existing_obj"):
        mw.sidebar.add_new_bbox_existing_obj.connect(
            lambda object_id: on_existing_object_bbox(mw, object_id))

    if hasattr(mw.video_widget, "bbox_drawn"):
        try:
            mw.video_widget.bbox_drawn.connect(lambda ann: handle_drawn_bbox(mw, ann))
        except TypeError:
            pass

    mw.overlay.boxMoved.connect(lambda i, x, y: _moved(mw, i, x, y))
    mw.overlay.boxResized.connect(lambda i, x, y, w, h: _resized(mw, i, x, y, w, h))
    mw.overlay.boxRotated.connect(lambda i, r: _rotated(mw, i, r))


def on_new_object_bbox(mw, object_type: str):
    """Enter drawing mode for a NEW object of given type."""
    mw.video_widget.start_drawing_mode(
        AnnotationState(mode=AnnotationMode.NEW, object_type=object_type))


def on_existing_object_bbox(mw, object_id: str):
    """Enter drawing mode to add a bbox to an EXISTING object id."""
    mw.video_widget.start_drawing_mode(
        AnnotationState(mode=AnnotationMode.EXISTING, object_id=object_id))


def handle_drawn_bbox(mw, annotation: AnnotationState):
    """Finalize newly drawn bbox → controller → refresh."""
    frame_idx = mw.video_controller.current_index()
    if annotation.mode == AnnotationMode.EXISTING:
        if not annotation.object_id:
            return
        mw.annotation_controller.create_bbox_existing_object(
            frame_number=frame_idx, bbox_info=asdict(annotation)
        )
    elif annotation.mode == AnnotationMode.NEW:
        mw.annotation_controller.create_new_object_bbox(
            frame_number=frame_idx, bbox_info=asdict(annotation)
        )
    refresh_frame(mw)


def delete_selected_bbox(mw):
    """Delete the currently selected bbox and record it for undo."""
    _ensure_undo_stack(mw)

    idx = mw.overlay.selected_index()
    if idx is None:
        return

    frame_key = mw.video_controller.current_index()
    try:
        object_key = mw._overlay_ids[idx]
    except Exception:
        return

    removed = mw.annotation_controller.delete_bbox(frame_key=frame_key, object_key=object_key)
    if removed is None:
        return

    mw._undo_stack.append(
        {
            "frame_key": frame_key,
            "object_key": object_key,
            "frame_obj": removed,
        }
    )

    mw.overlay.clear_selection()
    refresh_frame(mw)


def undo_delete(mw):
    """Restore the last deleted bbox, if any."""
    _ensure_undo_stack(mw)
    if not mw._undo_stack:
        return

    rec = mw._undo_stack.pop()
    frame_key = rec["frame_key"]
    object_key = rec["object_key"]
    frame_obj = rec["frame_obj"]

    mw.annotation_controller.restore_bbox(
        frame_key=frame_key, object_key=object_key, frame_obj=frame_obj)
    mw.overlay.clear_selection()
    refresh_frame(mw)


def _moved(mw, overlay_idx: int, x: float, y: float):
    fk = mw.video_controller.current_index()
    ok = mw._overlay_ids[overlay_idx]
    mw.annotation_controller.move_resize_bbox(
        frame_key=fk, object_key=ok, x_center=x, y_center=y
    )
    refresh_frame(mw)


def _resized(mw, overlay_idx: int, x: float, y: float, w: float, h: float):
    fk = mw.video_controller.current_index()
    ok = mw._overlay_ids[overlay_idx]
    mw.annotation_controller.move_resize_bbox(
        frame_key=fk, object_key=ok, x_center=x, y_center=y, width=w, height=h
    )
    refresh_frame(mw)


def _rotated(mw, overlay_idx: int, rotation: float):
    fk = mw.video_controller.current_index()
    ok = mw.project_state_controller.object_id_for_frame_index(fk, overlay_idx)
    mw.annotation_controller.move_resize_bbox(
        frame_key=fk, object_key=ok, rotation=rotation
    )
    refresh_frame(mw)


def _ensure_undo_stack(mw):
    if not hasattr(mw, "_undo_stack"):
        mw._undo_stack = []
