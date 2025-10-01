# savant_app/frontend/utils/render.py
from __future__ import annotations
from typing import Any, List, Tuple
from PyQt6.QtWidgets import QMessageBox


def wire(mw):
    """
    Hook overlay geometry to the video widget, and connect bbox events.
    No dependencies on methods inside MainWindow.
    """
    _sync_overlay_geometry(mw)

    original_resize = getattr(mw.video_widget, "resizeEvent", None)

    def _wrapped_resize(e):
        if callable(original_resize):
            original_resize(e)
        _sync_overlay_geometry(mw)

    mw.video_widget.resizeEvent = _wrapped_resize


def show_frame(mw, pixmap, frame_idx: int | None):
    """
    Render a frame and update overlay from the model (rotated boxes, active objects).
    Safe if pixmap/frame_idx are None at end-of-video.
    """
    if pixmap is not None:
        mw.video_widget.show_frame(pixmap)
    else:
        mw.overlay.set_rotated_boxes([])
        return

    if frame_idx is not None and hasattr(mw.seek_bar, "set_position"):
        mw.seek_bar.set_position(int(frame_idx))

    _update_overlay_from_model(mw)


def refresh_frame(mw):
    """
    Re-render current frame without changing index (after edits/zoom/etc.).
    """
    try:
        idx = mw.video_controller.current_index()
        pixmap, _ = mw.video_controller.jump_to_frame(idx)
        show_frame(mw, pixmap, idx)
    except Exception as e:
        QMessageBox.critical(mw, "Refresh failed", str(e))


def _sync_overlay_geometry(mw):
    """Ensure overlay matches the video widget's rect and sits on top."""
    mw.overlay.setGeometry(mw.video_widget.rect())
    mw.overlay.raise_()


def _update_overlay_from_model(mw):
    """Fetch boxes for current frame and update overlay + sidebar."""
    frame_idx = mw.video_controller.current_index()
    try:
        pairs: List[Tuple[str, Any]] = mw.project_state_controller.boxes_with_ids_for_frame(
            frame_idx)
        mw._overlay_ids = [oid for (oid, _) in pairs]
        boxes = [geom for (_, geom) in pairs]
        w, h = mw.video_controller.size()
        mw.overlay.set_frame_size(w, h)
        mw.overlay.set_rotated_boxes(boxes)
        active = mw.annotation_controller.get_active_objects(frame_idx)
        mw.sidebar.refresh_active_objects(active)

    except Exception:
        mw._overlay_ids = []
        mw.overlay.set_rotated_boxes([])
