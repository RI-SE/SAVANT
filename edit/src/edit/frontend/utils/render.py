# edit/frontend/utils/render.py
from __future__ import annotations

from PyQt6.QtCore import QTimer

from edit.frontend.utils.frame_sync import _update_overlay_from_model
from edit.frontend.utils.settings_store import get_lock_to_center


def wire(main_window):
    """
    Hook overlay geometry to the video widget, and connect bbox events.
    No dependencies on methods inside MainWindow.
    """
    _sync_overlay_geometry(main_window)

    original_resize = getattr(main_window.video_widget, "resizeEvent", None)

    def _wrapped_resize(e):
        if callable(original_resize):
            original_resize(e)
        _sync_overlay_geometry(main_window)

    main_window.video_widget.resizeEvent = _wrapped_resize


def show_frame(main_window, pixmap, frame_idx: int | None):
    """
    Render a frame and update overlay from the model (rotated boxes, active objects).
    Safe if pixmap/frame_idx are None at end-of-video.
    """
    if pixmap is not None:
        main_window.video_widget.show_frame(pixmap)
    else:
        _clear_selection_for_frame_change(main_window, None)
        main_window.overlay.set_rotated_boxes([])
        return

    if frame_idx is not None and hasattr(main_window.seek_bar, "set_position"):
        main_window.seek_bar.set_position(int(frame_idx))

    # Save selection before clear so set_rotated_boxes can restore it
    overlay = getattr(main_window, "overlay", None)
    _prev_selected_id = overlay.selected_object_id() if overlay is not None else None
    _clear_selection_for_frame_change(main_window, frame_idx)
    if _prev_selected_id is not None and overlay is not None:
        overlay._preserve_selection_id = _prev_selected_id
    main_window._frame_updating = True
    try:
        _update_overlay_from_model(main_window)
    finally:
        main_window._frame_updating = False
    if get_lock_to_center():
        pan_fn = getattr(main_window, "pan_to_selected_bbox", None)
        if callable(pan_fn):
            QTimer.singleShot(0, pan_fn)
    if hasattr(main_window, "update_issue_info"):
        main_window.update_issue_info()


def refresh_frame(main_window):
    """
    Re-render current frame without changing index (after edits/zoom/etc.).
    No-op if no frame has been read yet (video not loaded or not yet navigated to).
    """
    idx = main_window.video_controller.current_index()
    if idx < 0:
        return
    pixmap, _ = main_window.video_controller.jump_to_frame(idx)
    show_frame(main_window, pixmap, idx)


def _sync_overlay_geometry(main_window):
    """Ensure overlay matches the video widget's rect and sits on top."""
    main_window.overlay.setGeometry(main_window.video_widget.rect())
    main_window.overlay.raise_()


def _clear_selection_for_frame_change(main_window, frame_idx: int | None) -> None:
    """Clear overlay/side selection when the rendered frame changes."""
    overlay = getattr(main_window, "overlay", None)
    if overlay is None:
        return

    new_index = None if frame_idx is None else int(frame_idx)
    last_index = getattr(main_window, "_last_rendered_frame_idx", None)
    last_index = None if last_index is None else int(last_index)

    if new_index is None:
        if last_index is not None:
            overlay.clear_selection()
        main_window._last_rendered_frame_idx = None
        return

    if last_index is None or last_index != new_index:
        overlay.clear_selection()
        # Invalidate per-frame delta bases so revisiting a frame starts fresh.
        if hasattr(main_window, "_delta_base"):
            main_window._delta_base.clear()
    main_window._last_rendered_frame_idx = new_index
