# savant_app/frontend/utils/render.py
from __future__ import annotations

from savant_app.frontend.types import BBoxData, ConfidenceFlagMap


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

    _clear_selection_for_frame_change(main_window, frame_idx)
    _update_overlay_from_model(main_window)
    if hasattr(main_window, "update_issue_info"):
        main_window.update_issue_info()


def refresh_frame(main_window):
    """
    Re-render current frame without changing index (after edits/zoom/etc.).
    """
    idx = main_window.video_controller.current_index()
    pixmap, _ = main_window.video_controller.jump_to_frame(idx)
    show_frame(main_window, pixmap, idx)


def _sync_overlay_geometry(main_window):
    """Ensure overlay matches the video widget's rect and sits on top."""
    main_window.overlay.setGeometry(main_window.video_widget.rect())
    main_window.overlay.raise_()


# TODO: This creates high coupling between render and annotation_ops.
def _update_overlay_from_model(main_window):
    """Fetch boxes for current frame and update overlay + sidebar."""
    current_frame_index = main_window.video_controller.current_index()
    try:
        # Retrieve FrameBBox objects from backend
        frame_bounding_boxes = (
            main_window.project_state_controller.boxes_with_ids_for_frame(
                current_frame_index
            )
        )

        frame_bounding_boxes_frontend_data = [
            BBoxData(
                object_id=fbbox.object_id,
                object_type=fbbox.object_type,
                center_x=fbbox.bbox.cx,
                center_y=fbbox.bbox.cy,
                width=fbbox.bbox.width,
                height=fbbox.bbox.height,
                theta=fbbox.bbox.theta,
            )
            for fbbox in frame_bounding_boxes
        ]

        # Update overlay dimensions and set bounding boxes
        video_width, video_height = (
            main_window.project_state_controller.get_video_size()
        )
        main_window.overlay.set_frame_size(video_width, video_height)
        main_window.overlay.set_rotated_boxes(frame_bounding_boxes_frontend_data)
        frame_issues_map = main_window.state.confidence_issues()
        frame_issues = frame_issues_map.get(current_frame_index, [])
        flags: ConfidenceFlagMap = {}
        for issue in frame_issues:
            object_id = getattr(issue, "object_id", None)
            severity = getattr(issue, "severity", None)
            if not object_id or severity not in ("warning", "error"):
                continue
            if severity == "error":
                flags[object_id] = "error"
            elif severity == "warning" and object_id not in flags:
                flags[object_id] = "warning"
        main_window.overlay.set_confidence_flags(flags)

        # Refresh sidebar with active objects
        active_objects = main_window.annotation_controller.get_active_objects(
            current_frame_index
        )
        main_window.sidebar.refresh_active_objects(active_objects, flags)
        main_window.sidebar._refresh_active_frame_tags(current_frame_index)
        main_window.sidebar.refresh_confidence_issue_list(current_frame_index)

    except Exception:
        main_window.overlay.set_rotated_boxes([])
        raise


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
    main_window._last_rendered_frame_idx = new_index
