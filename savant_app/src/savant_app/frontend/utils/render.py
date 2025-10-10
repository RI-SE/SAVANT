# savant_app/frontend/utils/render.py
from __future__ import annotations
from savant_app.frontend.types import BBoxData


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
        main_window.overlay.set_rotated_boxes([])
        return

    if frame_idx is not None and hasattr(main_window.seek_bar, "set_position"):
        main_window.seek_bar.set_position(int(frame_idx))

    _update_overlay_from_model(main_window)


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

        frame_bounding_boxes_frontend_data = []
        for frame_bounding_box in frame_bounding_boxes:
            # Map backend FrameBBox to frontend BBox
            frame_bounding_boxes_frontend_data.append(
                BBoxData(
                    object_id=frame_bounding_box.object_id,
                    object_type=frame_bounding_box.object_type,
                    center_x=frame_bounding_box.bbox.cx,
                    center_y=frame_bounding_box.bbox.cy,
                    width=frame_bounding_box.bbox.width,
                    height=frame_bounding_box.bbox.height,
                    theta=frame_bounding_box.bbox.theta,
                )
            )

        # Update overlay dimensions and set bounding boxes
        video_width, video_height = main_window.video_controller.size()
        main_window.overlay.set_frame_size(video_width, video_height)
        main_window.overlay.set_rotated_boxes(frame_bounding_boxes_frontend_data)

        # Refresh sidebar with active objects
        active_objects = main_window.annotation_controller.get_active_objects(
            current_frame_index
        )
        main_window.sidebar.refresh_active_objects(active_objects)
        main_window.sidebar._refresh_active_frame_tags(current_frame_index)

    except Exception:
        main_window.overlay.set_rotated_boxes([])
        raise
