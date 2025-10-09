# savant_app/frontend/utils/annotation_ops.py
from dataclasses import asdict
from savant_app.frontend.states.annotation_state import AnnotationMode, AnnotationState
from .render import refresh_frame


def wire(main_window):
    """
    Connect all annotation-related signals. Safe to call once in MainWindow.__init__.
    """

    # TODO: Reverse naming of signal and function here.
    if hasattr(main_window.sidebar, "start_bbox_drawing"):
        main_window.sidebar.start_bbox_drawing.connect(
            lambda object_type: on_new_object_bbox(main_window, object_type)
        )
    
    # TODO: Reverse naming of signal and function here.
    if hasattr(main_window.sidebar, "add_new_bbox_existing_obj"):
        main_window.sidebar.add_new_bbox_existing_obj.connect(
            lambda object_id: on_existing_object_bbox(main_window, object_id)
        )

    if hasattr(main_window.sidebar, "highlight_selected_object"):
        main_window.sidebar.highlight_selected_object.connect(
            lambda object_id: highlight_selected_object(main_window, object_id)
        )

    if hasattr(main_window.video_widget, "bbox_drawn"):
        main_window.video_widget.bbox_drawn.connect(
            lambda ann: handle_drawn_bbox(main_window, ann)
        )
    

    main_window.overlay.boxMoved.connect(lambda id, x, y: _moved(main_window, id, x, y))
    main_window.overlay.boxResized.connect(
        lambda id, x, y, w, h: _resized(main_window, id, x, y, w, h)
    )
    main_window.overlay.boxRotated.connect(lambda id, r: _rotated(main_window, id, r))

def highlight_selected_object(main_window, object_id: str):
    """Highlight the selected object in the overlay."""
    main_window.overlay.select_box_by_obj_id(object_id)

def on_new_object_bbox(main_window, object_type: str):
    """Enter drawing mode for a NEW object of given type."""
    main_window.video_widget.start_drawing_mode(
        AnnotationState(mode=AnnotationMode.NEW, object_type=object_type)
    )


def on_existing_object_bbox(main_window, object_id: str):
    """Enter drawing mode to add a bbox to an EXISTING object id."""
    main_window.video_widget.start_drawing_mode(
        AnnotationState(mode=AnnotationMode.EXISTING, object_id=object_id)
    )


def handle_drawn_bbox(main_window, annotation: AnnotationState):
    """Finalize newly drawn bbox → controller → refresh."""
    frame_idx = main_window.video_controller.current_index()
    if annotation.mode == AnnotationMode.EXISTING:
        if not annotation.object_id:
            return
        main_window.annotation_controller.create_bbox_existing_object(
            frame_number=frame_idx, bbox_info=asdict(annotation)
        )
    elif annotation.mode == AnnotationMode.NEW:
        main_window.annotation_controller.create_new_object_bbox(
            frame_number=frame_idx, bbox_info=asdict(annotation)
        )
    refresh_frame(main_window)


def delete_selected_bbox(main_window):
    """Delete the currently selected bbox and record it for undo."""
    _ensure_undo_stack(main_window)

    # Get selected object ID directly from overlay
    object_id = main_window.overlay.selected_object_id()
    if object_id is None:
        return

    frame_key = main_window.video_controller.current_index()
    
    removed = main_window.annotation_controller.delete_bbox(
        frame_key=frame_key, object_key=object_id
    )
    if removed is None:
        return

    main_window._undo_stack.append(
        {
            "frame_key": frame_key,
            "object_key": object_id,
            "frame_obj": removed,
        }
    )

    main_window.overlay.clear_selection()
    refresh_frame(main_window)


def undo_delete(main_window):
    """Restore the last deleted bbox, if any."""
    _ensure_undo_stack(main_window)
    if not main_window._undo_stack:
        return

    rec = main_window._undo_stack.pop()
    frame_key = rec["frame_key"]
    object_key = rec["object_key"]
    frame_obj = rec["frame_obj"]

    main_window.annotation_controller.restore_bbox(
        frame_key=frame_key, object_key=object_key, frame_obj=frame_obj
    )
    main_window.overlay.clear_selection()
    refresh_frame(main_window)


def _moved(main_window, object_id: str, x: float, y: float):
    fk = main_window.video_controller.current_index()
    main_window.annotation_controller.move_resize_bbox(
        frame_key=fk, object_key=object_id, x_center=x, y_center=y
    )
    refresh_frame(main_window)


def _resized(main_window, object_id: str, x: float, y: float, w: float, h: float):
    fk = main_window.video_controller.current_index()
    main_window.annotation_controller.move_resize_bbox(
        frame_key=fk, object_key=object_id, x_center=x, y_center=y, width=w, height=h
    )
    refresh_frame(main_window)


def _rotated(main_window, object_id: str, rotation: float):
    fk = main_window.video_controller.current_index()
    main_window.annotation_controller.move_resize_bbox(
        frame_key=fk, object_key=object_id, rotation=rotation
    )
    refresh_frame(main_window)


def _ensure_undo_stack(main_window):
    if not hasattr(main_window, "_undo_stack"):
        main_window._undo_stack = []
