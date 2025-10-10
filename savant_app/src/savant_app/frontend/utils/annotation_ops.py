# savant_app/frontend/utils/annotation_ops.py
from dataclasses import asdict
from savant_app.frontend.exceptions import MissingObjectIDError
from savant_app.frontend.states.annotation_state import AnnotationMode, AnnotationState
from .render import refresh_frame
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMenu, QMessageBox


def wire(main_window):
    """
    Connect all annotation-related signals. Safe to call once in MainWindow.__init__.
    """
    if hasattr(main_window.sidebar, "start_bbox_drawing"):
        main_window.sidebar.start_bbox_drawing.connect(
            lambda object_type: on_new_object_bbox(main_window, object_type)
        )
    if hasattr(main_window.sidebar, "add_new_bbox_existing_obj"):
        main_window.sidebar.add_new_bbox_existing_obj.connect(
            lambda object_id: on_existing_object_bbox(main_window, object_id)
        )

    if hasattr(main_window.video_widget, "bbox_drawn"):
        # try:
        main_window.video_widget.bbox_drawn.connect(
            lambda ann: handle_drawn_bbox(main_window, ann)
        )
        # except TypeError:
        #    pass

    if hasattr(main_window.overlay, "deletePressed"):
        main_window.overlay.deletePressed.connect(
            lambda: delete_selected_bbox(main_window)
        )

    main_window.overlay.boxMoved.connect(lambda i, x, y: _moved(main_window, i, x, y))
    main_window.overlay.boxResized.connect(
        lambda i, x, y, w, h: _resized(main_window, i, x, y, w, h)
    )
    main_window.overlay.boxRotated.connect(lambda i, r: _rotated(main_window, i, r))
    _install_overlay_context_menu(main_window)


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

    idx = main_window.overlay.selected_index()
    if idx is None:
        return

    frame_key = main_window.video_controller.current_index()
    # try:
    object_key = main_window._overlay_ids[idx]
    # except Exception:
    #    return

    removed = main_window.annotation_controller.delete_bbox(
        frame_key=frame_key, object_key=object_key
    )
    if removed is None:
        return

    main_window._undo_stack.append(
        {
            "frame_key": frame_key,
            "object_key": object_key,
            "frame_obj": removed,
        }
    )

    main_window.overlay.clear_selection()
    refresh_frame(main_window)


def undo_delete(main_window):
    """Restore the last deletion (single bbox or batch cascade) and show a summary."""
    _ensure_undo_stack(main_window)
    if not main_window._undo_stack:
        return

    rec = main_window._undo_stack.pop()

    if rec.get("batch"):
        object_key = rec["object_key"]
        removed = rec["removed"]

        for frame_key, frame_obj in removed:
            main_window.annotation_controller.restore_bbox(
                frame_key=frame_key, object_key=object_key, frame_obj=frame_obj
            )

        frames_with_obj = sorted(int(fk) for fk, _ in removed)

        def _compress_ranges(sorted_indices):
            if not sorted_indices:
                return []
            ranges = []
            start = prev = sorted_indices[0]
            for i in sorted_indices[1:]:
                if i == prev + 1:
                    prev = i
                    continue
                ranges.append((start, prev))
                start = prev = i
            ranges.append((start, prev))
            return ranges

        ranges = _compress_ranges(frames_with_obj)
        ranges_str = ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)
        count_preview = len(frames_with_obj)

        msg = (f"Undo: restored {count_preview} bbox(es) for ID '{object_key}' "
               f"in frames {ranges_str}.")
        QMessageBox.information(main_window, "Undo Delete (Cascade)", msg)

        main_window.overlay.clear_selection()
        refresh_frame(main_window)
        return

    frame_key = rec["frame_key"]
    object_key = rec["object_key"]
    frame_obj = rec["frame_obj"]

    main_window.annotation_controller.restore_bbox(
        frame_key=frame_key, object_key=object_key, frame_obj=frame_obj
    )

    QMessageBox.information(
        main_window, "Undo Delete",
        f"Undo: restored bbox for ID '{object_key}' in frame {int(frame_key)}."
    )

    main_window.overlay.clear_selection()
    refresh_frame(main_window)


def _moved(main_window, overlay_idx: int, x: float, y: float):
    fk = main_window.video_controller.current_index()
    ok = main_window._overlay_ids[overlay_idx]
    main_window.annotation_controller.move_resize_bbox(
        frame_key=fk, object_key=ok, x_center=x, y_center=y
    )
    refresh_frame(main_window)


def _resized(main_window, overlay_idx: int, x: float, y: float, w: float, h: float):
    fk = main_window.video_controller.current_index()
    ok = main_window._overlay_ids[overlay_idx]
    main_window.annotation_controller.move_resize_bbox(
        frame_key=fk, object_key=ok, x_center=x, y_center=y, width=w, height=h
    )
    refresh_frame(main_window)


def _rotated(main_window, overlay_idx: int, rotation: float):
    fk = main_window.video_controller.current_index()
    ok = main_window.project_state_controller.object_id_for_frame_index(fk, overlay_idx)
    main_window.annotation_controller.move_resize_bbox(
        frame_key=fk, object_key=ok, rotation=rotation
    )
    refresh_frame(main_window)


def _ensure_undo_stack(main_window):
    if not hasattr(main_window, "_undo_stack"):
        main_window._undo_stack = []


def _install_overlay_context_menu(main_window):
    """Attach a custom right-click (context) menu to the video overlay."""
    overlay_widget = main_window.overlay
    overlay_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    overlay_widget.customContextMenuRequested.connect(
        lambda click_position: _on_overlay_context_menu(main_window, click_position)
    )


def _on_overlay_context_menu(main_window, click_position):
    """Handle right-clicks on the overlay and show a context menu for bbox actions."""
    overlay_widget = main_window.overlay
    bbox_index, hit_mode = overlay_widget._hit_test(click_position)

    if bbox_index is None:
        return

    overlay_widget._selected_idx = bbox_index
    overlay_widget.update()

    context_menu = QMenu(overlay_widget)
    action_delete_single = context_menu.addAction("Delete this bbox")
    action_delete_cascade = context_menu.addAction("Cascade delete all with this ID")

    selected_action = context_menu.exec(overlay_widget.mapToGlobal(click_position))
    if selected_action is None:
        return

    if selected_action == action_delete_single:
        delete_selected_bbox(main_window)
    elif selected_action == action_delete_cascade:
        try:
            _cascade_delete_same_id(main_window, bbox_index)
        except MissingObjectIDError as e:
            QMessageBox.warning(main_window, "Cascade Delete", str(e))


def _cascade_delete_same_id(main_window, overlay_bbox_index: int):
    """Delete all bboxes across all frames with the same object ID as the clicked bbox."""
    try:
        object_id = main_window._overlay_ids[overlay_bbox_index]
    except Exception as e:
        raise MissingObjectIDError(
            "Could not determine object ID for the selected bounding box.") from e

    openlabel_annotation = (
        main_window.annotation_controller.annotation_service.project_state.annotation_config
    )
    if not openlabel_annotation:
        return

    frames_with_object = sorted(
        int(frame_key)
        for frame_key, frame_data in getattr(openlabel_annotation, "frames", {}).items()
        if getattr(frame_data, "objects", None) and object_id in frame_data.objects
    )
    if not frames_with_object:
        return

    def _compress_frame_ranges(frame_indices):
        if not frame_indices:
            return []
        compressed_ranges = []
        start = previous = frame_indices[0]
        for frame_number in frame_indices[1:]:
            if frame_number == previous + 1:
                previous = frame_number
                continue
            compressed_ranges.append((start, previous))
            start = previous = frame_number
        compressed_ranges.append((start, previous))
        return compressed_ranges

    frame_ranges = _compress_frame_ranges(frames_with_object)
    frame_ranges_str = ", ".join(
        str(start) if start == end else f"{start}-{end}"
        for start, end in frame_ranges
    )
    total_bboxes = len(frames_with_object)

    user_choice = QMessageBox.question(
        main_window,
        "Cascade Delete",
        f"Delete all {total_bboxes} bboxes for ID '{object_id}' "
        f"across frames {frame_ranges_str}?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    if user_choice != QMessageBox.StandardButton.Yes:
        return

    deleted_bboxes = main_window.annotation_controller.delete_bboxes_by_object(object_id)
    if not deleted_bboxes:
        return

    _ensure_undo_stack(main_window)
    main_window._undo_stack.append({
        "batch": True,
        "object_key": object_id,
        "removed": deleted_bboxes,
    })

    main_window.overlay.clear_selection()
    refresh_frame(main_window)
