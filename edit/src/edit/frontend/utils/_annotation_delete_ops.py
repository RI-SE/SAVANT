# edit/frontend/utils/_annotation_delete_ops.py
from __future__ import annotations

from PyQt6.QtWidgets import QMessageBox

from edit.frontend.exceptions import MissingObjectIDError
from edit.frontend.utils.undo import CompositeCommand, DeleteBBoxCommand

from ._annotation_helpers import _refresh_after_annotation_change


def _compress_frame_ranges(frame_indices):
    """Compress a sorted list of frame indices into (start, end) range tuples."""
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


def _cascade_delete_directional(main_window, object_id: str, direction: str):
    """Delete all bboxes for object_id from the current frame forward or backward.

    Args:
        direction: "forward" (current frame … last) or "backward" (first … current frame)
    """
    openlabel_annotation = (
        main_window.annotation_controller.annotation_service.project_state.annotation_config
    )
    if not openlabel_annotation:
        return

    current_frame = int(main_window.video_controller.current_index())

    all_frames = sorted(
        int(frame_key)
        for frame_key, frame_data in getattr(openlabel_annotation, "frames", {}).items()
        if getattr(frame_data, "objects", None) and object_id in frame_data.objects
    )

    if direction == "forward":
        frames_to_delete = [f for f in all_frames if f >= current_frame]
        label = f"from frame {current_frame} forward"
    else:
        frames_to_delete = [f for f in all_frames if f <= current_frame]
        label = f"from frame {current_frame} backward"

    if not frames_to_delete:
        QMessageBox.information(
            main_window, "Cascade Delete", f"No bboxes found {label} for ID '{object_id}'."
        )
        return

    user_choice = QMessageBox.question(
        main_window,
        "Cascade Delete",
        f"Delete {len(frames_to_delete)} bbox(es) for ID '{object_id}' {label}?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    if user_choice != QMessageBox.StandardButton.Yes:
        return

    delete_commands = [
        DeleteBBoxCommand(frame_number=f, object_id=str(object_id))
        for f in frames_to_delete
    ]
    batch = CompositeCommand(
        description=f"Cascade delete {len(delete_commands)} bboxes ({label})",
        commands=delete_commands,
    )
    main_window.execute_undoable_command(batch)
    QMessageBox.information(
        main_window,
        "Cascade Delete",
        f"Deleted {len(frames_to_delete)} bbox(es) for ID '{object_id}' {label}.",
    )
    main_window.overlay.clear_selection()
    _refresh_after_annotation_change(main_window)


def _cascade_delete_same_id(main_window, overlay_bbox_index: int):
    """Delete all bboxes across all frames with the same object ID as the clicked bbox."""
    try:
        main_window.overlay._selected_idx = overlay_bbox_index
        object_id = main_window.overlay.selected_object_id()
    except Exception as e:
        raise MissingObjectIDError(
            "Could not determine object ID for the selected bounding box."
        ) from e

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

    frame_ranges = _compress_frame_ranges(frames_with_object)
    frame_ranges_str = ", ".join(
        str(start) if start == end else f"{start}-{end}" for start, end in frame_ranges
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

    delete_commands = [
        DeleteBBoxCommand(frame_number=frame_number, object_id=str(object_id))
        for frame_number in frames_with_object
    ]
    if not delete_commands:
        QMessageBox.information(
            main_window, "Cascade Delete", "No bounding boxes were deleted."
        )
        return
    batch_command = CompositeCommand(
        description=f"Cascade delete {len(delete_commands)} bounding boxes",
        commands=delete_commands,
    )
    main_window.execute_undoable_command(batch_command)
    QMessageBox.information(
        main_window,
        "Cascade Delete",
        f"Deleted {total_bboxes} bbox(es) for ID '{object_id}' across frames {frame_ranges_str}.",
    )
    main_window.overlay.clear_selection()
    _refresh_after_annotation_change(main_window)
