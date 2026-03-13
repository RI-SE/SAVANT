# edit/frontend/utils/_annotation_helpers.py
from __future__ import annotations

import math as _math

from PyQt6.QtWidgets import QMessageBox

from edit.frontend.utils.undo import UpdateBBoxGeometryCommand
from edit.services.exceptions import VideoLoadError

from .render import refresh_frame


def _refresh_after_bbox_update(main_window):
    """Refresh confidence markers and the current frame after bbox changes."""
    refresh_conf = getattr(main_window, "refresh_confidence_issues", None)
    try:
        if callable(refresh_conf):
            refresh_conf()
        else:
            refresh_frame(main_window)
    except VideoLoadError:
        return


def _refresh_after_annotation_change(main_window):
    """Refresh UI elements impacted by annotation updates."""
    _refresh_after_bbox_update(main_window)
    sidebar = getattr(main_window, "sidebar", None)
    if sidebar is None:
        return
    current_index = int(main_window.video_controller.current_index())
    sidebar._refresh_active_frame_tags(current_index)
    if sidebar._selected_annotation_object_id:
        sidebar._refresh_relationships(sidebar._selected_annotation_object_id)
    refresh_confidence_list = getattr(sidebar, "refresh_confidence_issue_list", None)
    if callable(refresh_confidence_list):
        refresh_confidence_list(current_index)


def _apply_geometry_update(
    main_window,
    object_id: str,
    annotator: str,
    snapshot_builder,
) -> None:
    frame_number = int(main_window.video_controller.current_index())
    gateway = main_window.undo_context.annotation_gateway
    before_snapshot = gateway.capture_geometry(frame_number, object_id)
    after_snapshot = snapshot_builder(before_snapshot)
    command = UpdateBBoxGeometryCommand(
        frame_number=frame_number,
        object_id=object_id,
        before=before_snapshot,
        after=after_snapshot,
        annotator=annotator,
    )
    main_window.execute_undoable_command(command)

    # Store the compound delta for the "repeat last adjustment" shortcut (R).
    # Use a base snapshot anchored to the first edit of this object in this frame,
    # so that multiple edits (move, rotate, resize) accumulate into one replayable delta.
    if hasattr(main_window, "last_bbox_deltas"):
        base_key = (frame_number, object_id)
        delta_base = getattr(main_window, "_delta_base", {})
        if base_key not in delta_base:
            delta_base[base_key] = before_snapshot
            main_window._delta_base = delta_base
        base = delta_base[base_key]
        dcx = after_snapshot.center_x - base.center_x
        dcy = after_snapshot.center_y - base.center_y
        dw = after_snapshot.width - base.width
        dh = after_snapshot.height - base.height
        dtheta = ((after_snapshot.rotation - base.rotation + _math.pi) % (2 * _math.pi)) - _math.pi
        main_window.last_bbox_deltas[object_id] = (dcx, dcy, dw, dh, dtheta)

    _refresh_after_annotation_change(main_window)


def _frames_to_ranges(frames: list[int]) -> str:
    """Convert a list of frame numbers into contiguous ranges as a string."""
    if not frames:
        return ""
    ranges = []
    start = prev = frames[0]
    for f in frames[1:]:
        if f == prev + 1:
            prev = f
        else:
            ranges.append((start, prev))
            start = prev = f
    ranges.append((start, prev))
    range_strs = [f"{s}-{e}" if s != e else f"{s}" for s, e in ranges]
    return ", ".join(range_strs)


def _cascade_property_description(center_x, center_y, width, height, rotation) -> str:
    """Build a human-readable list of properties being cascaded."""
    parts = []
    if center_x is not None or center_y is not None:
        parts.append("position")
    if width is not None or height is not None:
        parts.append("size")
    if rotation is not None:
        parts.append("rotation")
    return ", ".join(parts) or "properties"


def _confirm_cascade(main_window, object_id, property_desc, frame_range_str) -> bool:
    """Show a confirmation dialog before executing a cascade operation."""
    result = QMessageBox.warning(
        main_window,
        "Cascade Operation",
        f"This will overwrite the <b>{property_desc}</b> of object "
        f"'{object_id}' on frames {frame_range_str} with values from "
        f"the current frame.\n\n"
        f"Existing per-frame adjustments (e.g. from tracking) on those "
        f"frames will be lost.\n\n"
        f"Continue?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return result == QMessageBox.StandardButton.Yes
