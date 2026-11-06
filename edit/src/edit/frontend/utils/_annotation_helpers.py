# edit/frontend/utils/_annotation_helpers.py
from __future__ import annotations

import math as _math
from html import escape

from edit.frontend.utils.undo import UpdateBBoxGeometryCommand
from edit.services.exceptions import VideoLoadError

from ._annotation_pure import cascade_property_description as _cascade_property_description  # noqa: F401
from ._annotation_pure import frames_to_ranges as _frames_to_ranges  # noqa: F401

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
    sidebar.refresh_active_frame_tags(current_index)
    sidebar.refresh_selected_relationships()
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


def _confirm_cascade(main_window, object_id, property_desc, frame_range_str) -> bool:
    """Show a confirmation dialog before executing a cascade operation."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QMessageBox
    msg = QMessageBox(
        QMessageBox.Icon.Warning,
        "Cascade Operation",
        f"This will overwrite the <b>{escape(str(property_desc))}</b> of object "
        f"'{escape(str(object_id))}' on frames {escape(str(frame_range_str))} with values from "
        f"the current frame.<br><br>"
        f"Existing per-frame adjustments (e.g. from tracking) on those "
        f"frames will be lost.<br><br>"
        f"Continue?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        main_window,
    )
    msg.setDefaultButton(QMessageBox.StandardButton.No)
    msg.setWindowFlag(Qt.WindowType.Tool)
    msg.raise_()
    msg.activateWindow()
    return msg.exec() == QMessageBox.StandardButton.Yes


def _plain_message_box(parent, icon, title: str, text: str, buttons, default_button):
    """Run a message box showing text literally, since object IDs come from project files."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QMessageBox
    box = QMessageBox(icon, title, text, buttons, parent)
    box.setTextFormat(Qt.TextFormat.PlainText)
    box.setDefaultButton(default_button)
    return box.exec()


def _plain_question(parent, title: str, text: str) -> bool:
    """Ask a Yes/No question in plain text, defaulting to No."""
    from PyQt6.QtWidgets import QMessageBox
    no_button = QMessageBox.StandardButton.No
    choice = _plain_message_box(
        parent, QMessageBox.Icon.Question, title, text,
        QMessageBox.StandardButton.Yes | no_button, no_button,
    )
    return choice == QMessageBox.StandardButton.Yes


def _plain_information(parent, title: str, text: str) -> None:
    """Show an information message in plain text."""
    from PyQt6.QtWidgets import QMessageBox
    ok_button = QMessageBox.StandardButton.Ok
    _plain_message_box(parent, QMessageBox.Icon.Information, title, text, ok_button, ok_button)


def _tool_information(parent, title: str, text: str) -> None:
    """Show an information dialog that stays on top of the main window."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QMessageBox
    msg = QMessageBox(QMessageBox.Icon.Information, title, text,
                      QMessageBox.StandardButton.Ok, parent)
    msg.setDefaultButton(QMessageBox.StandardButton.Ok)
    msg.setWindowFlag(Qt.WindowType.Tool)
    msg.raise_()
    msg.activateWindow()
    msg.exec()
