# edit/frontend/utils/_annotation_relationship_ops.py
from __future__ import annotations

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from edit.frontend.types import Relationship
from edit.frontend.utils.undo import (
    CompositeCommand,
    CreateObjectRelationshipCommand,
    DeleteRelationshipCommand,
    LinkObjectIdsCommand,
)
from edit.frontend.widgets.create_relationship_widget import RelationLinkerWidget

from ._annotation_helpers import _frames_to_ranges, _refresh_after_annotation_change


def _get_selected_object_relationships(
    main_window, object_id: str
) -> list[Relationship]:
    object_relationships = main_window.annotation_controller.get_object_relationship(
        object_id
    )
    return [
        Relationship(
            id=relationship["id"],
            subject=relationship["subject"],
            relationship_type=relationship["type"],
            object=relationship["object"],
        )
        for relationship in object_relationships
    ]


def _on_delete_relationship(main_window, relation_ids: list[str]):
    """Delete relationships"""
    delete_commands = [
        DeleteRelationshipCommand(relation_id) for relation_id in relation_ids
    ]
    batch_command = CompositeCommand(
        description=f"Delete {len(delete_commands)} relationships",
        commands=delete_commands,
    )
    main_window.execute_undoable_command(batch_command)
    _refresh_after_annotation_change(main_window)


def _prompt_link_target_object(
    main_window, source_object_id: str, candidate_ids: list[str]
) -> str | None:
    """Display a dialog allowing the user to choose an object ID to link."""
    dialog = QDialog(main_window)
    dialog.setWindowTitle("Link Object IDs")
    layout = QVBoxLayout(dialog)

    selection_combo = QComboBox(dialog)
    selection_combo.setEditable(True)
    selection_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
    placeholder_text = "Type or select ID"
    selection_combo.lineEdit().setPlaceholderText(placeholder_text)
    selection_combo.setMinimumWidth(len(placeholder_text) * 10)

    unique_candidates = sorted(
        {candidate for candidate in candidate_ids if candidate != source_object_id},
        key=lambda value: (
            (0, f"{int(value):010d}") if str(value).isdigit() else (1, str(value))
        ),
    )
    if unique_candidates:
        selection_combo.addItems(unique_candidates)
    selection_combo.setCurrentIndex(-1)
    layout.addWidget(selection_combo)

    layout.addSpacing(8)

    selection_description = QLabel(
        "Select the target object ID that should be merged into the current object.",
        dialog,
    )
    selection_description.setWordWrap(True)
    hint_font: QFont = selection_description.font()
    hint_font.setItalic(True)
    hint_font.setPointSize(max(8, hint_font.pointSize() - 1))
    selection_description.setFont(hint_font)
    layout.addWidget(selection_description)

    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
        parent=dialog,
    )
    layout.addWidget(buttons)

    selection_state: dict[str, object] = {"value": None}

    def _accept():
        candidate = selection_combo.currentText().strip()
        if not candidate:
            QMessageBox.warning(
                dialog, "Link Object IDs", "Select an object ID to link."
            )
            return
        selection_state["value"] = candidate
        dialog.accept()

    buttons.accepted.connect(_accept)
    buttons.rejected.connect(dialog.reject)

    if dialog.exec() == QDialog.DialogCode.Accepted:
        return selection_state["value"]
    return None


def _ask_link_conflict_resolution(
    main_window,
    conflict_frames: list[int],
    primary_id: str,
    secondary_id: str,
) -> str | None:
    """Ask the user how to resolve frames that contain both object IDs.

    Returns "keep_primary", "keep_secondary", or None to cancel.
    """
    frame_summary = _frames_to_ranges(conflict_frames)
    n = len(conflict_frames)

    dialog = QDialog(main_window)
    dialog.setWindowTitle("Link Object IDs — Conflict")
    layout = QVBoxLayout(dialog)

    info = QLabel(
        f"<b>{n} frame(s)</b> already contain both '{primary_id}' and '{secondary_id}':"
        f"<br>{frame_summary}<br><br>"
        f"Choose how to resolve the conflict:"
    )
    info.setWordWrap(True)
    layout.addWidget(info)

    resolution: dict = {"value": None}

    keep_primary_btn = QPushButton(f"Keep '{primary_id}' (drop secondary in {n} frame(s))")
    keep_secondary_btn = QPushButton(f"Keep '{secondary_id}' (replace primary in {n} frame(s))")
    cancel_btn = QPushButton("Cancel")

    for btn in (keep_primary_btn, keep_secondary_btn, cancel_btn):
        layout.addWidget(btn)

    def _choose(value):
        resolution["value"] = value
        dialog.accept()

    keep_primary_btn.clicked.connect(lambda: _choose("keep_primary"))
    keep_secondary_btn.clicked.connect(lambda: _choose("keep_secondary"))
    cancel_btn.clicked.connect(dialog.reject)

    dialog.exec()
    return resolution["value"]


def _link_object_ids_interactive(
    main_window, primary_object_id: str, candidate_ids: list[str]
) -> None:
    """Interactive flow for replacing one object ID with another across frames."""
    if not primary_object_id:
        return

    target_object_id = _prompt_link_target_object(
        main_window, primary_object_id, candidate_ids
    )
    if not target_object_id:
        return

    conflict_frames = main_window.annotation_controller.conflicting_frames_for_link(
        primary_object_id, target_object_id
    ) or []

    conflict_resolution: str | None = None
    if conflict_frames:
        conflict_resolution = _ask_link_conflict_resolution(
            main_window, conflict_frames, primary_object_id, target_object_id
        )
        if conflict_resolution is None:
            return
    else:
        conflict_resolution = "keep_primary"  # no conflicts, value is irrelevant

    frames_with_target = main_window.annotation_controller.frames_for_object(
        target_object_id
    )
    frame_summary = _frames_to_ranges(frames_with_target)
    frame_count = len(frames_with_target)
    confirmation_text = (
        f"Replace all occurrences of ID '{target_object_id}' with '{primary_object_id}' "
        f"across {frame_count} frame(s)"
    )
    if frame_summary:
        confirmation_text += f": {frame_summary}"
    else:
        confirmation_text += "."
    if conflict_frames:
        confirmation_text += (
            f"\n\n{len(conflict_frames)} conflicting frame(s) will use "
            f"'{'keep primary' if conflict_resolution == 'keep_primary' else 'keep secondary'}' strategy."
        )
    confirmation_text += "\nYou can undo this action if needed."

    confirm = QMessageBox.question(
        main_window,
        "Link Object IDs",
        confirmation_text,
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    if confirm != QMessageBox.StandardButton.Yes:
        return

    command = LinkObjectIdsCommand(
        primary_object_id=str(primary_object_id),
        secondary_object_id=str(target_object_id),
        conflict_resolution=conflict_resolution,
    )
    main_window.execute_undoable_command(command)
    linked_frames = list(command.affected_frames)
    _refresh_after_annotation_change(main_window)
    result_summary = _frames_to_ranges(linked_frames)
    success_message = (
        f"Linked ID '{target_object_id}' into '{primary_object_id}' across "
        f"{len(linked_frames)} frame(s)"
    )
    if result_summary:
        success_message += f": {result_summary}"
    else:
        success_message += "."
    QMessageBox.information(main_window, "Link Object IDs", success_message)

    overlay = getattr(main_window, "overlay", None)
    if overlay is not None:
        overlay.bounding_box_selected.emit(primary_object_id)


def _open_relationship_dialog(main_window):
    """Open the relationship creation dialog."""
    current_frame = int(main_window.video_controller.current_index())
    current_objects = main_window.annotation_controller.get_active_objects(
        current_frame
    )

    linker_widget = RelationLinkerWidget(current_objects)
    linker_widget.relationship_created.connect(
        lambda subject_id, object_id, relationship_type: _on_create_relationship(
            main_window, subject_id, object_id, relationship_type
        )
    )
    linker_widget.exec()


def _on_create_relationship(
    main_window, subject_id: str, object_id: str, relationship_type: str
):
    """Handle the creation of a new object relationship."""
    # Temporarily hard coded until we implement ontology uid management
    ontology_uid = "1.3.0"

    command = CreateObjectRelationshipCommand(
        relationship_type=relationship_type,
        ontology_uid=ontology_uid,
        subject_object_id=subject_id,
        object_object_id=object_id,
    )

    main_window.execute_undoable_command(command)
    _refresh_after_annotation_change(main_window)
