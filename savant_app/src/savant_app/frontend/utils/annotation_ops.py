# savant_app/frontend/utils/annotation_ops.py
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QVBoxLayout,
)

from savant_app.frontend.exceptions import InvalidFrameRangeInput, MissingObjectIDError
from savant_app.frontend.states.annotation_state import AnnotationMode, AnnotationState
from savant_app.frontend.states.frontend_state import FrontendState
from savant_app.frontend.types import Relationship
from savant_app.frontend.utils.undo import (
    BBoxGeometrySnapshot,
    CascadeBBoxCommand,
    CompositeCommand,
    CreateExistingObjectBBoxCommand,
    CreateNewObjectBBoxCommand,
    CreateObjectRelationshipCommand,
    DeleteBBoxCommand,
    DeleteRelationshipCommand,
    LinkObjectIdsCommand,
    ResolveConfidenceCommand,
    UpdateBBoxGeometryCommand,
)
from savant_app.frontend.widgets.cascade_dropdown import CascadeDirection
from savant_app.frontend.widgets.create_relationship_widget import RelationLinkerWidget
from savant_app.frontend.widgets.delete_relationship_widget import RelationDeleterWidget
from savant_app.services.exceptions import VideoLoadError

from .render import refresh_frame


def wire(main_window, frontend_state: FrontendState):
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
            lambda annotation: handle_drawn_bbox(
                main_window, annotation, frontend_state.get_current_annotator()
            )
        )

    if hasattr(main_window.overlay, "bounding_box_selected"):
        main_window.overlay.bounding_box_selected.connect(
            lambda object_id: highlight_active_obj_list(main_window, object_id)
        )

    if hasattr(main_window.overlay, "deletePressed"):
        main_window.overlay.deletePressed.connect(
            lambda: delete_selected_bbox(main_window)
        )

    if hasattr(main_window.sidebar, "object_details_changed"):
        main_window.sidebar.object_details_changed.connect(
            lambda: refresh_frame(main_window)
        )

    if hasattr(main_window.sidebar, "create_relationship"):
        main_window.sidebar.create_relationship.connect(
            lambda: _open_relationship_dialog(main_window)
        )

    main_window.overlay.boxMoved.connect(
        lambda i, x, y: _moved(
            main_window, i, x, y, frontend_state.get_current_annotator()
        )
    )

    main_window.overlay.boxResized.connect(
        lambda id, x, y, w, h, rotation: _resized(
            main_window,
            id,
            x,
            y,
            w,
            h,
            rotation,
            frontend_state.get_current_annotator(),
        )
    )
    main_window.overlay.boxRotated.connect(
        lambda id, width, height, rotation: _rotated(
            main_window,
            id,
            width,
            height,
            rotation,
            frontend_state.get_current_annotator(),
        )
    )

    # Connect cascade signals
    if hasattr(main_window.overlay, "cascadeApplyAll"):
        main_window.overlay.cascadeApplyAll.connect(
            lambda object_id, width, height, rotation, direction: _apply_cascade_all_frames(
                main_window,
                object_id,
                width,
                height,
                frontend_state.get_current_annotator(),
                rotation,
                direction,
            )
        )
    if hasattr(main_window.overlay, "cascadeApplyFrameRange"):
        main_window.overlay.cascadeApplyFrameRange.connect(
            lambda object_id, width, height, rotation, direction: _apply_cascade_next_frames(
                main_window,
                object_id,
                width,
                height,
                frontend_state.get_current_annotator(),
                rotation,
                direction,
            )
        )

    # Keep this here so that right-click works without having to select a bbox first
    _install_overlay_context_menu(main_window)


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
    _refresh_after_annotation_change(main_window)


def highlight_selected_object(main_window, object_id: str):
    """Highlight the selected object in the overlay."""
    main_window.overlay.select_box_by_obj_id(object_id)


def highlight_active_obj_list(main_window, object_id: str):
    """Highlight the selected object in the active object list."""
    sidebar = getattr(main_window, "sidebar", None)
    if sidebar is None:
        return

    if object_id:
        sidebar.select_active_object_by_id(object_id)
        sidebar.show_object_editor(object_id, expand=True)
    else:
        sidebar._selected_annotation_object_id = None
        sidebar.active_objects.clearSelection()
        sidebar.hide_object_editor()


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


def handle_drawn_bbox(main_window, annotation: AnnotationState, annotator: str):
    """Finalize newly drawn bbox → controller → refresh."""
    frame_idx = int(main_window.video_controller.current_index())
    payload = {
        "object_id": annotation.object_id,
        "object_type": annotation.object_type,
        "coordinates": annotation.coordinates,
    }

    if annotation.mode == AnnotationMode.EXISTING:
        if not annotation.object_id:
            return
        command = CreateExistingObjectBBoxCommand(
            frame_number=frame_idx,
            bbox_info=payload,
            annotator=annotator,
        )
    elif annotation.mode == AnnotationMode.NEW:
        command = CreateNewObjectBBoxCommand(
            frame_number=frame_idx,
            bbox_info=payload,
            annotator=annotator,
        )
    else:
        return

    main_window.execute_undoable_command(command)
    _refresh_after_annotation_change(main_window)


def delete_selected_bbox(main_window):
    """Delete the currently selected bbox and record it for undo."""
    object_id = main_window.overlay.selected_object_id()
    if object_id is None:
        return

    frame_key = int(main_window.video_controller.current_index())

    # Find relationships to delete
    object_relationships = _get_selected_object_relationships(main_window, object_id)
    delete_commands = [
        DeleteRelationshipCommand(rel.id) for rel in object_relationships
    ]

    # Add bbox deletion
    delete_commands.append(
        DeleteBBoxCommand(frame_number=frame_key, object_id=str(object_id))
    )

    if len(delete_commands) > 1:
        command = CompositeCommand(
            description=f"Delete bbox and {len(object_relationships)} relationships",
            commands=delete_commands,
        )
    else:
        command = delete_commands[0]

    main_window.execute_undoable_command(command)
    main_window.overlay.clear_selection()
    _refresh_after_annotation_change(main_window)


def undo_last_action(main_window):
    """Undo the most recent annotation operation."""
    command = main_window.undo_last_command()
    if command is None:
        return
    _refresh_after_annotation_change(main_window)


def redo_last_action(main_window):
    """Redo the most recently undone annotation operation."""
    command = main_window.redo_last_command()
    if command is None:
        return
    _refresh_after_annotation_change(main_window)


def _moved(main_window, object_id: str, x: float, y: float, annotator: str):
    if not object_id:
        return

    def snapshot_builder(before: BBoxGeometrySnapshot) -> BBoxGeometrySnapshot:
        return BBoxGeometrySnapshot(
            center_x=x,
            center_y=y,
            width=before.width,
            height=before.height,
            rotation=before.rotation,
        )

    _apply_geometry_update(main_window, object_id, annotator, snapshot_builder)


def _resized(
    main_window,
    object_id: str,
    x: float,
    y: float,
    width: float,
    height: float,
    rotation: float,
    annotator: str,
):
    if not object_id:
        return

    def snapshot_builder(before: BBoxGeometrySnapshot) -> BBoxGeometrySnapshot:
        return BBoxGeometrySnapshot(
            center_x=x,
            center_y=y,
            width=width,
            height=height,
            rotation=rotation if rotation is not None else before.rotation,
        )

    _apply_geometry_update(main_window, object_id, annotator, snapshot_builder)


def _rotated(
    main_window,
    object_id: str,
    width: float,
    height: float,
    rotation: float,
    annotator: str,
):
    if not object_id:
        return

    def snapshot_builder(before: BBoxGeometrySnapshot) -> BBoxGeometrySnapshot:
        return BBoxGeometrySnapshot(
            center_x=before.center_x,
            center_y=before.center_y,
            width=width if width is not None else before.width,
            height=height if height is not None else before.height,
            rotation=rotation if rotation is not None else before.rotation,
        )

    _apply_geometry_update(main_window, object_id, annotator, snapshot_builder)


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


def _apply_cascade_all_frames(
    main_window,
    object_id: str,
    new_width: float,
    new_height: float,
    annotator: str,
    new_rotation: float = 0.0,
    direction: CascadeDirection = CascadeDirection.FORWARDS,
):
    """Apply the resize/rotation to all frames containing the object."""
    last_frame = main_window.project_state_controller.get_frame_count() - 1
    current_frame = int(main_window.video_controller.current_index())

    if direction == CascadeDirection.FORWARDS:
        start_frame = current_frame
        end_frame = last_frame
    else:  # backwards
        start_frame = 0
        end_frame = current_frame

    command = CascadeBBoxCommand(
        object_id=str(object_id),
        frame_start=start_frame,
        frame_end=end_frame,
        width=new_width,
        height=new_height,
        rotation=new_rotation,
        annotator=annotator,
    )
    main_window.execute_undoable_command(command)
    modified_frames = sorted(command.modified_frames)
    if not modified_frames:
        QMessageBox.information(
            main_window,
            "Cascade Operation",
            "No frames were updated for this object.",
        )
        _refresh_after_annotation_change(main_window)
        return

    # Show confirmation
    frame_ranges_str = _frames_to_ranges(modified_frames)
    QMessageBox.information(
        main_window,
        "Cascade Operation Complete",
        f"Applied changes to {len(modified_frames)} frames: {frame_ranges_str}",
    )
    _refresh_after_annotation_change(main_window)


def _apply_cascade_next_frames(
    main_window,
    object_id: str,
    width: float,
    height: float,
    annotator: str,
    rotation: float,
    direction: CascadeDirection = CascadeDirection.FORWARDS,
):
    """Ask user for number of frames and apply the resize/rotation to those frames."""
    current_frame = int(main_window.video_controller.current_index())
    if direction == CascadeDirection.FORWARDS:
        max_frames = (
            main_window.project_state_controller.get_frame_count() - current_frame - 1
        )
        prompt = "Apply to how many subsequent frames?"
    else:  # backwards
        max_frames = current_frame
        prompt = "Apply to how many previous frames?"

    # Ask user for number of frames
    num_frames, ok = QInputDialog.getInt(
        main_window,
        "Cascade Operation",
        prompt,
        5,  # default value
        1,  # min value
        max_frames,  # max value
    )

    if not ok:
        return

    if num_frames > max_frames or num_frames < 1:
        raise InvalidFrameRangeInput(
            f"Please enter a valid number of frames (1-{max_frames})."
        )

    if direction == CascadeDirection.FORWARDS:
        start_frame = current_frame + 1
        end_frame = current_frame + num_frames
    else:  # backwards
        start_frame = current_frame - num_frames
        end_frame = current_frame - 1

    command = CascadeBBoxCommand(
        object_id=str(object_id),
        frame_start=start_frame,
        frame_end=end_frame,
        width=width,
        height=height,
        rotation=rotation,
        annotator=annotator,
    )
    main_window.execute_undoable_command(command)
    modified_frames = sorted(command.modified_frames)
    if not modified_frames:
        QMessageBox.information(
            main_window,
            "Cascade Operation",
            "No frames were updated for this object.",
        )
        _refresh_after_annotation_change(main_window)
        return

    # Show confirmation
    frame_ranges_str = _frames_to_ranges(modified_frames)
    QMessageBox.information(
        main_window,
        "Cascade Operation Complete",
        f"Applied changes to {len(modified_frames)} frames: {frame_ranges_str}",
    )

    _refresh_after_annotation_change(main_window)


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
    bbox_index, _ = overlay_widget.hit_test(click_position)

    if bbox_index is None:
        return

    overlay_widget._selected_idx = bbox_index
    overlay_widget.update()
    obj_id = overlay_widget.selected_object_id()
    if obj_id:
        overlay_widget.bounding_box_selected.emit(obj_id)

    context_menu = QMenu(overlay_widget)
    action_delete_single = context_menu.addAction("Delete this bbox")
    action_delete_cascade = context_menu.addAction("Cascade delete all with this ID")
    action_delete_relationship = context_menu.addAction("Delete relationships")
    confidence_flags = overlay_widget.confidence_flags()
    mark_resolved_action = None
    if obj_id and confidence_flags.get(obj_id):
        mark_resolved_action = context_menu.addAction("Mark issue as resolved")

    link_ids_action = None
    available_ids: list[str] = []
    if obj_id:
        try:
            available_ids = [
                candidate_id
                for candidate_id in main_window.annotation_controller.list_object_ids()
                if candidate_id != obj_id
            ]
        except Exception:
            available_ids = []
    if obj_id and available_ids:
        link_ids_action = context_menu.addAction("Link object IDs")

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
    elif selected_action == mark_resolved_action:
        annotator = ""
        state = getattr(main_window, "state", None)
        if state and hasattr(state, "get_current_annotator"):
            annotator = state.get_current_annotator() or ""
        _mark_confidence_issue_resolved(main_window, obj_id, annotator)
    elif selected_action == link_ids_action:
        _link_object_ids_interactive(main_window, obj_id, available_ids)
    elif selected_action == action_delete_relationship:
        object_relationships = _get_selected_object_relationships(main_window, obj_id)
        relation_deleter_widget = RelationDeleterWidget(object_relationships)
        relation_deleter_widget.relationships_deleted.connect(
            lambda relation_ids: _on_delete_relationship(
                main_window, relation_ids=relation_ids
            )
        )
        relation_deleter_widget.exec()


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


def _mark_confidence_issue_resolved(
    main_window, object_id: str, annotator: str
) -> None:
    """Set the confidence for the selected bbox to 'resolved' (confidence = 1.0)."""
    try:
        frame_index = int(main_window.video_controller.current_index())
    except Exception:
        return

    if not object_id:
        return

    command = ResolveConfidenceCommand(
        frame_number=frame_index,
        object_id=str(object_id),
        annotator=annotator,
    )
    main_window.execute_undoable_command(command)
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
    # Get current frame objects for the linker widget
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

    # Create the command
    command = CreateObjectRelationshipCommand(
        relationship_type=relationship_type,
        ontology_uid=ontology_uid,
        subject_object_id=subject_id,
        object_object_id=object_id,
    )

    # Execute the command
    main_window.execute_undoable_command(command)

    # Refresh the UI
    _refresh_after_annotation_change(main_window)
