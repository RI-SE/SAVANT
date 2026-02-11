# edit/frontend/utils/annotation_ops.py
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QVBoxLayout,
)

from edit.frontend.exceptions import InvalidFrameRangeInput, MissingObjectIDError
from edit.frontend.states.annotation_state import AnnotationMode, AnnotationState
from edit.frontend.states.frontend_state import FrontendState
from edit.frontend.types import Relationship
from edit.frontend.utils.undo import (
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
    TrackObjectCommand,
    UpdateBBoxGeometryCommand,
)
from edit.frontend.widgets.cascade_dropdown import CascadeDirection
from edit.frontend.widgets.create_relationship_widget import RelationLinkerWidget
from edit.frontend.widgets.delete_relationship_widget import RelationDeleterWidget
from edit.services.exceptions import VideoLoadError

from .render import refresh_frame


def wire(main_window, frontend_state: FrontendState):
    """
    Connect all annotation-related signals. Safe to call once in MainWindow.__init__.
    """

    def _ensure_annotator_available() -> bool:
        annotator = frontend_state.require_current_annotator()
        return bool(annotator)

    def _call_with_annotator(callback, *args, **kwargs):
        annotator = frontend_state.require_current_annotator()
        if not annotator:
            return
        kwargs.setdefault("annotator", annotator)
        return callback(*args, **kwargs)

    # TODO: Reverse naming of signal and function here.
    if hasattr(main_window.sidebar, "start_bbox_drawing"):
        main_window.sidebar.start_bbox_drawing.connect(
            lambda object_type: _ensure_annotator_available()
            and on_new_object_bbox(main_window, object_type)
        )

    # TODO: Reverse naming of signal and function here.
    if hasattr(main_window.sidebar, "add_new_bbox_existing_obj"):
        main_window.sidebar.add_new_bbox_existing_obj.connect(
            lambda object_id: _ensure_annotator_available()
            and on_existing_object_bbox(main_window, object_id)
        )

    if hasattr(main_window.sidebar, "highlight_selected_object"):
        main_window.sidebar.highlight_selected_object.connect(
            lambda object_id: highlight_selected_object(main_window, object_id)
        )

    if hasattr(main_window.sidebar, "zoom_to_selected_object"):
        main_window.sidebar.zoom_to_selected_object.connect(
            lambda object_id: _zoom_to_object(main_window, object_id)
        )

    if hasattr(main_window.overlay, "cycle_bbox_requested"):
        main_window.overlay.cycle_bbox_requested.connect(
            lambda direction: _cycle_bbox(main_window, direction)
        )

    if hasattr(main_window.video_widget, "bbox_drawn"):
        main_window.video_widget.bbox_drawn.connect(
            lambda annotation: _call_with_annotator(
                handle_drawn_bbox, main_window, annotation
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
        lambda i, x, y: _call_with_annotator(_moved, main_window, i, x, y)
    )

    main_window.overlay.boxResized.connect(
        lambda id, x, y, w, h, rotation: _call_with_annotator(
            _resized,
            main_window,
            id,
            x,
            y,
            w,
            h,
            rotation=rotation,
        )
    )
    main_window.overlay.boxRotated.connect(
        lambda id, width, height, rotation: _call_with_annotator(
            _rotated,
            main_window,
            id,
            width,
            height,
            rotation=rotation,
        )
    )

    # Connect cascade signals
    if hasattr(main_window.overlay, "cascadeApplyAll"):
        main_window.overlay.cascadeApplyAll.connect(
            lambda object_id, center_x, center_y, width, height, rotation, direction: _call_with_annotator(
                _apply_cascade_all_frames,
                main_window,
                object_id,
                center_x,
                center_y,
                width,
                height,
                new_rotation=rotation,
                direction=direction,
            )
        )
    if hasattr(main_window.overlay, "cascadeApplyFrameRange"):
        main_window.overlay.cascadeApplyFrameRange.connect(
            lambda object_id, center_x, center_y, width, height, rotation, direction: _call_with_annotator(
                _apply_cascade_next_frames,
                main_window,
                object_id,
                center_x,
                center_y,
                width,
                height,
                rotation=rotation,
                direction=direction,
            )
        )

    # Keep this here so that right-click works without having to select a bbox first
    _install_overlay_context_menu(main_window, frontend_state)


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
    _refresh_after_annotation_change(main_window)


def highlight_selected_object(main_window, object_id: str):
    """Highlight the selected object in the overlay."""
    main_window.overlay.select_box_by_obj_id(object_id)


def _zoom_to_object(main_window, object_id: str):
    """Select the bbox in the overlay and zoom to it."""
    bbox = main_window.overlay._get_box_by_obj_id(object_id)
    if bbox is None:
        return
    main_window.overlay.select_box_by_obj_id(object_id)
    if hasattr(main_window, "zoom_to_bbox"):
        main_window.zoom_to_bbox(bbox)


def _cycle_bbox(main_window, direction: int):
    """Cycle to the next (+1) or previous (-1) bbox and zoom to it.

    Wraps around at list boundaries. If no bbox is selected, selects
    the first (direction=+1) or last (direction=-1) bbox.
    """
    overlay = main_window.overlay
    boxes = overlay._boxes
    if not boxes:
        return

    count = len(boxes)
    current_idx = overlay._selected_idx

    if current_idx is None:
        new_idx = 0 if direction > 0 else count - 1
    else:
        new_idx = (current_idx + direction) % count

    bbox = boxes[new_idx]
    overlay.select_box_by_obj_id(bbox.object_id)
    overlay.bounding_box_selected.emit(bbox.object_id)

    if hasattr(main_window, "zoom_to_bbox"):
        main_window.zoom_to_bbox(bbox)


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
    center_x: float,
    center_y: float,
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
        center_x=center_x,
        center_y=center_y,
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
    center_x: float,
    center_y: float,
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
        center_x=center_x,
        center_y=center_y,
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


def _install_overlay_context_menu(main_window, frontend_state: FrontendState):
    """Attach a custom right-click (context) menu to the video overlay."""
    overlay_widget = main_window.overlay
    overlay_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    overlay_widget.customContextMenuRequested.connect(
        lambda click_position: _on_overlay_context_menu(
            main_window, frontend_state, click_position
        )
    )


def _on_overlay_context_menu(main_window, frontend_state, click_position):
    """Handle right-clicks on the overlay and show a context menu for bbox actions."""
    overlay_widget = main_window.overlay
    bbox_index, _ = overlay_widget.hit_test(click_position)

    if bbox_index is None:
        _on_overlay_empty_space_context_menu(
            main_window, frontend_state, overlay_widget, click_position
        )
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

    # Add tracking actions
    context_menu.addSeparator()
    action_track_forward = context_menu.addAction("Track Forward")
    action_track_backward = context_menu.addAction("Track Backward")

    action_apply_static = None
    if obj_id:
        object_metadata = main_window.annotation_controller.get_object_metadata(obj_id)
        if object_metadata:
            object_type = object_metadata.get("type")
            if object_type:
                static_object_types = (
                    main_window.annotation_controller.allowed_bbox_types().get(
                        "StaticObject", []
                    )
                )
                if object_type.lower() in [
                    static_object_type.lower()
                    for static_object_type in static_object_types
                ]:
                    action_apply_static = context_menu.addAction(
                        "Apply to all frames (static)"
                    )

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

    action_copy_prev = None
    current_frame = int(main_window.video_controller.current_index())
    if obj_id and current_frame > 0:
        try:
            prev_bbox = main_window.annotation_controller.get_bbox(
                current_frame - 1, obj_id
            )
            if prev_bbox:
                context_menu.addSeparator()
                action_copy_prev = context_menu.addAction(
                    "Copy from previous frame"
                )
        except Exception:
            pass

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
        annotator = None
        if frontend_state is not None:
            annotator = frontend_state.require_current_annotator()
        if not annotator:
            return
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
    elif selected_action == action_apply_static:
        if obj_id:
            _apply_to_all_empty_frames(main_window, obj_id, frontend_state)
    elif selected_action == action_track_forward:
        _start_tracking(main_window, obj_id, "forward", frontend_state)
    elif selected_action == action_track_backward:
        _start_tracking(main_window, obj_id, "backward", frontend_state)
    elif selected_action == action_copy_prev:
        _copy_bbox_from_previous_frame(main_window, obj_id, frontend_state)


def _apply_to_all_empty_frames(
    main_window, object_id: str, frontend_state: FrontendState
):
    """Apply the bbox from the current frame to all frames where it's missing."""
    annotator = frontend_state.require_current_annotator()
    if not annotator:
        QMessageBox.warning(
            main_window, "Action Canceled", "An active annotator is required."
        )
        return

    current_frame = int(main_window.video_controller.current_index())
    source_bbox = main_window.annotation_controller.get_bbox(current_frame, object_id)
    if not source_bbox:
        QMessageBox.warning(
            main_window, "Action Canceled", "Source bounding box not found."
        )
        return

    object_metadata = main_window.annotation_controller.get_object_metadata(object_id)
    if not object_metadata:
        QMessageBox.warning(
            main_window, "Action Canceled", "Object metadata not found."
        )
        return
    object_type = object_metadata.get("type")
    if not object_type:
        QMessageBox.warning(
            main_window, "Action Canceled", "Object type not found in metadata."
        )
        return

    total_frames = main_window.project_state_controller.get_frame_count()
    existing_frames = main_window.annotation_controller.frames_for_object(object_id)
    all_frame_indices = set(range(total_frames))
    missing_frame_indices = sorted(list(all_frame_indices - set(existing_frames)))

    if not missing_frame_indices:
        QMessageBox.information(
            main_window, "Static BBox", "Object already exists on all frames."
        )
        return

    frame_ranges_str = _frames_to_ranges(missing_frame_indices)
    user_choice = QMessageBox.question(
        main_window,
        "Apply Static BBox",
        f"Apply this bounding box to frames: {frame_ranges_str} for object ID '{object_id}'?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    if user_choice != QMessageBox.StandardButton.Yes:
        return

    bbox_info = {
        "object_id": object_id,
        "coordinates": (
            source_bbox.x_center,
            source_bbox.y_center,
            source_bbox.width,
            source_bbox.height,
            source_bbox.rotation,
        ),
    }

    commands = [
        CreateExistingObjectBBoxCommand(
            frame_number=frame, bbox_info=bbox_info, annotator=annotator
        )
        for frame in missing_frame_indices
    ]

    composite_command = CompositeCommand(
        description=f"Apply static bbox to {len(commands)} frames", commands=commands
    )

    main_window.execute_undoable_command(composite_command)
    _refresh_after_annotation_change(main_window)

    QMessageBox.information(
        main_window,
        "Operation Complete",
        f"Applied bounding box to {len(missing_frame_indices)} frames.",
    )


def _start_tracking(
    main_window, object_id: str, direction: str, frontend_state: FrontendState
):
    """Start tracking the selected object forward or backward.

    Args:
        main_window: Main application window
        object_id: ID of the object to track
        direction: "forward" or "backward"
        frontend_state: Current frontend state
    """
    annotator = frontend_state.require_current_annotator()
    if not annotator:
        QMessageBox.warning(
            main_window, "Tracking", "An active annotator is required."
        )
        return

    tracking_service = getattr(main_window, "tracking_service", None)
    if tracking_service is None:
        QMessageBox.warning(
            main_window,
            "Tracking",
            "Tracking service not available. Check OpenCV installation.",
        )
        return

    current_frame = int(main_window.video_controller.current_index())
    bbox = main_window.overlay._get_selected_bbox()
    if not bbox:
        QMessageBox.warning(main_window, "Tracking", "No bounding box selected.")
        return

    # Create progress dialog
    progress = QProgressDialog(
        f"Tracking {direction}...", "Cancel", 0, 0, main_window
    )
    progress.setWindowTitle("Object Tracking")
    progress.setWindowModality(Qt.WindowModality.WindowModal)
    progress.setMinimumDuration(0)
    progress.setValue(0)
    progress.show()

    def on_progress(current_frame_idx: int, total_tracked: int) -> bool:
        """Update progress dialog. Returns True to cancel tracking."""
        progress.setLabelText(
            f"Tracking {direction}... Frame {current_frame_idx} "
            f"({total_tracked} frames tracked)"
        )
        QApplication.processEvents()
        return progress.wasCanceled()

    try:
        if direction == "forward":
            tracked_frames = tracking_service.track_forward(
                current_frame, bbox, object_id,
                iou_threshold=0.3,
                progress_callback=on_progress,
            )
        else:
            tracked_frames = tracking_service.track_backward(
                current_frame, bbox, object_id,
                iou_threshold=0.3,
                progress_callback=on_progress,
            )
    except RuntimeError as e:
        progress.close()
        QMessageBox.warning(main_window, "Tracking Error", str(e))
        return
    finally:
        progress.close()

    if progress.wasCanceled():
        if tracked_frames:
            # User cancelled but some frames were tracked - ask if they want to keep them
            keep = QMessageBox.question(
                main_window,
                "Tracking Cancelled",
                f"Tracking was cancelled. Keep the {len(tracked_frames)} frames tracked so far?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if keep != QMessageBox.StandardButton.Yes:
                return
        else:
            return

    if not tracked_frames:
        QMessageBox.information(
            main_window,
            "Tracking",
            "Tracking stopped immediately (object lost or overlap detected).",
        )
        return

    # Create undoable command for all tracked frames
    command = TrackObjectCommand(
        object_id=object_id,
        tracked_frames=tracked_frames,
        annotator=annotator,
    )
    main_window.execute_undoable_command(command)
    _refresh_after_annotation_change(main_window)

    frame_ranges_str = _frames_to_ranges([tf.frame_idx for tf in tracked_frames])
    QMessageBox.information(
        main_window,
        "Tracking Complete",
        f"Added bboxes to {len(tracked_frames)} frames: {frame_ranges_str}",
    )


def _on_overlay_empty_space_context_menu(
    main_window, frontend_state, overlay_widget, click_position
):
    """Show a context menu on empty space listing objects from the previous frame
    that are absent on the current frame, allowing the user to copy one."""
    current_frame = int(main_window.video_controller.current_index())
    if current_frame <= 0:
        return
    try:
        prev_objects = main_window.annotation_controller.get_active_objects(
            current_frame - 1
        )
        cur_objects = main_window.annotation_controller.get_active_objects(
            current_frame
        )
    except Exception:
        return
    cur_ids = {o["id"] for o in cur_objects}
    missing = [o for o in prev_objects if o["id"] not in cur_ids]
    if not missing:
        return
    context_menu = QMenu(overlay_widget)
    actions = {}
    for obj in missing:
        label = f"Copy {obj['id']} ({obj['type']}) from prev frame"
        actions[context_menu.addAction(label)] = obj["id"]
    selected_action = context_menu.exec(overlay_widget.mapToGlobal(click_position))
    if selected_action and selected_action in actions:
        _create_bbox_from_previous_frame(
            main_window, actions[selected_action], frontend_state
        )


def _copy_bbox_from_previous_frame(main_window, object_id, frontend_state):
    """Overwrite the current bbox geometry with values from the previous frame."""
    annotator = frontend_state.require_current_annotator()
    if not annotator:
        return
    current_frame = int(main_window.video_controller.current_index())
    prev_bbox = main_window.annotation_controller.get_bbox(
        current_frame - 1, object_id
    )
    if not prev_bbox:
        return

    def snapshot_builder(before):
        return BBoxGeometrySnapshot(
            center_x=prev_bbox.x_center,
            center_y=prev_bbox.y_center,
            width=prev_bbox.width,
            height=prev_bbox.height,
            rotation=prev_bbox.rotation,
        )

    _apply_geometry_update(main_window, object_id, annotator, snapshot_builder)


def _create_bbox_from_previous_frame(main_window, object_id, frontend_state):
    """Create a bbox on the current frame using geometry from the previous frame."""
    annotator = frontend_state.require_current_annotator()
    if not annotator:
        return
    current_frame = int(main_window.video_controller.current_index())
    prev_bbox = main_window.annotation_controller.get_bbox(
        current_frame - 1, object_id
    )
    if not prev_bbox:
        return
    bbox_info = {
        "object_id": object_id,
        "coordinates": (
            prev_bbox.x_center,
            prev_bbox.y_center,
            prev_bbox.width,
            prev_bbox.height,
            prev_bbox.rotation,
        ),
    }
    command = CreateExistingObjectBBoxCommand(
        frame_number=current_frame,
        bbox_info=bbox_info,
        annotator=annotator,
    )
    main_window.execute_undoable_command(command)
    _refresh_after_annotation_change(main_window)


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
