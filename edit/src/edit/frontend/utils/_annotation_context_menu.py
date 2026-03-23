# edit/frontend/utils/_annotation_context_menu.py
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMenu, QMessageBox

from edit.frontend.states.frontend_state import FrontendState
from edit.frontend.utils.undo import ResolveConfidenceCommand
from edit.frontend.widgets.delete_relationship_widget import RelationDeleterWidget

from ._annotation_helpers import _refresh_after_annotation_change
from ._annotation_delete_ops import _cascade_delete_directional, _cascade_delete_same_id
from ._annotation_relationship_ops import (
    _get_selected_object_relationships,
    _link_object_ids_interactive,
    _on_delete_relationship,
)
from ._annotation_tracking_ops import (
    _apply_to_all_empty_frames,
    _copy_bbox_from_previous_frame,
    _create_bbox_from_previous_frame,
    _start_tracking,
    _start_tracking_to_frame,
)


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


def _install_overlay_context_menu(main_window, frontend_state: FrontendState):
    """Attach a custom right-click (context) menu to the video overlay."""
    overlay_widget = main_window.overlay
    overlay_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    overlay_widget.customContextMenuRequested.connect(
        lambda click_position: _on_overlay_context_menu(
            main_window, frontend_state, click_position
        )
    )


def _build_bbox_context_menu(main_window, obj_id, overlay_widget, current_frame):
    """Build the right-click context menu for a bounding box.

    Returns:
        tuple[QMenu, dict]: The menu and a dict mapping string keys to QAction | None.
    """
    context_menu = QMenu(overlay_widget)
    action_delete_single = context_menu.addAction("Delete this bbox")
    action_delete_cascade = context_menu.addAction("Cascade delete all with this ID")
    action_delete_forward = context_menu.addAction("Cascade delete from here forward")
    action_delete_backward = context_menu.addAction("Cascade delete from here backward")
    action_delete_relationship = context_menu.addAction("Delete relationships")

    context_menu.addSeparator()
    action_track_forward = context_menu.addAction("Track Forward")
    action_track_forward_to = context_menu.addAction("Track Forward to Frame...")
    action_track_backward = context_menu.addAction("Track Backward")
    action_track_backward_to = context_menu.addAction("Track Backward to Frame...")

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
    if obj_id and current_frame > 0:
        prev_bbox = main_window.annotation_controller.try_get_bbox(
            current_frame - 1, obj_id
        )
        if prev_bbox:
            context_menu.addSeparator()
            action_copy_prev = context_menu.addAction("Copy from previous frame")

    actions = {
        "delete_single": action_delete_single,
        "delete_cascade": action_delete_cascade,
        "delete_forward": action_delete_forward,
        "delete_backward": action_delete_backward,
        "delete_relationship": action_delete_relationship,
        "track_forward": action_track_forward,
        "track_forward_to": action_track_forward_to,
        "track_backward": action_track_backward,
        "track_backward_to": action_track_backward_to,
        "apply_static": action_apply_static,
        "mark_resolved": mark_resolved_action,
        "link_ids": link_ids_action,
        "copy_prev": action_copy_prev,
        "available_ids": available_ids,  # type: ignore[dict-item]
    }
    return context_menu, actions


def _dispatch_bbox_context_action(
    main_window, frontend_state, obj_id, bbox_index, selected_action, actions
):
    """Execute the action selected from the bbox context menu."""
    # Lazy import to avoid circular dependency with annotation_ops
    from edit.frontend.utils.annotation_ops import delete_selected_bbox

    if selected_action == actions["delete_single"]:
        delete_selected_bbox(main_window)
    elif selected_action == actions["delete_cascade"]:
        from edit.frontend.exceptions import MissingObjectIDError
        try:
            _cascade_delete_same_id(main_window, bbox_index)
        except MissingObjectIDError as e:
            QMessageBox.warning(main_window, "Cascade Delete", str(e))
    elif selected_action == actions["delete_forward"]:
        if obj_id:
            _cascade_delete_directional(main_window, obj_id, "forward")
    elif selected_action == actions["delete_backward"]:
        if obj_id:
            _cascade_delete_directional(main_window, obj_id, "backward")
    elif selected_action == actions["mark_resolved"]:
        annotator = None
        if frontend_state is not None:
            annotator = frontend_state.require_current_annotator()
        if not annotator:
            return
        _mark_confidence_issue_resolved(main_window, obj_id, annotator)
    elif selected_action == actions["link_ids"]:
        _link_object_ids_interactive(main_window, obj_id, actions["available_ids"])
    elif selected_action == actions["delete_relationship"]:
        object_relationships = _get_selected_object_relationships(main_window, obj_id)
        relation_deleter_widget = RelationDeleterWidget(object_relationships)
        relation_deleter_widget.relationships_deleted.connect(
            lambda relation_ids: _on_delete_relationship(
                main_window, relation_ids=relation_ids
            )
        )
        relation_deleter_widget.exec()
    elif selected_action == actions["apply_static"]:
        if obj_id:
            _apply_to_all_empty_frames(main_window, obj_id, frontend_state)
    elif selected_action == actions["track_forward"]:
        _start_tracking(main_window, obj_id, "forward", frontend_state)
    elif selected_action == actions["track_forward_to"]:
        _start_tracking_to_frame(main_window, obj_id, "forward", frontend_state)
    elif selected_action == actions["track_backward"]:
        _start_tracking(main_window, obj_id, "backward", frontend_state)
    elif selected_action == actions["track_backward_to"]:
        _start_tracking_to_frame(main_window, obj_id, "backward", frontend_state)
    elif selected_action == actions["copy_prev"]:
        _copy_bbox_from_previous_frame(main_window, obj_id, frontend_state)


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

    current_frame = int(main_window.video_controller.current_index())
    context_menu, actions = _build_bbox_context_menu(
        main_window, obj_id, overlay_widget, current_frame
    )

    overlay_widget.setFocus()
    selected_action = context_menu.exec(overlay_widget.mapToGlobal(click_position))
    overlay_widget.setFocus()
    if selected_action is None:
        return

    _dispatch_bbox_context_action(
        main_window, frontend_state, obj_id, bbox_index, selected_action, actions
    )
    overlay_widget.setFocus()


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
