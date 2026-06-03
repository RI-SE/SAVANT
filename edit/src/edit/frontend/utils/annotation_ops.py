# edit/frontend/utils/annotation_ops.py
from PyQt6.QtCore import QTimer

from edit.frontend.states.annotation_state import AnnotationMode, AnnotationState
from edit.frontend.states.frontend_state import FrontendState
from edit.frontend.utils.undo import (
    BBoxGeometrySnapshot,
    CompositeCommand,
    CreateExistingObjectBBoxCommand,
    CreateNewObjectBBoxCommand,
    DeleteBBoxCommand,
    DeleteRelationshipCommand,
)

from edit.frontend.utils.settings_store import get_lock_to_center

from .render import refresh_frame
from ._annotation_helpers import (
    _apply_geometry_update,
    _refresh_after_annotation_change,
)
from ._annotation_helpers import _cascade_property_description  # noqa: F401 – re-exported for callers
from ._annotation_helpers import _frames_to_ranges  # noqa: F401 – re-exported for callers
from edit.frontend.utils.undo import CascadeDeltaBBoxCommand
from edit.frontend.widgets.cascade_dropdown import CascadeDirection
from ._annotation_cascade_ops import (
    _apply_cascade_all_frames,
    _apply_cascade_delta_all_frames,
    _apply_cascade_delta_next_frames,
    _apply_cascade_next_frames,
    _apply_rotate90_all_frames,
    _apply_rotate90_next_frames,
    _execute_cascade_command,
)
from ._annotation_context_menu import _install_overlay_context_menu
from ._annotation_relationship_ops import (
    _get_selected_object_relationships,
    _open_relationship_dialog,
)


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

    if hasattr(main_window.sidebar, "new_object_bbox_requested"):
        main_window.sidebar.new_object_bbox_requested.connect(
            lambda object_type: _ensure_annotator_available()
            and on_new_object_bbox(main_window, object_type)
        )

    if hasattr(main_window.sidebar, "add_new_bbox_existing_obj"):
        main_window.sidebar.add_new_bbox_existing_obj.connect(
            lambda object_id: _ensure_annotator_available()
            and on_existing_object_bbox(main_window, object_id)
        )

    if hasattr(main_window.sidebar, "highlight_selected_object"):
        main_window.sidebar.highlight_selected_object.connect(
            lambda object_id: highlight_selected_object(main_window, object_id)
        )
        main_window.sidebar.highlight_selected_object.connect(
            lambda object_id: _pan_to_center_if_locked(main_window, object_id)
        )

    if hasattr(main_window.sidebar, "lock_to_center_checkbox"):
        main_window.sidebar.lock_to_center_checkbox.toggled.connect(
            lambda checked: _on_lock_to_center_toggled(main_window, checked)
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
        main_window.overlay.bounding_box_selected.connect(
            lambda object_id: _pan_to_center_if_locked(main_window, object_id)
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
        lambda e: _call_with_annotator(
            _resized,
            main_window,
            e.object_id,
            e.center_x,
            e.center_y,
            e.width,
            e.height,
            rotation=e.rotation,
        )
    )
    main_window.overlay.boxRotated.connect(
        lambda e: _call_with_annotator(
            _rotated,
            main_window,
            e.object_id,
            e.width,
            e.height,
            rotation=e.rotation,
        )
    )

    # Connect cascade signals
    if hasattr(main_window.overlay, "cascadeApplyAll"):
        main_window.overlay.cascadeApplyAll.connect(
            lambda e: _call_with_annotator(
                _apply_cascade_all_frames,
                main_window,
                e.object_id,
                e.center_x,
                e.center_y,
                e.width,
                e.height,
                new_rotation=e.rotation,
                direction=e.direction,
            )
        )
    if hasattr(main_window.overlay, "cascadeApplyFrameRange"):
        main_window.overlay.cascadeApplyFrameRange.connect(
            lambda e: _call_with_annotator(
                _apply_cascade_next_frames,
                main_window,
                e.object_id,
                e.center_x,
                e.center_y,
                e.width,
                e.height,
                rotation=e.rotation,
                direction=e.direction,
            )
        )
    if hasattr(main_window.overlay, "cascadeRotate90"):
        main_window.overlay.cascadeRotate90.connect(
            lambda object_id, clockwise, direction: _call_with_annotator(
                _apply_rotate90_all_frames,
                main_window,
                object_id,
                clockwise,
                direction=direction,
            )
        )
    if hasattr(main_window.overlay, "cascadeRotate90FrameRange"):
        main_window.overlay.cascadeRotate90FrameRange.connect(
            lambda object_id, clockwise, direction: _call_with_annotator(
                _apply_rotate90_next_frames,
                main_window,
                object_id,
                clockwise,
                direction=direction,
            )
        )
    if hasattr(main_window.overlay, "cascadeDeltaAll"):
        main_window.overlay.cascadeDeltaAll.connect(
            lambda object_id, direction: _call_with_annotator(
                _apply_cascade_delta_all_frames,
                main_window,
                object_id,
                direction=direction,
            )
        )
    if hasattr(main_window.overlay, "cascadeDeltaFrameRange"):
        main_window.overlay.cascadeDeltaFrameRange.connect(
            lambda object_id, direction: _call_with_annotator(
                _apply_cascade_delta_next_frames,
                main_window,
                object_id,
                direction=direction,
            )
        )

    # Keep this here so that right-click works without having to select a bbox first
    _install_overlay_context_menu(main_window, frontend_state)


def highlight_selected_object(main_window, object_id: str):
    """Highlight the selected object in the overlay."""
    main_window.overlay.select_box_by_obj_id(object_id)


def repeat_last_adjustment(main_window) -> None:
    """Re-apply the last geometry delta for the currently selected object.

    If the user moved/resized/rotated object X in the previous frame,
    pressing R in the next frame applies the same delta (dcx, dcy, dw, dh, dtheta)
    to that object's bbox in the current frame.
    """
    import math as _math

    object_id = main_window.overlay.selected_object_id()
    if not object_id:
        return

    last_deltas = getattr(main_window, "last_bbox_deltas", {})
    delta = last_deltas.get(object_id)
    if delta is None:
        return  # No previous adjustment for this object — silently do nothing

    dcx, dcy, dw, dh, dtheta = delta

    annotator = main_window.state.require_current_annotator()
    if not annotator:
        return

    def _builder(before):
        from edit.frontend.utils.undo.snapshots import BBoxGeometrySnapshot
        new_rotation = (before.rotation + dtheta) % (2 * _math.pi)
        return BBoxGeometrySnapshot(
            center_x=before.center_x + dcx,
            center_y=before.center_y + dcy,
            width=max(1.0, before.width + dw),
            height=max(1.0, before.height + dh),
            rotation=new_rotation,
        )

    _apply_geometry_update(main_window, object_id, annotator, _builder)


def cascade_delta_forward_all(main_window) -> None:
    """Shift+R: apply last delta to all subsequent frames — no confirmation dialog."""
    _cascade_delta_all_shortcut(main_window, CascadeDirection.FORWARDS)


def cascade_delta_backward_all(main_window) -> None:
    """Ctrl+R: apply last delta to all previous frames — with lightweight confirmation."""
    from PyQt6.QtWidgets import QMessageBox

    object_id = main_window.overlay.selected_object_id()
    if not object_id:
        return
    last_deltas = getattr(main_window, "last_bbox_deltas", {})
    if last_deltas.get(object_id) is None:
        return

    current_frame = int(main_window.video_controller.current_index())
    if current_frame < 1:
        return

    result = QMessageBox.question(
        main_window,
        "Cascade Delta Backward",
        f"Apply last delta to all frames before frame {current_frame} "
        f"for object '{object_id}'?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    if result != QMessageBox.StandardButton.Yes:
        return

    _cascade_delta_all_shortcut(main_window, CascadeDirection.BACKWARDS)


def _cascade_delta_all_shortcut(main_window, direction: CascadeDirection) -> None:
    """Keyboard-shortcut path for cascade-delta-to-all: skips the confirmation dialog."""
    object_id = main_window.overlay.selected_object_id()
    if not object_id:
        return

    last_deltas = getattr(main_window, "last_bbox_deltas", {})
    delta = last_deltas.get(object_id)
    if delta is None:
        return  # no delta recorded yet — same silent behaviour as R

    annotator = main_window.state.require_current_annotator()
    if not annotator:
        return

    dcx, dcy, dw, dh, dtheta = delta
    current_frame = int(main_window.video_controller.current_index())
    last_frame = main_window.project_state_controller.get_frame_count() - 1

    if direction == CascadeDirection.FORWARDS:
        start_frame, end_frame = current_frame + 1, last_frame
    else:
        start_frame, end_frame = 0, current_frame - 1

    if start_frame > end_frame:
        return  # already at boundary — nothing to do

    command = CascadeDeltaBBoxCommand(
        object_id=str(object_id),
        frame_start=start_frame,
        frame_end=end_frame,
        dcx=dcx, dcy=dcy, dw=dw, dh=dh, dtheta=dtheta,
        annotator=annotator,
    )
    _execute_cascade_command(
        main_window, command, "Applied delta to",
        empty_title="Cascade Delta", complete_title="Cascade Delta Complete",
    )


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


def _on_lock_to_center_toggled(main_window, checked: bool):
    """When lock-to-center is enabled, immediately centre the selected bbox."""
    if not checked:
        return
    pan_fn = getattr(main_window, "pan_to_selected_bbox", None)
    if callable(pan_fn):
        QTimer.singleShot(0, pan_fn)


def _pan_to_center_if_locked(main_window, object_id):
    """If lock-to-center is active and a bbox was selected, centre it immediately.

    Skipped during frame advances (`_frame_updating` flag set in render.show_frame)
    because render.show_frame calls pan_to_selected_bbox directly once the overlay
    state is fully settled.
    """
    if not get_lock_to_center():
        return
    if getattr(main_window, "_frame_updating", False):
        return
    if not object_id:
        return
    pan_fn = getattr(main_window, "pan_to_selected_bbox", None)
    if callable(pan_fn):
        QTimer.singleShot(0, pan_fn)


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
