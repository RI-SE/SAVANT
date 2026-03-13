# edit/frontend/utils/_annotation_tracking_ops.py
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QInputDialog, QMessageBox, QProgressDialog

from edit.frontend.states.frontend_state import FrontendState
from edit.frontend.utils.undo import (
    BBoxGeometrySnapshot,
    CompositeCommand,
    CreateExistingObjectBBoxCommand,
    TrackObjectCommand,
)

from ._annotation_helpers import (
    _apply_geometry_update,
    _frames_to_ranges,
    _refresh_after_annotation_change,
)


def _validate_tracking_preconditions(main_window, object_id, direction, frontend_state):
    """Validate that all preconditions for tracking are met.

    Returns:
        tuple[str, object, int, object] | None: (annotator, tracking_service,
            current_frame, bbox) if valid, or None if any check fails.
    """
    annotator = frontend_state.require_current_annotator()
    if not annotator:
        QMessageBox.warning(
            main_window, "Tracking", "An active annotator is required."
        )
        return None

    tracking_service = getattr(main_window, "tracking_service", None)
    if tracking_service is None:
        QMessageBox.warning(
            main_window,
            "Tracking",
            "Tracking service not available. Check OpenCV installation.",
        )
        return None

    current_frame = int(main_window.video_controller.current_index())
    bbox = main_window.overlay._get_selected_bbox()
    if not bbox:
        QMessageBox.warning(main_window, "Tracking", "No bounding box selected.")
        return None

    return annotator, tracking_service, current_frame, bbox


def _start_tracking(
    main_window, object_id: str, direction: str, frontend_state: FrontendState,
    stop_frame: Optional[int] = None,
):
    """Start tracking the selected object forward or backward.

    Args:
        main_window: Main application window
        object_id: ID of the object to track
        direction: "forward" or "backward"
        frontend_state: Current frontend state
        stop_frame: Optional frame index at which tracking must stop (inclusive).
    """
    validated = _validate_tracking_preconditions(
        main_window, object_id, direction, frontend_state
    )
    if validated is None:
        return
    annotator, tracking_service, current_frame, bbox = validated

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
                stop_frame=stop_frame,
            )
        else:
            tracked_frames = tracking_service.track_backward(
                current_frame, bbox, object_id,
                iou_threshold=0.3,
                progress_callback=on_progress,
                stop_frame=stop_frame,
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


def _start_tracking_to_frame(
    main_window, object_id: str, direction: str, frontend_state: FrontendState
):
    """Ask the user for a stop frame and then start tracking toward it."""
    current_frame = int(main_window.video_controller.current_index())
    frame_count = main_window.project_state_controller.get_frame_count() or 0

    if direction == "forward":
        default_stop = frame_count - 1
        min_stop = current_frame + 1
        max_stop = frame_count - 1
        prompt = f"Track forward until frame (current: {current_frame}, max: {max_stop}):"
    else:
        default_stop = 0
        min_stop = 0
        max_stop = max(0, current_frame - 1)
        prompt = f"Track backward until frame (current: {current_frame}, min: {min_stop}):"

    if min_stop > max_stop:
        QMessageBox.information(main_window, "Tracking", "No frames available to track toward.")
        return

    stop, ok = QInputDialog.getInt(
        main_window, "Track to Frame", prompt,
        value=default_stop, min=min_stop, max=max_stop,
    )
    if not ok:
        return

    _start_tracking(main_window, object_id, direction, frontend_state, stop_frame=stop)


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
