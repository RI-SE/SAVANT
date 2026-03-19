# edit/frontend/utils/_annotation_cascade_ops.py
from __future__ import annotations

import math as _math

from PyQt6.QtWidgets import QInputDialog

from edit.frontend.exceptions import InvalidFrameRangeInput
from edit.frontend.utils.undo import CascadeBBoxCommand, CascadeDeltaBBoxCommand, Rotate90CascadeCommand
from edit.frontend.widgets.cascade_dropdown import CascadeDirection

from ._annotation_helpers import (
    _cascade_property_description,
    _confirm_cascade,
    _frames_to_ranges,
    _refresh_after_annotation_change,
    _tool_information,
)


def _ask_next_frame_range(
    main_window,
    direction: CascadeDirection,
    dialog_title: str = "Cascade Operation",
    prompt_prefix: str = "Apply",
):
    """Ask the user for a number of frames and return (start_frame, end_frame), or None if cancelled."""
    current_frame = int(main_window.video_controller.current_index())
    if direction == CascadeDirection.FORWARDS:
        max_frames = main_window.project_state_controller.get_frame_count() - current_frame - 1
        prompt = f"{prompt_prefix} to how many subsequent frames?"
    else:
        max_frames = current_frame
        prompt = f"{prompt_prefix} to how many previous frames?"

    num_frames, ok = QInputDialog.getInt(main_window, dialog_title, prompt, 5, 1, max_frames)
    if not ok:
        return None
    if num_frames > max_frames or num_frames < 1:
        raise InvalidFrameRangeInput(f"Please enter a valid number of frames (1-{max_frames}).")

    if direction == CascadeDirection.FORWARDS:
        start_frame = current_frame + 1
        end_frame = current_frame + num_frames
    else:
        start_frame = current_frame - num_frames
        end_frame = current_frame - 1
    return start_frame, end_frame


def _execute_cascade_command(
    main_window,
    command,
    success_msg_prefix: str,
    empty_title: str = "Cascade Operation",
    complete_title: str = "Cascade Operation Complete",
):
    """Execute a cascade command, show result dialog, and refresh the view."""
    main_window.execute_undoable_command(command)
    modified_frames = sorted(command.modified_frames)
    if not modified_frames:
        _tool_information(main_window, empty_title, "No frames were updated for this object.")
        _refresh_after_annotation_change(main_window)
        return
    frame_ranges_str = _frames_to_ranges(modified_frames)
    _tool_information(
        main_window,
        complete_title,
        f"{success_msg_prefix} {len(modified_frames)} frames: {frame_ranges_str}",
    )
    _refresh_after_annotation_change(main_window)


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

    prop_desc = _cascade_property_description(
        center_x, center_y, new_width, new_height, new_rotation
    )
    if not _confirm_cascade(
        main_window, object_id, prop_desc, f"{start_frame}–{end_frame}"
    ):
        return

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
    _execute_cascade_command(main_window, command, "Applied changes to")


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
    frame_range = _ask_next_frame_range(main_window, direction)
    if frame_range is None:
        return
    start_frame, end_frame = frame_range

    prop_desc = _cascade_property_description(center_x, center_y, width, height, rotation)
    if not _confirm_cascade(
        main_window, object_id, prop_desc, f"{start_frame}–{end_frame}"
    ):
        return

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
    _execute_cascade_command(main_window, command, "Applied changes to")


def _apply_rotate90_all_frames(
    main_window,
    object_id: str,
    clockwise: bool,
    annotator: str,
    direction: CascadeDirection = CascadeDirection.FORWARDS,
):
    """Rotate heading 90° CW/CCW on all frames containing the object."""
    last_frame = main_window.project_state_controller.get_frame_count() - 1
    current_frame = int(main_window.video_controller.current_index())

    if direction == CascadeDirection.FORWARDS:
        start_frame = current_frame
        end_frame = last_frame
    else:
        start_frame = 0
        end_frame = current_frame

    label = "90° CW" if clockwise else "90° CCW"
    if not _confirm_cascade(
        main_window, object_id, f"heading {label}", f"{start_frame}–{end_frame}"
    ):
        return

    command = Rotate90CascadeCommand(
        object_id=str(object_id),
        frame_start=start_frame,
        frame_end=end_frame,
        clockwise=clockwise,
        annotator=annotator,
    )
    _execute_cascade_command(main_window, command, f"Rotated heading {label} on")


def _apply_rotate90_next_frames(
    main_window,
    object_id: str,
    clockwise: bool,
    annotator: str,
    direction: CascadeDirection = CascadeDirection.FORWARDS,
):
    """Ask user for number of frames, then rotate heading 90° CW/CCW."""
    frame_range = _ask_next_frame_range(main_window, direction)
    if frame_range is None:
        return
    start_frame, end_frame = frame_range

    label = "90° CW" if clockwise else "90° CCW"
    if not _confirm_cascade(
        main_window, object_id, f"heading {label}", f"{start_frame}–{end_frame}"
    ):
        return

    command = Rotate90CascadeCommand(
        object_id=str(object_id),
        frame_start=start_frame,
        frame_end=end_frame,
        clockwise=clockwise,
        annotator=annotator,
    )
    _execute_cascade_command(main_window, command, f"Rotated heading {label} on")


def _apply_cascade_delta_all_frames(
    main_window,
    object_id: str,
    annotator: str,
    direction: CascadeDirection = CascadeDirection.FORWARDS,
):
    """Apply the last recorded geometry delta to all annotated frames for the object."""
    last_deltas = getattr(main_window, "last_bbox_deltas", {})
    delta = last_deltas.get(object_id)
    if delta is None:
        QMessageBox.information(
            main_window,
            "Cascade Delta",
            "No recorded adjustment for this object yet.\n"
            "Move, resize or rotate the bbox first (the R-key shortcut records the delta).",
        )
        return

    dcx, dcy, dw, dh, dtheta = delta
    last_frame = main_window.project_state_controller.get_frame_count() - 1
    current_frame = int(main_window.video_controller.current_index())

    if direction == CascadeDirection.FORWARDS:
        start_frame = current_frame + 1
        end_frame = last_frame
        range_str = f"{start_frame}–{end_frame}"
    else:
        start_frame = 0
        end_frame = current_frame - 1
        range_str = f"{start_frame}–{end_frame}"

    prop_desc = (
        f"delta (Δcx={dcx:+.1f}, Δcy={dcy:+.1f}, "
        f"Δw={dw:+.1f}, Δh={dh:+.1f}, "
        f"Δθ={_math.degrees(dtheta):+.1f}°)"
    )
    if not _confirm_cascade(main_window, object_id, prop_desc, range_str):
        return

    command = CascadeDeltaBBoxCommand(
        object_id=str(object_id),
        frame_start=start_frame,
        frame_end=end_frame,
        dcx=dcx,
        dcy=dcy,
        dw=dw,
        dh=dh,
        dtheta=dtheta,
        annotator=annotator,
    )
    _execute_cascade_command(
        main_window, command, "Applied delta to",
        empty_title="Cascade Delta", complete_title="Cascade Delta Complete",
    )


def _apply_cascade_delta_next_frames(
    main_window,
    object_id: str,
    annotator: str,
    direction: CascadeDirection = CascadeDirection.FORWARDS,
):
    """Ask user for number of frames, then apply the last geometry delta to those frames."""
    last_deltas = getattr(main_window, "last_bbox_deltas", {})
    delta = last_deltas.get(object_id)
    if delta is None:
        QMessageBox.information(
            main_window,
            "Cascade Delta",
            "No recorded adjustment for this object yet.\n"
            "Move, resize or rotate the bbox first (the R-key shortcut records the delta).",
        )
        return

    dcx, dcy, dw, dh, dtheta = delta
    frame_range = _ask_next_frame_range(
        main_window, direction, dialog_title="Cascade Delta", prompt_prefix="Apply delta"
    )
    if frame_range is None:
        return
    start_frame, end_frame = frame_range

    prop_desc = (
        f"delta (Δcx={dcx:+.1f}, Δcy={dcy:+.1f}, "
        f"Δw={dw:+.1f}, Δh={dh:+.1f}, "
        f"Δθ={_math.degrees(dtheta):+.1f}°)"
    )
    if not _confirm_cascade(main_window, object_id, prop_desc, f"{start_frame}–{end_frame}"):
        return

    command = CascadeDeltaBBoxCommand(
        object_id=str(object_id),
        frame_start=start_frame,
        frame_end=end_frame,
        dcx=dcx,
        dcy=dcy,
        dw=dw,
        dh=dh,
        dtheta=dtheta,
        annotator=annotator,
    )
    _execute_cascade_command(
        main_window, command, "Applied delta to",
        empty_title="Cascade Delta", complete_title="Cascade Delta Complete",
    )
