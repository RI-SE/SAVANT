# edit/frontend/utils/_annotation_spline_ops.py
"""Handler for spline-based angle interpolation from the context menu."""

from __future__ import annotations

from PyQt6.QtWidgets import QMessageBox

from edit.frontend.utils.undo import SplineAngleCommand
from edit.frontend.widgets.spline_angle_dialog import SplineAngleDialog

from ._annotation_helpers import _refresh_after_annotation_change


def _open_spline_angle_dialog(main_window, frontend_state, obj_id: str) -> None:
    """Open SplineAngleDialog and execute the command on confirmation."""
    annotator = None
    if frontend_state is not None:
        annotator = frontend_state.require_current_annotator()
    if not annotator:
        return

    total_frames = int(main_window.project_state_controller.get_frame_count())
    current_frame = int(main_window.video_controller.current_index())

    def _on_apply(object_id, smoothing_factor, start_frame, end_frame):
        command = SplineAngleCommand(
            object_id=object_id,
            smoothing_factor=smoothing_factor,
            annotator=annotator,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        try:
            main_window.execute_undoable_command(command)
        except Exception as exc:
            QMessageBox.warning(
                main_window,
                "Spline Interpolation Failed",
                str(exc),
            )
            return
        _refresh_after_annotation_change(main_window)

    dialog = SplineAngleDialog(
        parent=main_window,
        object_id=obj_id,
        total_frames=total_frames,
        current_frame=current_frame,
        on_apply=_on_apply,
    )
    dialog.exec()
