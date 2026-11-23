"""Key-frame review: jump between key frames and confirm that every box in a frame is checked.

Confirmed frames carry a review entry in their frame_properties and serve as ground truth for box errors
(trainit build-residuals).
"""

import logging

from PyQt6.QtGui import QCursor, QKeySequence, QShortcut
from PyQt6.QtWidgets import QToolTip

from edit.frontend.utils.settings_store import get_keyframe_interval
from edit.frontend.utils.undo import ConfirmFrameCommand

from ._annotation_helpers import _refresh_after_annotation_change
from .navigation import on_seek

logger = logging.getLogger(__name__)


def wire(main_window, frontend_state) -> None:
    """K confirms the current frame; '.' and ',' jump to the next/previous key frame."""
    QShortcut(QKeySequence("K"), main_window, activated=lambda: confirm_current_frame(main_window, frontend_state))
    QShortcut(QKeySequence("."), main_window, activated=lambda: jump_to_keyframe(main_window, +1))
    QShortcut(QKeySequence(","), main_window, activated=lambda: jump_to_keyframe(main_window, -1))


def next_keyframe(current: int, interval: int, direction: int, frame_count: int):
    """Nearest multiple of interval strictly after (direction > 0) or before current, or None."""
    target = (current // interval + 1) * interval if direction > 0 else ((current - 1) // interval) * interval
    return target if 0 <= target < frame_count else None


def jump_to_keyframe(main_window, direction: int) -> None:
    try:
        current = int(main_window.video_controller.current_index())
        frame_count = int(main_window.project_state_controller.get_frame_count())
    except Exception:
        logger.warning("Could not read playback position to jump to key frame", exc_info=True)
        return
    target = next_keyframe(current, get_keyframe_interval(), direction, frame_count)
    if target is not None:
        on_seek(main_window, target)


def confirm_current_frame(main_window, frontend_state) -> None:
    annotator = frontend_state.require_current_annotator()
    if not annotator:
        return
    frame = int(main_window.video_controller.current_index())
    command = ConfirmFrameCommand(frame_number=frame, annotator=annotator)
    main_window.execute_undoable_command(command)
    _refresh_after_annotation_change(main_window)
    confirmed = main_window.annotation_controller.reviewed_frames() or []
    QToolTip.showText(
        QCursor.pos(),
        f"Frame {frame} confirmed ({command.marked} boxes marked reviewed); {len(confirmed)} frames confirmed",
    )
