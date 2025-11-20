from __future__ import annotations

from savant_app.frontend.utils import render
from savant_app.frontend.utils.settings_store import (
    get_error_range,
    get_show_errors,
    get_show_warnings,
    get_warning_range,
)


def refresh_confidence_issues(main_window) -> None:
    """
    Sync warning/error confidence issues from the backend into the frontend state,
    then propagate the markers to the visible widgets.
    """
    project_state_controller = getattr(main_window, "project_state_controller", None)
    state = getattr(main_window, "state", None)
    if project_state_controller is None or state is None:
        return

    project_state = getattr(project_state_controller, "project_state", None)
    annotation_config = getattr(project_state, "annotation_config", None)
    if not project_state or annotation_config is None:
        state.set_confidence_issues({})
        apply_confidence_markers(main_window)
        return

    try:
        issues = project_state_controller.confidence_issues(
            warning_range=get_warning_range(),
            error_range=get_error_range(),
            show_warnings=get_show_warnings(),
            show_errors=get_show_errors(),
        )
    except Exception:
        issues = {}

    state.set_confidence_issues(issues or {})
    apply_confidence_markers(main_window)

    try:
        render.refresh_frame(main_window)
    except Exception:
        # Rendering can fail while the video controller is initialising; swallow errors.
        pass


def apply_confidence_markers(main_window) -> None:
    """
    Push warning/error marker information from the frontend state to visual widgets.
    """
    state = getattr(main_window, "state", None)
    seek_bar = getattr(main_window, "seek_bar", None)
    overlay = getattr(main_window, "overlay", None)
    playback_controls = getattr(main_window, "playback_controls", None)
    if state is None or seek_bar is None or overlay is None:
        return

    warning_frames = sorted(state.warning_frames())
    error_frames = sorted(state.error_frames())

    seek_bar.set_warning_frames(warning_frames)
    seek_bar.set_error_frames(error_frames)
    show_warnings = get_show_warnings()
    show_errors = get_show_errors()
    seek_bar.set_warning_visibility(show_warnings)
    seek_bar.set_error_visibility(show_errors)

    overlay.set_warning_flag_visibility(show_warnings)
    overlay.set_error_flag_visibility(show_errors)

    if playback_controls is not None and hasattr(
        playback_controls, "set_issue_navigation_visible"
    ):
        playback_controls.set_issue_navigation_visible(show_warnings or show_errors)
