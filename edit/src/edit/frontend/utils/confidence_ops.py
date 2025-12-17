from __future__ import annotations

from edit.frontend.utils import render
from edit.frontend.utils.settings_store import (
    get_enabled_tag_frames,
    get_error_range,
    get_show_errors,
    get_show_warnings,
    get_tag_options,
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
    tag_frame_map = get_enabled_tag_frames()
    extra_warning_frames = sorted(
        set(tag_frame_map.get("frame", [])) | set(tag_frame_map.get("object", []))
    )
    tag_markers_active = bool(extra_warning_frames)
    if tag_markers_active:
        warning_frames = sorted(set(warning_frames) | set(extra_warning_frames))

    seek_bar.set_warning_frames(warning_frames)
    seek_bar.set_error_frames(error_frames)
    show_warnings = get_show_warnings()
    show_errors = get_show_errors()
    seek_bar.set_warning_visibility(show_warnings or tag_markers_active)
    seek_bar.set_error_visibility(show_errors)

    overlay.set_warning_flag_visibility(show_warnings)
    overlay.set_error_flag_visibility(show_errors)

    if playback_controls is not None and hasattr(
        playback_controls, "set_issue_navigation_visible"
    ):
        playback_controls.set_issue_navigation_visible(
            show_warnings or show_errors or tag_markers_active
        )
    update_issue_info(main_window)


def update_issue_info(main_window) -> None:
    """Update the playback controls with current frame issue information."""
    state = getattr(main_window, "state", None)
    playback_controls = getattr(main_window, "playback_controls", None)
    video_controller = getattr(main_window, "video_controller", None)
    if state is None or playback_controls is None or video_controller is None:
        return
    try:
        frame_idx = int(video_controller.current_index())
    except Exception:
        frame_idx = 0

    entries: list[dict] = []
    issues_map = state.confidence_issues()
    for issue in issues_map.get(frame_idx, []):
        severity = getattr(issue, "severity", "")
        type_label = "Error" if severity == "error" else "Warning"
        confidence = getattr(issue, "confidence", None)
        if isinstance(confidence, (int, float)):
            info_text = f"Confidence {confidence:.3f}"
        else:
            info_text = "Confidence unavailable"
        entries.append(
            {
                "type": type_label,
                "object": str(getattr(issue, "object_id", "Unknown")),
                "info": info_text,
            }
        )

    tag_options = get_tag_options()
    tag_details = state.frame_tag_details_for(frame_idx)
    for detail in tag_details:
        category = detail.get("category")
        tag_name = detail.get("tag_name")
        if category == "frame_tag":
            continue
        option_key = "object"
        if not tag_options.get(option_key, {}).get(tag_name, False):
            continue
        type_label = "Object Tag"
        object_label = detail.get("object_id") or detail.get("object_name") or "Unknown"
        info_text = detail.get("description") or tag_name
        entries.append(
            {
                "type": type_label,
                "object": object_label,
                "info": info_text,
            }
        )

    playback_controls.display_issue_details(entries)
