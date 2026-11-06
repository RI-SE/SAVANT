"""Utilities for persisting per-project configuration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from edit.frontend.exceptions import (
    InvalidActionIntervalOffsetError,
    FileOperationError,
    InvalidFrameHistoryCountError,
    InvalidOntologyNamespaceError,
    InvalidWarningErrorRange,
    InvalidZoomRateError,
)
from edit.frontend.utils.settings_store import (
    get_action_interval_offset,
    get_keyframe_interval,
    get_bookmark_notes,
    get_default_ontology_namespace,
    get_error_range,
    get_frame_history_count,
    get_lock_to_center,
    get_movement_sensitivity,
    get_ontology_namespace,
    get_rotation_sensitivity,
    get_show_errors,
    get_show_warnings,
    get_tag_options,
    get_warning_range,
    get_zoom_rate,
    set_action_interval_offset,
    set_keyframe_interval,
    reset_project_scoped_settings,
    set_bookmarks,
    set_frame_history_count,
    set_lock_to_center,
    set_movement_sensitivity,
    set_ontology_namespace,
    set_rotation_sensitivity,
    set_show_errors,
    set_show_warnings,
    set_tag_option_states,
    set_threshold_ranges,
    set_zoom_rate,
)

PROJECT_CONFIG_FILENAME = "savant_project_config.json"


@dataclass
class ProjectConfig:
    """Simple container for project-specific metadata."""

    settings: Dict[str, Any] = field(default_factory=dict)
    annotators: List[str] = field(default_factory=list)
    last_annotator: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "settings": dict(self.settings or {}),
            "annotators": list(self.annotators or []),
        }
        if self.last_annotator:
            result["last_annotator"] = self.last_annotator
        return result


_active_project_dir: Path | None = None


def set_active_project_dir(project_dir: Path | str | None) -> None:
    """Track the folder that backs the currently open project."""

    global _active_project_dir
    if project_dir is None:
        _active_project_dir = None
        return
    _active_project_dir = Path(project_dir).resolve()


def get_active_project_dir() -> Path | None:
    return _active_project_dir


def _config_path(project_dir: Path) -> Path:
    return Path(project_dir) / PROJECT_CONFIG_FILENAME


def _default_config() -> ProjectConfig:
    """Return a new project's config: current preferences, but nothing tied to another project."""
    settings = {
        **_snapshot_settings(),
        "bookmarks": {},
        "tag_options": {},
        "ontology_namespace": get_default_ontology_namespace(),
    }
    return ProjectConfig(settings=settings, annotators=[])


def _read_config(path: Path) -> ProjectConfig:
    if not path.is_file():
        return _default_config()
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh) or {}
    except (OSError, ValueError) as exc:
        raise FileOperationError(f"Failed to read project config '{path}': {exc}")
    if not isinstance(raw, dict):
        raise FileOperationError(f"Invalid project config '{path}': expected a JSON object")
    settings = raw.get("settings") or {}
    if not isinstance(settings, dict):
        raise FileOperationError(f"Invalid project config '{path}': 'settings' must be an object")
    annotators = raw.get("annotators")
    last_annotator = raw.get("last_annotator", "")
    normalized = ProjectConfig(
        settings=settings,
        annotators=_normalize_annotators(annotators),
        last_annotator=str(last_annotator or "").strip(),
    )
    return normalized


def _write_config(path: Path, config: ProjectConfig) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(config.to_dict(), fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except OSError as exc:
        raise FileOperationError(f"Failed to write project config '{path}': {exc}")


def load_project_config(project_dir: Path | None = None) -> ProjectConfig:
    """Return the config for the provided folder or active project."""

    if project_dir is None:
        project_dir = _active_project_dir
    if project_dir is None:
        return _default_config()
    return _read_config(_config_path(Path(project_dir)))


def ensure_project_config(project_dir: Path) -> ProjectConfig:
    """Ensure the config file exists and return its contents."""

    path = _config_path(project_dir)
    config = _read_config(path)
    _write_config(path, config)
    return config


def _snapshot_settings() -> Dict[str, Any]:
    """Capture the user-adjustable settings that should persist per project."""

    return {
        "zoom_rate": get_zoom_rate(),
        "movement_sensitivity": get_movement_sensitivity(),
        "rotation_sensitivity": get_rotation_sensitivity(),
        "frame_history_count": get_frame_history_count(),
        "ontology_namespace": get_ontology_namespace(),
        "action_interval_offset": get_action_interval_offset(),
        "keyframe_interval": get_keyframe_interval(),
        "tag_options": get_tag_options(),
        "warning_range": tuple(get_warning_range()),
        "show_warnings": get_show_warnings(),
        "error_range": tuple(get_error_range()),
        "show_errors": get_show_errors(),
        "bookmarks": _serialized_bookmarks(),
        "lock_to_center": get_lock_to_center(),
    }


def _serialized_bookmarks() -> Dict[str, str]:
    return {str(k): v for k, v in get_bookmark_notes().items()}


def _serialize_settings(settings: Dict[str, Any]) -> str:
    """Return a JSON string for consistent comparisons."""

    try:
        return json.dumps(settings, sort_keys=True)
    except TypeError:
        # Fallback in case future settings add unsupported types.
        return repr(settings)


def settings_changed_since_last_save(project_dir: Path | None = None) -> bool:
    """Return True if the current in-memory settings differ from the stored config."""

    if project_dir is None:
        project_dir = _active_project_dir
    if project_dir is None:
        return False
    project_dir = Path(project_dir)
    current_settings = _snapshot_settings()
    stored_settings = load_project_config(project_dir).settings or {}
    return _serialize_settings(current_settings) != _serialize_settings(
        dict(stored_settings)
    )


def persist_current_settings(project_dir: Path | None = None) -> ProjectConfig:
    """Persist the in-memory settings to the active project's config file."""

    if project_dir is None:
        project_dir = _active_project_dir
    if project_dir is None:
        return _default_config()
    project_dir = Path(project_dir)
    config = load_project_config(project_dir)
    config.settings = _snapshot_settings()
    _write_config(_config_path(project_dir), config)
    return config


def persist_bookmarks(project_dir: Path | None = None) -> None:
    """Save only the bookmarks; other settings stay behind the 'Save Settings?' prompt."""

    if project_dir is None:
        project_dir = _active_project_dir
    if project_dir is None:
        return
    project_dir = Path(project_dir)
    config = load_project_config(project_dir)
    config.settings["bookmarks"] = _serialized_bookmarks()
    _write_config(_config_path(project_dir), config)


def apply_project_settings(config: ProjectConfig | None) -> None:
    """Apply stored settings to the global settings store."""

    if not isinstance(config, ProjectConfig):
        return
    reset_project_scoped_settings()
    settings = config.settings or {}

    zoom_rate = settings.get("zoom_rate")
    if isinstance(zoom_rate, (int, float)):
        try:
            set_zoom_rate(float(zoom_rate))
        except InvalidZoomRateError:
            pass

    movement = settings.get("movement_sensitivity")
    if isinstance(movement, (int, float)):
        set_movement_sensitivity(float(movement))

    rotation = settings.get("rotation_sensitivity")
    if isinstance(rotation, (int, float)):
        set_rotation_sensitivity(float(rotation))

    frame_history = settings.get("frame_history_count")
    if isinstance(frame_history, (int, float)):
        try:
            set_frame_history_count(int(frame_history))
        except InvalidFrameHistoryCountError:
            pass

    namespace = settings.get("ontology_namespace")
    if isinstance(namespace, str) and namespace.strip():
        try:
            set_ontology_namespace(namespace.strip())
        except InvalidOntologyNamespaceError:
            pass

    keyframe_interval = settings.get("keyframe_interval")
    if isinstance(keyframe_interval, (int, float)) and keyframe_interval >= 1:
        set_keyframe_interval(int(keyframe_interval))

    action_offset = settings.get("action_interval_offset")
    if isinstance(action_offset, (int, float)):
        try:
            set_action_interval_offset(int(action_offset))
        except InvalidActionIntervalOffsetError:
            pass

    warning_range = _tuple_or_none(settings.get("warning_range"))
    error_range = _tuple_or_none(settings.get("error_range"))
    show_warnings = settings.get("show_warnings")
    show_errors = settings.get("show_errors")

    try:
        if warning_range and error_range:
            set_threshold_ranges(
                warning_range=warning_range,
                error_range=error_range,
                show_warnings=(
                    bool(show_warnings)
                    if show_warnings is not None
                    else get_show_warnings()
                ),
                show_errors=(
                    bool(show_errors) if show_errors is not None else get_show_errors()
                ),
            )
    except InvalidWarningErrorRange:
        pass

    if show_warnings is not None:
        set_show_warnings(bool(show_warnings))
    if show_errors is not None:
        set_show_errors(bool(show_errors))

    bookmarks = settings.get("bookmarks")
    if isinstance(bookmarks, (list, dict)):
        set_bookmarks(bookmarks)

    lock_to_center = settings.get("lock_to_center")
    if lock_to_center is not None:
        set_lock_to_center(bool(lock_to_center))


def restore_tag_option_states(project_dir: Path | None = None) -> None:
    """Reapply stored tag checkbox states after options are rebuilt."""

    config = load_project_config(project_dir)
    tag_options = config.settings.get("tag_options") if config.settings else None
    if isinstance(tag_options, dict):
        set_tag_option_states(tag_options)


def record_annotator_login(name: str, project_dir: Path | None = None) -> None:
    """Append the annotator name to the config if it is new."""

    normalized = (name or "").strip()
    if not normalized:
        return
    if project_dir is None:
        project_dir = _active_project_dir
    if project_dir is None:
        return
    project_dir = Path(project_dir)
    config = load_project_config(project_dir)
    annotators = _normalize_annotators(config.annotators)
    lowered = {existing.lower(): idx for idx, existing in enumerate(annotators)}
    if normalized.lower() not in lowered:
        annotators.append(normalized)
        config.annotators = annotators
    config.last_annotator = normalized
    _write_config(_config_path(project_dir), config)


def _normalize_annotators(annotators: Any) -> List[str]:
    result: list[str] = []
    seen: set[str] = set()
    if not isinstance(annotators, list):
        return []
    for entry in annotators:
        if not isinstance(entry, str):
            entry = str(entry or "")
        cleaned = entry.strip()
        lowered = cleaned.lower()
        if not cleaned or lowered in seen:
            continue
        seen.add(lowered)
        result.append(cleaned)
    return result


def _tuple_or_none(value: Any) -> Tuple[float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return (float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            return None
    return None
