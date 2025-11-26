# savant_app/frontend/utils/settings_store.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

_ontology_path: Optional[Path] = None
_action_interval_offset: int = 0
_default_ontology_namespace = "http://savant.ri.se/ontology#"
_ontology_namespace: str = _default_ontology_namespace
_warning_range: tuple[float, float] = (0.4, 0.6)
_error_range: tuple[float, float] = (0.0, 0.4)
_show_warnings: bool = False
_show_errors: bool = False
_tag_options: dict[str, dict[str, bool]] = {"frame": {}, "object": {}}
_tag_frames: dict[str, dict[str, list[int]]] = {"frame": {}, "object": {}}

# New movement sensitivity setting
_movement_sensitivity: float = 1.0  # default 1.0x
_rotation_sensitivity: float = 0.1  # default 0.1


def get_ontology_path() -> Optional[Path]:
    return _ontology_path


def set_ontology_path(path: str | Path) -> None:
    global _ontology_path
    p = Path(path)
    if not p.is_file() or p.suffix.lower() != ".ttl":
        raise ValueError(f"Invalid ontology file: {path}")
    _ontology_path = p


def get_action_interval_offset() -> int:
    return int(_action_interval_offset)


def set_action_interval_offset(value: int) -> None:
    global _action_interval_offset
    interval = int(value)
    if interval < 0:
        raise ValueError("Action interval offset must be >= 0.")
    _action_interval_offset = interval


def get_ontology_namespace() -> str:
    return _ontology_namespace


def set_ontology_namespace(ns: str) -> None:
    global _ontology_namespace
    ns = str(ns).strip()
    if not ns:
        raise ValueError("Ontology namespace cannot be empty.")
    if not (ns.endswith("#") or ns.endswith("/") or ns.endswith(":")):
        raise ValueError(f"Ontology namespace '{ns}' must end with '#', '/' or ':'.")
    _ontology_namespace = ns


def get_warning_range() -> tuple[float, float]:
    return tuple(_warning_range)


def get_error_range() -> tuple[float, float]:
    return tuple(_error_range)


def set_threshold_ranges(
    *,
    warning_range: tuple[float, float],
    error_range: tuple[float, float],
    show_warnings: bool,
    show_errors: bool,
) -> None:
    global _warning_range, _error_range

    warn_min, warn_max = (float(warning_range[0]), float(warning_range[1]))
    err_min, err_max = (float(error_range[0]), float(error_range[1]))

    for label, minimum, maximum in (
        ("Warning", warn_min, warn_max),
        ("Error", err_min, err_max),
    ):
        if minimum < 0.0 or maximum > 1.0:
            raise ValueError(f"{label} range values must be between 0.0 and 1.0.")
        if minimum > maximum:
            raise ValueError(f"{label} range minimum cannot exceed its maximum.")

    if show_warnings and show_errors:
        overlaps = not (warn_max <= err_min or err_max <= warn_min)
        if overlaps:
            raise ValueError(
                "Warning and error ranges must not overlap when both markers are visible."
            )

    _warning_range = (warn_min, warn_max)
    _error_range = (err_min, err_max)


def get_show_warnings() -> bool:
    return bool(_show_warnings)


def set_show_warnings(value: bool) -> None:
    global _show_warnings
    _show_warnings = bool(value)


def get_show_errors() -> bool:
    return bool(_show_errors)


def set_show_errors(value: bool) -> None:
    global _show_errors
    _show_errors = bool(value)


# --- Movement sensitivity getters/setters ---


def get_movement_sensitivity() -> float:
    """Return the current movement sensitivity."""
    return float(_movement_sensitivity)


def set_movement_sensitivity(value: float) -> None:
    """
    Update movement sensitivity.
    """
    global _movement_sensitivity
    float_value = float(value)

    _movement_sensitivity = float_value


def get_rotation_sensitivity() -> float:
    """Return the current rotation senstivity."""
    return float(_rotation_sensitivity)


def set_rotation_sensitivity(value: float) -> None:
    """Update rotation sensitivty."""
    global _rotation_sensitivity
    float_value = float(value)

    _rotation_sensitivity = float_value


def update_tag_options(tag_data: dict[str, dict[str, Iterable[int]]]) -> None:
    """Update the available tag options discovered from the project."""
    global _tag_options, _tag_frames
    new_options: dict[str, dict[str, bool]] = {"frame": {}, "object": {}}
    new_frames: dict[str, dict[str, list[int]]] = {"frame": {}, "object": {}}

    for category, entries in (tag_data or {}).items():
        normalized_entries: dict[str, list[int]] = {}
        for name, frames in (entries or {}).items():
            if not isinstance(name, str):
                name = str(name)
            clean_name = name.strip()
            if not clean_name:
                continue

            if frames is None:
                normalized_entries[clean_name] = []
                continue

            try:
                iterator = list(frames)
            except TypeError:
                iterator = [frames]

            cleaned = sorted(
                {
                    int(value)
                    for value in iterator
                    if isinstance(value, (int, float)) and int(value) >= 0
                }
            )
            normalized_entries[clean_name] = cleaned

        existing_states = _tag_options.get(category, {})
        new_options[category] = {
            name: existing_states.get(name, False)
            for name in sorted(normalized_entries.keys(), key=lambda n: n.lower())
        }
        new_frames[category] = normalized_entries

    _tag_options = new_options
    _tag_frames = new_frames


def get_tag_options() -> dict[str, dict[str, bool]]:
    """Return a copy of the current tag options."""
    return {category: dict(options) for category, options in _tag_options.items()}


def set_tag_option_states(states: dict[str, dict[str, bool]]) -> None:
    """Persist checkbox selections for known tags."""
    global _tag_options
    for category, category_states in (states or {}).items():
        if category not in _tag_options:
            continue
        for name, enabled in category_states.items():
            if name in _tag_options[category]:
                _tag_options[category][name] = bool(enabled)


def get_enabled_tag_frames() -> dict[str, list[int]]:
    """Return enabled frames derived from tag selections."""
    result: dict[str, list[int]] = {"frame": [], "object": []}
    for category in ("frame", "object"):
        frames: set[int] = set()
        options = _tag_options.get(category, {})
        frame_map = _tag_frames.get(category, {})
        for name, enabled in options.items():
            if not enabled:
                continue
            frames.update(frame_map.get(name, []))
        result[category] = sorted(frames)
    return result
