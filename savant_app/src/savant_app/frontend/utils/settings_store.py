# savant_app/frontend/utils/settings_store.py
from __future__ import annotations
from pathlib import Path
from typing import Optional


_ONTOLOGY_PATH: Optional[Path] = None

_ACTION_INTERVAL_OFFSET: int = 0

_DEFAULT_ONTOLOGY_NAMESPACE = "http://savant.ri.se/ontology#"

_ONTOLOGY_NAMESPACE: str = _DEFAULT_ONTOLOGY_NAMESPACE

_WARNING_RANGE: tuple[float, float] = (0.4, 0.6)
_ERROR_RANGE: tuple[float, float] = (0.0, 0.4)
_SHOW_WARNINGS: bool = False
_SHOW_ERRORS: bool = False


def get_ontology_path() -> Optional[Path]:
    """
    Return the current Turtle (.ttl) ontology path used for frame tags.
    """
    return _ONTOLOGY_PATH


def set_ontology_path(path: str | Path) -> None:
    """
    Update the ontology (.ttl) file path used for frame tags.

    Raises:
        ValueError: If the file is invalid or not a .ttl.
    """
    global _ONTOLOGY_PATH
    p = Path(path)
    if not p.is_file() or p.suffix.lower() != ".ttl":
        raise ValueError(f"Invalid ontology file: {path}")
    _ONTOLOGY_PATH = p


def get_action_interval_offset() -> int:
    """
    Return the default action interval offset (in frames).
    """
    return int(_ACTION_INTERVAL_OFFSET)


def set_action_interval_offset(value: int) -> None:
    """
    Update the default action interval offset (in frames).

    Args:
        value: Non-negative integer number of frames.

    Raises:
        ValueError: If value is negative.
    """
    global _ACTION_INTERVAL_OFFSET
    interval = int(value)
    if interval < 0:
        raise ValueError("Action interval offset must be >= 0.")
    _ACTION_INTERVAL_OFFSET = interval


def get_ontology_namespace() -> str:
    """
    Return the base namespace IRI for the ontology.
    """
    return _ONTOLOGY_NAMESPACE


def set_ontology_namespace(ns: str) -> None:
    """
    Set the base namespace IRI for the ontology.

    Args:
        ns: A valid namespace URI ending with '#', '/' or ':'.

    Raises:
        ValueError: If empty or doesn’t end with an allowed delimiter.
    """
    global _ONTOLOGY_NAMESPACE
    ns = str(ns).strip()
    if not ns:
        raise ValueError("Ontology namespace cannot be empty.")
    if not (ns.endswith("#") or ns.endswith("/") or ns.endswith(":")):
        raise ValueError(f"Ontology namespace '{ns}' must end with '#', '/' or ':'.")
    _ONTOLOGY_NAMESPACE = ns


def get_warning_range() -> tuple[float, float]:
    return tuple(_WARNING_RANGE)


def get_error_range() -> tuple[float, float]:
    return tuple(_ERROR_RANGE)


def set_threshold_ranges(
    *,
    warning_range: tuple[float, float],
    error_range: tuple[float, float],
    show_warnings: bool,
    show_errors: bool,
) -> None:
    global _WARNING_RANGE, _ERROR_RANGE

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

    _WARNING_RANGE = (warn_min, warn_max)
    _ERROR_RANGE = (err_min, err_max)


def get_show_warnings() -> bool:
    return bool(_SHOW_WARNINGS)


def set_show_warnings(value: bool) -> None:
    global _SHOW_WARNINGS
    _SHOW_WARNINGS = bool(value)


def get_show_errors() -> bool:
    return bool(_SHOW_ERRORS)


def set_show_errors(value: bool) -> None:
    global _SHOW_ERRORS
    _SHOW_ERRORS = bool(value)
