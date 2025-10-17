# savant_app/frontend/utils/settings_store.py
from __future__ import annotations
from pathlib import Path
from savant_app.frontend.utils.assets import asset_path


def _find_default_ontology() -> Path:
    """Return path to default ontology file in assets"""
    path = Path(asset_path("savant_ontology_1.0.0.ttl"))
    return path


_DEFAULT_ONTOLOGY = _find_default_ontology()

_ONTOLOGY_PATH: Path = _DEFAULT_ONTOLOGY

_ACTION_INTERVAL_OFFSET: int = 0

_DEFAULT_ONTOLOGY_NAMESPACE = "http://savant.ri.se/ontology#"

_ONTOLOGY_NAMESPACE: str = _DEFAULT_ONTOLOGY_NAMESPACE


def get_ontology_path() -> Path:
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
