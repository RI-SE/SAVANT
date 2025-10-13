# savant_app/frontend/utils/ontology_utils.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from rdflib import Graph, RDFS, Namespace
from savant_app.frontend.utils.settings_store import (
    get_ontology_path,
    get_ontology_namespace,
)


CATEGORY_ACTION = "Action"
CATEGORY_DYNAMIC = "DynamicObject"
CATEGORY_STATIC = "StaticObject"

ontology_cache_key: Optional[Tuple[str, float]] = None
labels_by_category_cache: Optional[Dict[str, List[str]]] = None


def _read_text(path: Path) -> str:
    """Read file as UTF-8 (ignore errors)."""
    return path.read_text(encoding="utf-8", errors="ignore")


def _parse_ontology_labels(ttl_text: str) -> Dict[str, List[str]]:
    graph = Graph()
    graph.parse(data=ttl_text, format="turtle")

    ontology_ns = Namespace(get_ontology_namespace())

    def labels_under(root_local: str) -> List[str]:
        root = ontology_ns[root_local]
        seen, out = set(), []

        def visit(cls):
            for child in graph.subjects(RDFS.subClassOf, cls):
                if child in seen:
                    continue
                seen.add(child)
                lbl = graph.value(child, RDFS.label)
                if lbl:
                    out.append(str(lbl))
                visit(child)

        visit(root)
        return sorted({s.lower(): s for s in out}.values(), key=str.lower)

    return {
        CATEGORY_ACTION: labels_under(CATEGORY_ACTION),
        CATEGORY_DYNAMIC: labels_under(CATEGORY_DYNAMIC),
        CATEGORY_STATIC: labels_under(CATEGORY_STATIC),
    }


def _get_ontology_path() -> Path:
    """
    Return the ontology path from Settings. Raises if not set or missing.

    The ontology path is always controlled by Settings (default or user choice).
    """
    path = Path(str(get_ontology_path())).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Ontology file not found: {path}")
    return path


def _load_labels_by_category() -> Dict[str, List[str]]:
    """
    Internal loader with caching (path + mtime). Returns categorized labels.

    The ontology path always comes from Settings (get_ontology_path).
    """
    global ontology_cache_key, labels_by_category_cache

    ontology_path = Path(str(_get_ontology_path())).resolve()
    if not ontology_path.is_file():
        raise FileNotFoundError(f"Ontology file not found: {ontology_path}")

    ontology_modified_time = ontology_path.stat().st_mtime
    cache_key = (str(ontology_path), float(ontology_modified_time))

    if ontology_cache_key == cache_key and labels_by_category_cache is not None:
        return labels_by_category_cache

    ttl_text = _read_text(ontology_path)
    categorized = _parse_ontology_labels(ttl_text)
    ontology_cache_key, labels_by_category_cache = cache_key, categorized
    return categorized


def get_action_labels(ontology_path: Optional[Path] = None) -> List[str]:
    """
    Return the list of 'Action' class labels from the ontology.
    """
    return _load_labels_by_category().get(CATEGORY_ACTION, [])


def get_bbox_type_labels(ontology_path: Optional[Path] = None) -> Dict[str, List[str]]:
    """
    Return bbox type labels grouped by ontology category.

    Keys:
      - "DynamicObject": labels for subclasses (transitively) of :DynamicObject
      - "StaticObject" : labels for subclasses (transitively) of :StaticObject
    """
    labels_by_category = _load_labels_by_category()
    return {
        CATEGORY_DYNAMIC: labels_by_category.get(CATEGORY_DYNAMIC, []),
        CATEGORY_STATIC: labels_by_category.get(CATEGORY_STATIC, []),
    }
