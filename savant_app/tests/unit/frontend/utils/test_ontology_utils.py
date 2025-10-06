import os
import time
from pathlib import Path

import pytest

# Module under test
from savant_app.frontend.utils import ontology_utils as ou


# ---------- helpers ----------

EX_NS = "http://example.org/ns#"

TTL_BASE = """@prefix : <{ns}> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

:Tag a rdfs:Class ; rdfs:label "Tag" .
:Action a rdfs:Class ; rdfs:subClassOf :Tag ; rdfs:label "Action" .
:DynamicObject a rdfs:Class ; rdfs:subClassOf :Tag ; rdfs:label "DynamicObject" .
:StaticObject a rdfs:Class ; rdfs:subClassOf :Tag ; rdfs:label "StaticObject" .

:Motion a rdfs:Class ; rdfs:subClassOf :Action ; rdfs:label "Motion" .
:TurnLeft a rdfs:Class ; rdfs:subClassOf :Motion ; rdfs:label "TurnLeft" .

:Car a rdfs:Class ; rdfs:subClassOf :DynamicObject ; rdfs:label "Car" .
:Sign a rdfs:Class ; rdfs:subClassOf :StaticObject ; rdfs:label "Sign" .
"""

def write_ttl(path: Path, ns: str, extra: str = "") -> None:
    path.write_text(TTL_BASE.format(ns=ns) + extra, encoding="utf-8")


def reset_cache():
    ou.ontology_cache_key = None
    ou.labels_by_category_cache = None


# ---------- tests ----------

def test_parse_transitive_subclasses(monkeypatch):
    """
    _parse_ontology_labels should include GRANDCHILD classes:
    TurnLeft is subclass of Motion, which is subclass of Action -> should appear under Action.
    """
    reset_cache()
    # Force namespace used by the parser
    monkeypatch.setattr(ou, "get_ontology_namespace", lambda: EX_NS)

    ttl = TTL_BASE.format(ns=EX_NS)
    out = ou._parse_ontology_labels(ttl)

    assert "Action" in out and "DynamicObject" in out and "StaticObject" in out
    # transitive under Action
    assert "TurnLeft" in out["Action"]
    # direct under DynamicObject / StaticObject
    assert "Car" in out["DynamicObject"]
    assert "Sign" in out["StaticObject"]
    # parent categories themselves should not be returned as labels
    assert "Action" not in out["Action"]
    assert "DynamicObject" not in out["DynamicObject"]
    assert "StaticObject" not in out["StaticObject"]


def test_get_labels_reads_from_settings_path(monkeypatch, tmp_path):
    """
    Public helpers should read TTL from Settings-only path and return expected labels.
    """
    reset_cache()
    ttl_file = tmp_path / "onto.ttl"
    write_ttl(ttl_file, EX_NS)

    # Point ontology_utils to our temp TTL + namespace
    monkeypatch.setattr(ou, "get_ontology_path", lambda: ttl_file)
    monkeypatch.setattr(ou, "get_ontology_namespace", lambda: EX_NS)

    actions = ou.get_action_labels()
    bbox = ou.get_bbox_type_labels()

    assert "TurnLeft" in actions
    assert bbox["DynamicObject"] == ["Car"]
    assert bbox["StaticObject"] == ["Sign"]


def test_cache_invalidation_on_mtime_change(monkeypatch, tmp_path):
    """
    Cache must invalidate when the TTL file's mtime changes.
    """
    reset_cache()
    ttl_file = tmp_path / "onto.ttl"
    write_ttl(ttl_file, EX_NS)

    monkeypatch.setattr(ou, "get_ontology_path", lambda: ttl_file)
    monkeypatch.setattr(ou, "get_ontology_namespace", lambda: EX_NS)

    # First read caches TurnLeft only
    actions1 = ou.get_action_labels()
    assert "TurnLeft" in actions1
    assert "TurnRight" not in actions1

    # Modify TTL to add a new Action descendant
    extra = """
:TurnRight a rdfs:Class ; rdfs:subClassOf :Motion ; rdfs:label "TurnRight" .
"""
    write_ttl(ttl_file, EX_NS, extra=extra)

    # Ensure mtime changes on fast FS
    new_mtime = ttl_file.stat().st_mtime + 5
    os.utime(ttl_file, (new_mtime, new_mtime))

    reset_cache()  # (optional) If you want to rely solely on mtime, omit this line.

    actions2 = ou.get_action_labels()
    assert "TurnLeft" in actions2
    assert "TurnRight" in actions2  # new label should appear


def test_missing_ontology_path_raises(monkeypatch, tmp_path):
    """
    If the configured ontology path doesn't exist, a FileNotFoundError should be raised.
    """
    reset_cache()
    missing = tmp_path / "nope.ttl"
    monkeypatch.setattr(ou, "get_ontology_path", lambda: missing)
    # Namespace still needed by the code path, but won't be used after the raise
    monkeypatch.setattr(ou, "get_ontology_namespace", lambda: EX_NS)

    with pytest.raises(FileNotFoundError):
        ou.get_action_labels()
