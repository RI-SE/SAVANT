"""Shared dataprov helpers (savant_common.provenance) and per-box origin counts."""

import json
from collections import Counter

from dataprov import ProvenanceChain

from savant_common.openlabel import UNKNOWN_ORIGIN, OpenLabel, newest_origin
from savant_common.provenance import load_or_create_chain, origin_properties
from savant_common.quality import origin_shares


def _add_step(chain, tmp_path, custom_properties):
    (tmp_path / "out.txt").write_text("y")
    chain.add(started_at="2026-01-01T00:00:00Z", ended_at="2026-01-01T00:00:01Z", tool_name="t", tool_version="1",
              operation="o", inputs=[str(tmp_path / "in.txt")], input_formats=["TXT"],
              outputs=[str(tmp_path / "out.txt")], output_formats=["TXT"], custom_properties=custom_properties)


def test_existing_chain_gets_savant_and_su_prefixes(tmp_path):
    (tmp_path / "in.txt").write_text("x")
    path = tmp_path / "chain.json"
    ProvenanceChain.create(entity_id="e", initial_source=str(tmp_path / "in.txt"), description="d").save(str(path))

    chain = load_or_create_chain(path, entity_id="e", initial_source="in.txt", description="d")
    _add_step(chain, tmp_path, {"activity:savant:x": 1, **origin_properties(Counter(detected=1))})
    chain.save(str(path))

    assert {"savant", "su"} <= set(json.loads(path.read_text())["prefix"])


def test_origin_shares():
    shares = origin_shares(Counter(detected=5, interpolated=2, manual_interpolated=1, manual=1, reviewed=1))
    assert shares["share_interpolated"]["value"] == 0.3 and shares["share_interpolated"]["n"] == 10
    assert shares["share_reviewed"]["value"] == 0.2
    assert origin_shares(Counter())["share_reviewed"]["value"] is None


class _Entry:
    def __init__(self, name, val):
        self.name, self.val = name, val


def test_newest_origin_uses_recorded_then_inferred_history():
    assert newest_origin([_Entry("annotator", ["alice"]), _Entry("origin", ["reviewed"])]) == "reviewed"
    assert newest_origin([_Entry("annotator", ["markit_housekeeping(gap)", "markit_yolo"])]) == "interpolated"
    assert newest_origin([_Entry("confidence", [0.5])]) == UNKNOWN_ORIGIN


def test_origin_counts_per_frame():
    def box(annotator):
        return {"object_data": {"rbbox": [{"name": "shape", "val": [10, 10, 4, 4, 0]}],
                                "vec": [{"name": "annotator", "val": [annotator]}]}}
    openlabel = OpenLabel(metadata={"schema_version": "1.1"}, ontologies={}, objects={},
                          frames={"0": {"objects": {"1": box("markit_yolo"), "2": box("alice")}},
                                  "1": {"objects": {"1": box("markit_yolo")}}})
    assert openlabel.origin_counts() == Counter(detected=2, manual=1)
    assert openlabel.origin_counts([1, 5]) == Counter(detected=1)


def test_human_annotators():
    def box(annotators, origins=None):
        vec = [{"name": "annotator", "val": annotators}]
        if origins:
            vec.append({"name": "origin", "val": origins})
        return {"object_data": {"rbbox": [{"name": "shape", "val": [10, 10, 4, 4, 0]}], "vec": vec}}
    openlabel = OpenLabel(metadata={"schema_version": "1.1"}, ontologies={}, objects={},
                          frames={"0": {"objects": {"1": box(["bob", "markit_yolo"]),
                                                    "2": box(["alice", "markit_yolo"], ["reviewed", "detected"])}}})
    assert openlabel.human_annotators() == ["alice", "bob"]


def test_edit_session_reads_version_annotators_and_origins(tmp_path):
    from savant_common.provenance import edit_session

    box = {"object_data": {"rbbox": [{"name": "shape", "val": [10, 10, 4, 4, 0]}],
                           "vec": [{"name": "annotator", "val": ["alice", "markit_yolo"]},
                                   {"name": "origin", "val": ["manual", "detected"]}]}}
    openlabel = {"metadata": {"schema_version": "1.1", "savant_edit": {"version": "2.5.0"}}, "ontologies": {},
                 "objects": {}, "frames": {"0": {"objects": {"1": box}}}}
    path = tmp_path / "An.json"
    path.write_text(json.dumps({"openlabel": openlabel}))

    session = edit_session(path)
    assert session["tool_version"] == "2.5.0" and session["annotators"] == ["alice"]
    assert session["custom_properties"]["activity:su:share_reviewed"]["value"] == 1.0


def test_markit_provenance_roles(tmp_path):
    """Decision log recorded as a log output next to the primary OpenLABEL."""
    from dataprov import ProvenanceChain

    from savant_common.provenance import ROLE_LOG, ROLE_PRIMARY

    (tmp_path / "in.txt").write_text("x")
    chain = load_or_create_chain(tmp_path / "chain.json", entity_id="e", initial_source=str(tmp_path / "in.txt"),
                                 description="d")
    for name in ("out.json", "decisions.json"):
        (tmp_path / name).write_text("{}")
    chain.add(started_at="2026-01-01T00:00:00Z", ended_at="2026-01-01T00:00:01Z", tool_name="t", tool_version="1",
              operation="o", inputs=[str(tmp_path / "in.txt")], outputs=[str(tmp_path / "out.json"),
              str(tmp_path / "decisions.json")], output_roles=[ROLE_PRIMARY, ROLE_LOG])
    chain.save(str(tmp_path / "chain.json"))
    loaded = ProvenanceChain.load(str(tmp_path / "chain.json"))
    assert [o["path"] for o in loaded.get_outputs(role=ROLE_LOG)] == [str(tmp_path / "decisions.json")]
