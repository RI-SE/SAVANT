"""Per-box origin history: shared rules (savant_common) and what markit writes."""

from markit.markitlib.postprocessing import DuplicateRemovalPass, GapFillingPass
from markit.markitlib.postprocessing._passes._common import update_housekeeping_annotator
from savant_common.openlabel import Origin, aligned_origins, infer_origin, updated_origins


def _vec(obj):
    return {e["name"]: e["val"] for e in obj["object_data"]["vec"]}


def _box(annotator="markit_yolo", x=100.0, origin=True):
    vec = [{"name": "annotator", "val": [annotator]}, {"name": "confidence", "val": [0.6]}]
    if origin:
        vec.append({"name": "origin", "val": ["detected"]})
    return {"object_data": {"rbbox": [{"name": "shape", "val": [x, 100.0, 50.0, 30.0, 0.0]}], "vec": vec}}


class TestOriginRules:
    def test_legacy_inference(self):
        assert infer_origin("markit_yolo") == Origin.DETECTED
        assert infer_origin("markit_housekeeping(gap)") == Origin.INTERPOLATED
        assert infer_origin("alice") == Origin.MANUAL
        assert aligned_origins(["alice", "markit_yolo"], None) == ["manual", "detected"]

    def test_updates(self):
        assert updated_origins(["markit_yolo"], ["detected"], Origin.MANUAL, prepended=True) == ["manual", "detected"]
        # Same annotator again: newest origin replaced, but a review keeps that they changed the box
        assert updated_origins(["alice"], ["manual"], Origin.MANUAL_TRACKED, prepended=False) == ["manual_tracked"]
        assert updated_origins(["alice"], ["manual"], Origin.REVIEWED, prepended=False) == ["manual"]


class TestMarkitOrigins:
    def test_housekeeping_keeps_origin_aligned(self):
        box = _box()
        update_housekeeping_annotator(box, "rot")
        update_housekeeping_annotator(box, "smooth")  # merged into the same housekeeping entry
        vec = _vec(box)
        assert vec["annotator"] == ["markit_housekeeping(rot,smooth)", "markit_yolo"]
        assert vec["origin"] == ["detected", "detected"]

    def test_housekeeping_on_legacy_box_infers_history(self):
        box = _box(origin=False)
        update_housekeeping_annotator(box, "rot")
        assert _vec(box)["origin"] == ["detected", "detected"]

    def test_gap_filling_marks_interpolated_with_distance(self):
        frames = {str(i): {"objects": {"1": _box(x=100.0 + i)}} for i in (0, 4)}
        data = {"openlabel": {"frames": frames, "objects": {"1": {"name": "1", "type": "car"}}}}
        result = GapFillingPass().process(data)["openlabel"]["frames"]
        filled = {i: _vec(result[str(i)]["objects"]["1"]) for i in (1, 2, 3)}
        assert all(v["origin"] == ["interpolated"] for v in filled.values())
        assert [filled[i]["gap_distance"] for i in (1, 2, 3)] == [[1], [2], [1]]

    def test_frames_taken_over_from_a_duplicate_are_merged(self, monkeypatch):
        frames = {"0": {"objects": {"A": _box(), "B": _box("markit_oflow")}},
                  "1": {"objects": {"B": _box("markit_oflow")}}}
        data = {"openlabel": {"frames": frames, "objects": {k: {"name": k, "type": "car"} for k in "AB"}}}
        dup = DuplicateRemovalPass()
        monkeypatch.setattr(dup, "_are_duplicates", lambda a, b, *args: (True, {}))
        monkeypatch.setattr(dup, "_choose_object_to_delete", lambda a, b, *args: "B")
        moved = dup.process(data)["openlabel"]["frames"]["1"]["objects"]["A"]
        vec = _vec(moved)
        assert vec["origin"][0] == "merged"
        assert vec["annotator"][0].startswith("markit_housekeeping(") and "merge" in vec["annotator"][0]
        assert len(vec["origin"]) == len(vec["annotator"])
