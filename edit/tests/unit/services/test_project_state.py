from tests.unit.test_utils import read_json
from src.edit.models.OpenLabel import (
    ActionMetadata,
    FrameInterval,
    OpenLabel,
    ObjectMetadataData,
    ObjectMetadataVecEntry,
)
from src.edit.services.project_state import ProjectState
from pathlib import Path
import shutil


class TestProjectState:

    def setup_method(self):
        """Setup method to backup the original file before each test."""
        self.test_config_path = str(
            Path(__file__).parent.parent.parent / "assets" / "Kraklanda_short.json"
        )
        self.backup_path = self.test_config_path + ".backup"
        # Create a backup of the original file
        shutil.copy2(self.test_config_path, self.backup_path)

    def teardown_method(self):
        """Teardown method to restore the original file after each test."""
        # Restore the original file from backup
        shutil.move(self.backup_path, self.test_config_path)

    def test_load_config(self):

        project_state = ProjectState()
        test_config_path = str(
            Path(__file__).parent.parent.parent / "assets" / "Kraklanda_short.json"
        )

        project_state.load_openlabel_config(test_config_path)

        expected_data_dict = read_json(test_config_path)
        expected_result = OpenLabel(**expected_data_dict["openlabel"])

        assert (
            expected_result.model_dump() == project_state.annotation_config.model_dump()
        )

    def test_save_config(self):

        project_state = ProjectState()
        test_config_path = str(
            Path(__file__).parent.parent.parent / "assets" / "Kraklanda_short.json"
        )

        project_state.load_openlabel_config(test_config_path)

        project_state.annotation_config.ontologies["0"] = "test"

        project_state.save_openlabel_config()

        # Verify that the new key-value pair is saved correctly
        saved_config = read_json(test_config_path)
        assert saved_config["openlabel"]["ontologies"]["0"] == "test"

    def test_load_save_load_flow(self):
        project_state = ProjectState()
        test_config_path = str(
            Path(__file__).parent.parent.parent / "assets" / "Kraklanda_short.json"
        )

        project_state.load_openlabel_config(test_config_path)

        project_state.annotation_config.ontologies["0"] = "test"

        project_state.save_openlabel_config()

        # Create a new ProjectState instance and load the config
        new_project_state = ProjectState()
        new_project_state.load_openlabel_config(test_config_path)

        # Verify that the changes are persisted
        assert new_project_state.annotation_config.ontologies["0"] == "test"

    def test_get_tag_categories(self):
        project_state = ProjectState()
        test_config_path = str(
            Path(__file__).parent.parent.parent / "assets" / "Kraklanda_short.json"
        )

        project_state.load_openlabel_config(test_config_path)
        # Inject object-level metadata tags to ensure detection
        first_obj = next(iter(project_state.annotation_config.objects.values()))
        first_obj.object_data = ObjectMetadataData(
            vec=[ObjectMetadataVecEntry(name="suddendisappear", val=[10])]
        )
        project_state.annotation_config.actions = {
            "0": ActionMetadata(
                name="lanechange",
                type="test",
                frame_intervals=[FrameInterval(frame_start=5, frame_end=8)],
            )
        }
        tags = project_state.get_tag_categories()

        assert tags == {
            "frame": {"lanechange": [5]},
            "object": {"suddendisappear": [10]},
        }

    def test_get_tag_frame_details(self):
        project_state = ProjectState()
        test_config_path = str(
            Path(__file__).parent.parent.parent / "assets" / "Kraklanda_short.json"
        )

        project_state.load_openlabel_config(test_config_path)
        first_obj = next(iter(project_state.annotation_config.objects.values()))
        first_obj.object_data = ObjectMetadataData(
            vec=[ObjectMetadataVecEntry(name="suddenappear", val=[15, 16])]
        )
        project_state.annotation_config.actions = {
            "0": ActionMetadata(
                name="lanechange",
                type="test",
                frame_intervals=[FrameInterval(frame_start=7, frame_end=9)],
            )
        }

        details = project_state.get_tag_frame_details()
        assert 15 in details
        assert any(
            entry["tag_name"] == "suddenappear" and entry["category"] == "object_tag"
            for entry in details[15]
        )
        assert 7 in details
        assert any(
            entry["tag_name"] == "lanechange" and entry["category"] == "frame_tag"
            for entry in details[7]
        )


def test_failed_save_keeps_original_file(tmp_path, monkeypatch):
    import src.edit.services.project_state as project_state_module

    source = Path(__file__).parent.parent.parent / "assets" / "Kraklanda_short.json"
    target = tmp_path / source.name
    shutil.copy2(source, target)
    original = target.read_text()
    project_state = ProjectState()
    project_state.load_openlabel_config(str(target))

    def failing_dump(obj, f, **kwargs):
        f.write("{partial")
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(project_state_module.json, "dump", failing_dump)
    try:
        project_state.save_openlabel_config()
    except RuntimeError:
        pass

    assert target.read_text() == original
    assert [p.name for p in tmp_path.iterdir()] == [target.name]


def test_unsaved_changes_tracks_edits_since_load_and_save(tmp_path):
    source = Path(__file__).parent.parent.parent / "assets" / "Kraklanda_short.json"
    target = tmp_path / source.name
    shutil.copy2(source, target)
    project_state = ProjectState()
    project_state.load_openlabel_config(str(target))
    assert not project_state.has_unsaved_changes()

    project_state.annotation_config.ontologies["0"] = "changed"
    assert project_state.has_unsaved_changes()

    project_state.save_openlabel_config()
    assert not project_state.has_unsaved_changes()


def test_save_rebuilds_intervals_and_drops_boxless_objects(tmp_path):
    from src.edit.models.OpenLabel import RDFItem, RelationMetadata

    source = Path(__file__).parent.parent.parent / "assets" / "Kraklanda_short.json"
    target = tmp_path / source.name
    shutil.copy2(source, target)
    state = ProjectState()
    state.load_openlabel_config(str(target))
    config = state.annotation_config
    orphan, kept = sorted(config.objects)[:2]
    for frame in config.frames.values():
        frame.objects.pop(orphan, None)
    # Remove one middle frame of the kept object to create a gap
    kept_frames = sorted(int(k) for k, f in config.frames.items() if kept in f.objects)
    gap = kept_frames[len(kept_frames) // 2]
    config.frames[str(gap)].objects.pop(kept)
    config.relations = {
        "0": RelationMetadata(name="r", type="near", ontology_uid="0",
                              rdf_subjects=[RDFItem(type="object", uid=orphan)],
                              rdf_objects=[RDFItem(type="object", uid=kept)]),
    }

    state.save_openlabel_config()

    saved = read_json(str(target))["openlabel"]
    assert orphan not in saved["objects"]
    assert saved["relations"] == {}
    intervals = saved["objects"][kept]["frame_intervals"]
    assert {"frame_start": kept_frames[0], "frame_end": gap - 1} in intervals
    assert {"frame_start": gap + 1, "frame_end": kept_frames[-1]} in intervals
    # The in-memory model is untouched, so undo can still restore the orphan's boxes
    assert orphan in state.annotation_config.objects
    assert not state.has_unsaved_changes()


def _copied_asset(tmp_path):
    source = Path(__file__).parent.parent.parent / "assets" / "Kraklanda_short.json"
    target = tmp_path / source.name
    shutil.copy2(source, target)
    return target


def test_save_keeps_metadata_written_by_other_tools(tmp_path):
    import json

    target = _copied_asset(tmp_path)
    data = read_json(str(target))
    data["openlabel"]["metadata"].update(
        {"name": "Kraklanda run", "comment": "markit comment", "tags": ["drone"], "quality": {"spec": "x"}})
    target.write_text(json.dumps(data))
    state = ProjectState()
    state.load_openlabel_config(str(target))

    state.save_openlabel_config()

    saved = read_json(str(target))["openlabel"]["metadata"]
    assert saved["name"] == "Kraklanda run"
    assert saved["comment"] == "markit comment"
    assert saved["tags"] == ["drone"]
    assert saved["quality"] == {"spec": "x"}


def _set_histories(state, annotators, confidences):
    """Give every box the same annotator/confidence histories (newest first)."""
    from collections import deque

    from src.edit.models.OpenLabel import AnnotatorData, ConfidenceData

    for frame in state.annotation_config.frames.values():
        for frame_obj in frame.objects.values():
            frame_obj.object_data.vec = [
                AnnotatorData(val=deque(annotators)),
                ConfidenceData(val=deque(confidences)),
            ]


def _issue_confidences(state):
    issues = state.confidence_issues(warning_range=(0.4, 0.6), error_range=(0.0, 0.4))
    return {issue.confidence for frame_issues in issues.values() for issue in frame_issues}


def test_confidence_warning_sees_detector_score_under_housekeeping(tmp_path):
    state = ProjectState()
    state.load_openlabel_config(str(_copied_asset(tmp_path)))
    _set_histories(state, ["markit_housekeeping(rot)", "markit_yolo"], [0.8888, 0.45])
    assert _issue_confidences(state) == {0.45}


def test_human_edit_resolves_low_detector_score(tmp_path):
    state = ProjectState()
    state.load_openlabel_config(str(_copied_asset(tmp_path)))
    _set_histories(state, ["alice", "markit_housekeeping(rot)", "markit_yolo"], [1.0, 0.8888, 0.45])
    assert _issue_confidences(state) == set()
