from tests.unit.test_utils import read_json
from src.savant_app.models.OpenLabel import (
    ActionMetadata,
    FrameInterval,
    OpenLabel,
    ObjectMetadataData,
    ObjectMetadataVecEntry,
)
from src.savant_app.services.project_state import ProjectState
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
