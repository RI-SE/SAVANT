from tests.unit.test_utils import read_json
from src.savant_app.models.OpenLabel import OpenLabel
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

    def test_get_actor_types_returns_expected_list(self):
        project_state = ProjectState()
        actors = project_state.get_actor_types()
        expected = [
            "RoadUser",
            "Vehicle",
            "Car",
            "Van",
            "Truck",
            "Trailer",
            "Motorbike",
            "Bicycle",
            "Bus",
            "Tram",
            "Train",
            "Caravan",
            "StandupScooter",
            "AgriculturalVehicle",
            "ConstructionVehicle",
            "EmergencyVehicle",
            "SlowMovingVehicle",
            "Human",
            "Pedestrian",
            "WheelChairUser",
            "Animal",
        ]
        assert actors == expected
        assert len(actors) == 21

    def test_actor_list_encapsulation(self):
        project_state = ProjectState()
        original_actors = project_state.get_actor_types()
        original_actors.append("TestActor")

        assert "TestActor" not in project_state.get_actor_types()
