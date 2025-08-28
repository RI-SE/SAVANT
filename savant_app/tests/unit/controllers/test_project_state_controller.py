import pytest
from unittest.mock import create_autospec
from savant_app.controllers.project_state_controller import ProjectStateController
from savant_app.services.project_state import ProjectState


@pytest.fixture
def mock_service():
    return create_autospec(ProjectState)


@pytest.fixture
def controller(mock_service):
    return ProjectStateController(mock_service)


class TestProjectStateController:
    def test_load_openlabel_config_calls_service(self, controller, mock_service):
        test_path = "/test/path.json"
        controller.load_openlabel_config(test_path)
        mock_service.load_openlabel_config.assert_called_once_with(test_path)

    def test_load_openlabel_config_propagates_errors(self, controller, mock_service):
        mock_service.load_openlabel_config.side_effect = FileNotFoundError
        with pytest.raises(FileNotFoundError):
            controller.load_openlabel_config("/invalid/path.json")

    def test_save_openlabel_config_calls_service(self, controller, mock_service):
        controller.save_openlabel_config()
        mock_service.save_openlabel_config.assert_called_once()

    def test_get_actor_types_returns_service_value(self, controller, mock_service):
        expected_actors = ["Car", "Pedestrian"]
        mock_service.get_actor_types.return_value = expected_actors
        assert controller.get_actor_types() == expected_actors
        mock_service.get_actor_types.assert_called_once()

    def test_service_error_handling(self, controller, mock_service):
        mock_service.save_openlabel_config.side_effect = ValueError("Invalid config")
        with pytest.raises(ValueError) as exc_info:
            controller.save_openlabel_config()
        assert "Invalid config" in str(exc_info.value)
