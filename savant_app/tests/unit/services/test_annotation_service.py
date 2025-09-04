import pytest
from unittest.mock import MagicMock
from src.savant_app.services.annotation_service import AnnotationService
from src.savant_app.services.project_state import ProjectState
from src.savant_app.models.OpenLabel import OpenLabel


class TestAnnotationService:
    @pytest.fixture
    def mock_project_state(self):
        """Fixture for a mocked ProjectState instance"""
        state = MagicMock(spec=ProjectState)
        state.annotation_config = MagicMock(spec=OpenLabel)
        return state

    @pytest.fixture
    def annotation_service(self, mock_project_state):
        """Fixture for AnnotationService with mocked dependencies"""
        return AnnotationService(project_state=mock_project_state)

    def test_add_new_object_calls_config(self, annotation_service, mock_project_state):
        """Test that add_new_object calls the config with correct object type"""
        obj_type = "car"
        annotation_service.add_new_object(obj_type)
        mock_project_state.annotation_config.add_new_object.assert_called_once_with(
            obj_type=obj_type
        )

    @pytest.mark.parametrize(
        "frame_number, bbox_info",
        [
            (0, {"x": 10, "y": 20, "width": 30, "height": 40, "type": "person"}),
            (100, {"x": 50, "y": 60, "width": 70, "height": 80, "type": "vehicle"}),
        ],
    )
    def test_add_new_object_bbox_valid_input(
        self, annotation_service, mock_project_state, frame_number, bbox_info
    ):
        """Test add_new_object_bbox with valid inputs"""
        annotation_service.add_new_object_bbox(frame_number, bbox_info)

        # Verify config method was called with expected arguments
        mock_project_state.annotation_config.append_new_object_bbox.assert_called_once()
        call_args = (
            mock_project_state.annotation_config.append_new_object_bbox.call_args[1]
        )

        assert call_args["frame_id"] == frame_number
        assert call_args["bbox_info"] == bbox_info
        assert call_args["confidence_data"]["val"] == [0.9]
        assert call_args["annotater_data"]["val"] == ["example_name"]
