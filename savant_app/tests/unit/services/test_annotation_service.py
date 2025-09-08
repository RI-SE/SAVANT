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
        
        # Initialize mock objects
        state.annotation_config.objects = MagicMock()
        state.annotation_config.frames = {}
        return state

    @pytest.fixture
    def annotation_service(self, mock_project_state):
        """Fixture for AnnotationService with mocked dependencies"""
        return AnnotationService(project_state=mock_project_state)

    @pytest.mark.parametrize(
        "frame_number, bbox_info",
        [
            (0, {"type": "person", "coordinates": {"x": 10, "y": 20, "width": 30, "height": 40}}),
            (100, {"type": "vehicle", "coordinates": {"x": 50, "y": 60, "width": 70, "height": 80}}),
        ],
    )
    def test_create_new_object_bbox_valid_input(
        self, annotation_service, mock_project_state, frame_number, bbox_info
    ):
        """Test create_new_object_bbox with valid inputs"""
        # Mock object ID generation
        mock_project_state.annotation_config.objects.keys.return_value = ["1", "2", "3"]
        
        annotation_service.create_new_object_bbox(frame_number, bbox_info)

        # Verify config methods were called
        mock_project_state.annotation_config.add_new_object.assert_called_once()
        mock_project_state.annotation_config.append_new_object_bbox.assert_called_once()

    def test_get_active_objects_empty_frame(self, annotation_service, mock_project_state):
        """Test get_active_objects with no active objects"""
        frame_number = 0
        mock_frame = MagicMock()
        mock_frame.objects = {}
        mock_project_state.annotation_config.frames = {str(frame_number): mock_frame}
        
        result = annotation_service.get_active_objects(frame_number)
        assert result == []

    def test_get_active_objects_with_objects(self, annotation_service, mock_project_state):
        """Test get_active_objects with active objects"""
        frame_number = 10
        mock_frame = MagicMock()
        mock_frame.objects = {"1": None, "2": None}  # Keys represent active objects
        mock_project_state.annotation_config.frames = {str(frame_number): mock_frame}
        
        # Create proper mock objects
        mock_obj1 = MagicMock()
        mock_obj1.type = "car"
        mock_obj1.name = "car_1"
        mock_obj2 = MagicMock()
        mock_obj2.type = "person"
        mock_obj2.name = "person_1"
        
        # Configure objects dictionary
        mock_project_state.annotation_config.objects = {
            "1": mock_obj1,
            "2": mock_obj2
        }
        
        result = annotation_service.get_active_objects(frame_number)
        expected = [
            {"type": "car", "name": "car_1"},
            {"type": "person", "name": "person_1"}
        ]
        assert result == expected

    def test_get_active_objects_invalid_frame(self, annotation_service, mock_project_state):
        """Test get_active_objects with non-existent frame"""
        frame_number = 999
        mock_project_state.annotation_config.frames = {}
        
        result = annotation_service.get_active_objects(frame_number)
        assert result == []
