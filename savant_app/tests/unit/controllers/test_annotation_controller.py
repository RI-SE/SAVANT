import pytest
from unittest.mock import MagicMock, patch
from savant_app.controllers.annotation_controller import AnnotationController
from savant_app.services.annotation_service import AnnotationService

class TestAnnotationController:
    @pytest.fixture
    def mock_annotation_service(self):
        """Fixture for a mocked AnnotationService instance"""
        return MagicMock(spec=AnnotationService)

    @pytest.fixture
    def annotation_controller(self, mock_annotation_service):
        """Fixture for AnnotationController with mocked service"""
        return AnnotationController(annotation_service=mock_annotation_service)

    def test_add_new_object_annotation_calls_service_methods(
        self, 
        annotation_controller, 
        mock_annotation_service
    ):
        """Test that controller properly delegates to service methods"""
        frame_number = 42
        bbox_info = {"x": 10, "y": 20, "width": 30, "height": 40, "type": "car"}
        
        annotation_controller.add_new_object_annotation(frame_number, bbox_info)
        
        # Verify service methods were called with correct parameters
        mock_annotation_service.add_new_object.assert_called_once_with(obj_type="car")
        mock_annotation_service.add_new_object_bbox.assert_called_once_with(
            frame_number=frame_number, 
            bbox_info=bbox_info
        )