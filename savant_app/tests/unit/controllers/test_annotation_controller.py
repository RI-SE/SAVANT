import pytest
from unittest.mock import MagicMock
from savant_app.controllers.annotation_controller import AnnotationController
from savant_app.services.annotation_service import AnnotationService
from savant_app.services.exceptions import ObjectNotFoundError


class TestAnnotationController:
    @pytest.fixture
    def mock_annotation_service(self):
        """Fixture for a mocked AnnotationService instance"""
        return MagicMock(spec=AnnotationService)

    @pytest.fixture
    def annotation_controller(self, mock_annotation_service):
        """Fixture for AnnotationController with mocked service"""
        return AnnotationController(annotation_service=mock_annotation_service)

    def test_create_new_object_bbox_calls_service_method(
        self, annotation_controller, mock_annotation_service
    ):
        """Test that controller delegates to service method"""
        frame_number = 42
        bbox_info = {
            "object_type": "car",
            "coordinates": {"x": 10, "y": 20, "width": 30, "height": 40},
        }

        annotation_controller.create_new_object_bbox(frame_number, bbox_info)

        # Verify service method is called with correct parameters
        mock_annotation_service.create_new_object_bbox.assert_called_once_with(
            frame_number=frame_number,
            obj_type=bbox_info["object_type"],
            coordinates=bbox_info["coordinates"],
        )

    def test_get_active_objects_calls_service_method(
        self, annotation_controller, mock_annotation_service
    ):
        """Test that controller delegates to service method"""
        frame_number = 10
        mock_annotation_service.get_active_objects.return_value = [
            {"type": "car", "name": "car_1"},
            {"type": "person", "name": "person_1"},
        ]

        result = annotation_controller.get_active_objects(frame_number)

        # Verify service method is called and result is returned
        mock_annotation_service.get_active_objects.assert_called_once_with(frame_number)
        assert result == [
            {"type": "car", "name": "car_1"},
            {"type": "person", "name": "person_1"},
        ]

    def test_create_bbox_existing_object_calls_service_method(
        self, annotation_controller, mock_annotation_service
    ):
        """Test that controller delegates to service method"""
        frame_number = 42
        bbox_info = {
            "object_type": "car",
            "coordinates": {"x": 10, "y": 20, "width": 30, "height": 40},
            "object_id": "car_123",
        }

        annotation_controller.create_bbox_existing_object(frame_number, bbox_info)

        # Verify service method is called with correct parameters
        mock_annotation_service.create_existing_object_bbox.assert_called_once_with(
            frame_number=frame_number,
            coordinates=bbox_info["coordinates"],
            object_name=bbox_info["object_id"],
        )

    def test_create_bbox_existing_object_propagates_errors(
        self, annotation_controller, mock_annotation_service
    ):
        """Test that controller propagates service errors"""
        frame_number = 42
        bbox_info = {
            "object_type": "car",
            "coordinates": {"x": 10, "y": 20, "width": 30, "height": 40},
            "object_id": "invalid_id",
        }
        mock_annotation_service.create_existing_object_bbox.side_effect = (
            ObjectNotFoundError("Error")
        )

        with pytest.raises(ObjectNotFoundError):
            annotation_controller.create_bbox_existing_object(frame_number, bbox_info)
