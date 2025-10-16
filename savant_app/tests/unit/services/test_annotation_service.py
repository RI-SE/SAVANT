import pytest
from unittest.mock import MagicMock, call
from savant_app.services.annotation_service import AnnotationService
from savant_app.services.project_state import ProjectState
from savant_app.models.OpenLabel import OpenLabel
from savant_app.services.exceptions import ObjectNotFoundError


@pytest.fixture
def mock_project_state():
    """Fixture for a mocked ProjectState instance"""
    state = MagicMock(spec=ProjectState)
    state.annotation_config = MagicMock(spec=OpenLabel)

    # Add video_metadata mock
    state.annotation_config.video_metadata = MagicMock()

    # Initialize mock objects
    state.annotation_config.objects = MagicMock()
    state.annotation_config.frames = {}
    return state


@pytest.fixture
def annotation_service(mock_project_state):
    """Fixture for AnnotationService with mocked dependencies"""
    return AnnotationService(project_state=mock_project_state)


class TestAnnotationService:
    @pytest.mark.parametrize(
        "frame_number, bbox_info",
        [
            (
                0,
                {
                    "type": "person",
                    "coordinates": {"x": 10, "y": 20, "width": 30, "height": 40},
                },
            ),
            (
                100,
                {
                    "type": "vehicle",
                    "coordinates": {"x": 50, "y": 60, "width": 70, "height": 80},
                },
            ),
        ],
    )
    def test_create_new_object_bbox_valid_input(
        self, annotation_service, mock_project_state, frame_number, bbox_info
    ):
        """Test create_new_object_bbox with valid inputs"""
        # Mock object ID generation
        mock_project_state.annotation_config.objects.keys.return_value = ["1", "2", "3"]

        # Extract values from bbox_info and pass as separate arguments
        annotation_service.create_new_object_bbox(
            frame_number, bbox_info["type"], bbox_info["coordinates"]
        )

        # Verify config methods were called
        mock_project_state.annotation_config.add_new_object.assert_called_once()
        mock_project_state.annotation_config.append_object_bbox.assert_called_once()

    def test_get_active_objects_empty_frame(
        self, annotation_service, mock_project_state
    ):
        """Test get_active_objects with no active objects"""
        frame_number = 0
        mock_frame = MagicMock()
        mock_frame.objects = {}
        mock_project_state.annotation_config.frames = {str(frame_number): mock_frame}

        result = annotation_service.get_active_objects(frame_number)
        assert result == []

    def test_get_active_objects_with_objects(
        self, annotation_service, mock_project_state
    ):
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
        mock_project_state.annotation_config.objects = {"1": mock_obj1, "2": mock_obj2}

        result = annotation_service.get_active_objects(frame_number)
        expected = [
            {"type": "car", "name": "car_1"},
            {"type": "person", "name": "person_1"},
        ]
        assert result == expected

    def test_get_active_objects_invalid_frame(
        self, annotation_service, mock_project_state
    ):
        """Test get_active_objects with non-existent frame"""
        frame_number = 999
        mock_project_state.annotation_config.frames = {}

        with pytest.raises(KeyError):
            annotation_service.get_active_objects(frame_number)

    def test_create_existing_object_bbox_valid(
        self, annotation_service, mock_project_state
    ):
        """Test create_existing_object_bbox with valid existing object"""
        frame_number = 42
        coordinates = (10, 20, 30, 40)
        object_id = "1"
        object_name = "car_1"

        # Mock object existence check
        annotation_service._does_object_exist = MagicMock(return_value=True)
        annotation_service._does_object_exist_in_frame = MagicMock(return_value=False)
        annotation_service._get_objectid_by_name = MagicMock(return_value=object_id)

        annotation_service.create_existing_object_bbox(
            frame_number, coordinates, object_name
        )

        # Verify config method was called
        mock_project_state.annotation_config.append_object_bbox.assert_called_once_with(
            frame_id=frame_number,
            bbox_coordinates=coordinates,
            confidence_data={"val": [0.9]},
            annotater_data={"val": ["example_name"]},
            obj_id=object_id,
        )

    def test_create_existing_object_bbox_invalid_object(
        self, annotation_service, mock_project_state
    ):
        """Test create_existing_object_bbox with non-existent object"""
        frame_number = 42
        coordinates = (10, 20, 30, 40)
        object_id = "invalid_id"

        # Mock object existence check
        annotation_service._does_object_exist = MagicMock(return_value=False)

        with pytest.raises(ObjectNotFoundError):
            annotation_service.create_existing_object_bbox(
                frame_number, coordinates, object_id
            )

    def test_does_object_exist_true(self, annotation_service, mock_project_state):
        """Test _does_object_exist returns True for existing object"""
        object_id = "car_123"
        # Setup mock objects with names
        mock_obj1 = MagicMock()
        mock_obj1.name = "Object-1"
        mock_obj2 = MagicMock()
        mock_obj2.name = "car_123"
        mock_project_state.annotation_config.objects = {"1": mock_obj1, "2": mock_obj2}

        assert annotation_service._does_object_exist(object_id) is True

    def test_does_object_exist_false(self, annotation_service, mock_project_state):
        """Test _does_object_exist returns False for non-existent object"""
        object_id = "invalid_id"
        # Setup mock objects with names
        mock_obj1 = MagicMock()
        mock_obj1.name = "Object-1"
        mock_obj2 = MagicMock()
        mock_obj2.name = "car_123"
        mock_project_state.annotation_config.objects = {"1": mock_obj1, "2": mock_obj2}

        assert annotation_service._does_object_exist(object_id) is False


class TestCascadeBboxEdit:
    @pytest.fixture
    def setup_cascade_test(self, mock_project_state):
        """Setup common test environment for cascade_bbox_edit tests"""
        # Configure video metadata with 100 frames
        mock_project_state.annotation_config.video_metadata.frame_count = 100

        # Create frames with test object "obj1" present in frames 10-90
        frames = {}
        for frame_num in range(0, 100, 10):  # Frames 0,10,20,...,90
            frame_key = str(frame_num)
            mock_frame = MagicMock()
            mock_frame.objects = {"obj1": MagicMock()} if frame_num >= 10 else {}
            frames[frame_key] = mock_frame

        mock_project_state.annotation_config.frames = frames
        return mock_project_state

    def test_cascade_bbox_edit_deltas(self, setup_cascade_test, annotation_service):
        """Test applying changes to all frames"""
        mock_project_state = setup_cascade_test
        mock_update = mock_project_state.annotation_config.update_bbox

        # Apply deltas starting from frame 20
        edited_frames = annotation_service.cascade_bbox_edit(
            frame_start=20, frame_end=90, object_key="obj1", width=5, height=-3, rotation=15
        )

        # Verify edited frames (20-90 in increments of 10)
        assert edited_frames == [20, 30, 40, 50, 60, 70, 80, 90]

        # Verify update_bbox called with correct parameters
        expected_calls = [
            call(
                frame_key=str(frame),
                object_key="obj1",
                bbox_index=0,
                width=5,
                height=-3,
                rotation=15,
                min_width=1e-6,
                min_height=1e-6,
            )
            for frame in [20, 30, 40, 50, 60, 70, 80, 90]
        ]
        mock_update.assert_has_calls(expected_calls, any_order=False)

    def test_cascade_bbox_edit_mixed_params(
        self, setup_cascade_test, annotation_service
    ):
        """Test combination of absolute and delta parameters"""
        mock_project_state = setup_cascade_test
        mock_update = mock_project_state.annotation_config.update_bbox

        # Apply mixed parameters starting from frame 40
        edited_frames = annotation_service.cascade_bbox_edit(
            frame_start=40, frame_end=90, object_key="obj1", width=60, height=40, rotation=10
        )

        # Verify edited frames (40–90)
        assert edited_frames == [40, 50, 60, 70, 80, 90]

        # Verify update_bbox called with correct parameters
        expected_calls = [
            call(
                frame_key=str(frame),
                object_key="obj1",
                bbox_index=0,
                width=60,
                height=40,
                rotation=10,
                min_width=1e-6,
                min_height=1e-6,
            )
            for frame in [40, 50, 60, 70, 80, 90]
        ]
        mock_update.assert_has_calls(expected_calls, any_order=False)

    def test_cascade_bbox_edit_object_not_present(
        self, setup_cascade_test, annotation_service
    ):
        """Test when object is not present in some frames"""
        mock_project_state = setup_cascade_test
        mock_update = mock_project_state.annotation_config.update_bbox

        # Should only edit frames where object exists (10-90)
        edited_frames = annotation_service.cascade_bbox_edit(
            frame_start=0,
            frame_end=90,
            object_key="obj1",
        )

        # Verify only frames 10-90 were edited (not frame 0)
        assert edited_frames == [10, 20, 30, 40, 50, 60, 70, 80, 90]
        assert mock_update.call_count == 9  # Frames 10-90

    def test_cascade_bbox_edit_min_constraints(
        self, setup_cascade_test, annotation_service
    ):
        """Test min_width and min_height constraints"""
        mock_project_state = setup_cascade_test
        mock_update = mock_project_state.annotation_config.update_bbox

        # Apply changes with min constraints
        edited_frames = annotation_service.cascade_bbox_edit(
            frame_start=50,
            frame_end=90,
            object_key="obj1",
            width=-100,  # Would make width too small
            min_width=10,
            min_height=10,
        )

        # Verify edited frames
        assert edited_frames == [50, 60, 70, 80, 90]

        # Verify min constraints passed to update_bbox
        expected_calls = [
            call(
                frame_key=str(frame),
                object_key="obj1",
                bbox_index=0,
                width=-100,
                height=None,
                rotation=None,
                min_width=10,
                min_height=10,
            )
            for frame in [50, 60, 70, 80, 90]
        ]
        mock_update.assert_has_calls(expected_calls, any_order=False)
