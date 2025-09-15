from .project_state import ProjectState
from .exceptions import ObjectInFrameError, ObjectNotFoundError, FrameNotFoundError


class AnnotationService:
    def __init__(self, project_state: ProjectState) -> None:
        # This is a temporary cache for unsaved annotations.
        # It may hold several annotations for a given user session.
        self.project_state = project_state

        # This is a temporary cache for unsaved annotations.
        # It may hold several annotations for a given user session.
        self.cached_annotations: list[dict] = []

    def create_new_object_bbox(
        self, frame_number: int, obj_type: str, coordinates: tuple
    ) -> None:
        """Handles both, the creation of a new object and adding a bbox for it."""
        # Generate ID
        obj_id = self._generate_new_object_id()

        # Add new object and bbox
        self._add_new_object(obj_type=obj_type, obj_id=obj_id)
        self._add_object_bbox(
            frame_number=frame_number, bbox_coordinates=coordinates, obj_id=obj_id
        )

    def _add_new_object(self, obj_type: str, obj_id: str) -> None:
        self.project_state.annotation_config.add_new_object(
            obj_type=obj_type, obj_id=obj_id
        )

    # Refactor error handling to use pydantic.
    def _add_object_bbox(
        self, frame_number: int, bbox_coordinates: dict, obj_id: str
    ) -> None:
        """
        Service function to add new annotations to the config.
        """
        # Temporary hard code of annotater and confidence score.
        annotater_data = {"val": ["example_name"]}
        confidence_data = {"val": [0.9]}

        # Append new bounding box (under frames)
        self.project_state.annotation_config.append_object_bbox(
            frame_id=frame_number,
            bbox_coordinates=bbox_coordinates,
            confidence_data=confidence_data,
            annotater_data=annotater_data,
            obj_id=obj_id,
        )

    def _does_object_exist(self, object_name: str) -> bool:
        """Check if an object exists in the annotation config."""
        existing_ids = [
            value.name
            for _, value in self.project_state.annotation_config.objects.items()
        ]
        return object_name in existing_ids

    def _does_object_exist_in_frame(self, frame_number: int, object_id: str) -> bool:
        """Check if an object exists in a specific frame."""
        try:
            frame = self.project_state.annotation_config.frames[str(frame_number)]
        except KeyError:
            raise FrameNotFoundError(f"Frame number {frame_number} does not exist.")

        return object_id in [key for key in frame.objects.keys()]

    def _get_objectid_by_name(self, object_name: str):
        """Get the object ID given the object name."""
        for key, value in self.project_state.annotation_config.objects.items():
            if value.name == object_name:
                return key
        return None

    def create_existing_object_bbox(
        self, frame_number: int, obj_type: str, coordinates: tuple, object_name: str
    ) -> None:
        """Handles adding a bbox for an existing object."""

        # verify the object exists
        # TODO: Refactor error handling
        if not self._does_object_exist(object_name):
            raise ObjectNotFoundError(f"Object: {object_name} does not exist.")

        object_id = self._get_objectid_by_name(object_name)
        # Verify if current frame already has the object
        if self._does_object_exist_in_frame(frame_number, object_id):
            raise ObjectInFrameError(
                f"Object ID {object_name} already has a bbox in frame {frame_number}."
            )

        self._add_object_bbox(
            frame_number=frame_number, bbox_coordinates=coordinates, obj_id=object_id
        )

    def get_active_objects(self, frame_number: int) -> list[dict]:
        """Get a list of active objects for the given frame number.

        Note, filtering of active objects is done at the service level,
        due to being related to the context of the application,
        whereas the model handles what the data is and manipulating itself.
        """
        try:
            # Get frame object keys
            frame = self.project_state.annotation_config.frames[str(frame_number)]
            active_object_keys = frame.objects.keys()
        except KeyError:
            return []

        return [
            {"type": obj.type, "name": obj.name}
            for key, obj in self.project_state.annotation_config.objects.items()
            if key in active_object_keys
        ]

    def _generate_new_object_id(self) -> str:
        return str(
            int(list(self.project_state.annotation_config.objects.keys())[-1]) + 1
        )
