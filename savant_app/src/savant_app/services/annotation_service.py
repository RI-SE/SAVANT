from .project_state import ProjectState


class AnnotationService:
    def __init__(self, project_state: ProjectState) -> None:
        # This is a temporary cache for unsaved annotations.
        # It may hold several annotations for a given user session.
        self.project_state = project_state

        # This is a temporary cache for unsaved annotations.
        # It may hold several annotations for a given user session.
        self.cached_annotations: list[dict] = []

    def create_new_object_bbox(self, frame_number: int, bbox_info: dict) -> None:
        """Handles both, the creation of a new object and adding a bbox for it."""
        # Generate ID
        obj_id = self._generate_new_object_id()
        obj_type = bbox_info["type"]
        coordinates = bbox_info["coordinates"]
        
        # Add new object and bbox
        self._add_new_object(obj_type=obj_type, new_object_id=obj_id)
        self._add_new_object_bbox(
            frame_number=frame_number, bbox_coordinates=coordinates, new_bbox_key=obj_id
        )

    def _add_new_object(self, obj_type: str, new_object_id: str) -> None:
        self.project_state.annotation_config.add_new_object(
            obj_type=obj_type, new_object_id=new_object_id
        )

    # Refactor error handling to use pydantic.
    def _add_new_object_bbox(
        self, frame_number: int, bbox_coordinates: dict, new_bbox_key: str
    ) -> None:
        """
        Service function to add new annotations to the config.
        """
        # Temporary hard code of annotater and confidence score.
        annotater_data = {"val": ["example_name"]}
        confidence_data = {"val": [0.9]}

        # Append new bounding box (under frames)
        self.project_state.annotation_config.append_new_object_bbox(
            frame_id=frame_number,
            bbox_coordinates=bbox_coordinates,
            confidence_data=confidence_data,
            annotater_data=annotater_data,
            new_bbox_key=new_bbox_key,
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
