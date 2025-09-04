from .project_state import ProjectState


class AnnotationService:
    def __init__(self, project_state: ProjectState) -> None:
        # This is a temporary cache for unsaved annotations.
        # It may hold several annotations for a given user session.
        self.project_state = project_state

        # This is a temporary cache for unsaved annotations.
        # It may hold several annotations for a given user session.
        self.cached_annotations: list[dict] = []

    def add_new_object(self, obj_type: str):
        self.project_state.annotation_config.add_new_object(obj_type=obj_type)

    # Refactor error handling to use pydantic.
    def add_new_object_bbox(self, frame_number: int, bbox_info: dict) -> None:
        """
        Service function to add new annotations to the config.
        """
        # Temporary hard code of annotater and confidence score.
        annotater_data = {"val": ["example_name"]}
        confidence_data = {"val": [0.9]}

        # Append new bounding box (under frames)
        self.project_state.annotation_config.append_new_object_bbox(
            frame_id=frame_number,
            bbox_info=bbox_info,
            confidence_data=confidence_data,
            annotater_data=annotater_data,
        )
