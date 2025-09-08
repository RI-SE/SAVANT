from .project_state import ProjectState
from typing import Optional, Union
from savant_app.models.OpenLabel import OpenLabel, RotatedBBox


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

    def get_bbox(
        self,
        frame_key: Union[int, str],
        object_key: Union[int, str],
        bbox_index: int = 0,
    ) -> RotatedBBox:
        """
        Read a bbox from the current annotation config (model).
        """
        openlabel_model: OpenLabel = self.project_state.annotation_config
        return openlabel_model.get_bbox(
            frame_key=frame_key,
            object_key=object_key,
            bbox_index=bbox_index,
        )

    def move_resize_bbox(
        self,
        frame_key: Union[int, str],
        object_key: Union[int, str],
        *,
        bbox_index: int = 0,
        # absolute (optional)
        x_center: Optional[float] = None,
        y_center: Optional[float] = None,
        width:    Optional[float] = None,
        height:   Optional[float] = None,
        rotation: Optional[float] = None,
        # deltas (optional)
        delta_x: float = 0.0,
        delta_y: float = 0.0,
        delta_w: float = 0.0,
        delta_h: float = 0.0,
        delta_theta: float = 0.0,
        # clamps
        min_width: float = 1e-6,
        min_height: float = 1e-6,
    ) -> RotatedBBox:
        """
        Update bbox geometry in the model; return the updated RotatedBBox.
        """
        openlabel_model: OpenLabel = self.project_state.annotation_config
        updated_bbox = openlabel_model.update_bbox(
            frame_key=frame_key,
            object_key=object_key,
            bbox_index=bbox_index,
            x_center=x_center,
            y_center=y_center,
            width=width,
            height=height,
            rotation=rotation,
            delta_x=delta_x,
            delta_y=delta_y,
            delta_w=delta_w,
            delta_h=delta_h,
            delta_theta=delta_theta,
            min_width=min_width,
            min_height=min_height,
        )
        # place for side-effects, e.g. mark project dirty, log, mirror, etc.
        return updated_bbox
