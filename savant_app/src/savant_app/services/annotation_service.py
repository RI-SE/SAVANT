from .project_state import ProjectState
from .exceptions import (
    ObjectInFrameError,
    ObjectNotFoundError,
    FrameNotFoundError,
    InvalidFrameRangeError,
    InvalidInputError,
    BBoxNotFoundError,
)
from typing import Optional, Union
from savant_app.models.OpenLabel import OpenLabel, RotatedBBox
from savant_app.models.OpenLabel import FrameLevelObject
from pydantic import ValidationError


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
        try: 
            # Generate ID
            obj_id = self._generate_new_object_id()

            # Add new object and bbox
            self._add_new_object(obj_type=obj_type, obj_id=obj_id)
            self._add_object_bbox(
                frame_number=frame_number, bbox_coordinates=coordinates, obj_id=obj_id
            )
        except ValidationError as e:
            errors = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
            raise InvalidInputError(f"Invalid input data: {errors}", e)

    def create_existing_object_bbox(
        self, frame_number: int, coordinates: tuple, object_name: str
    ) -> None:
        """Handles adding a bbox for an existing object."""

        # verify the object exists
        #if not self._does_object_exist(object_name):
        #    raise ObjectNotFoundError(f"Object: {object_name} does not exist.")
        try: 
            object_id = self._get_objectid_by_name(object_name)

            # Verify if current frame already has the object
            if self._does_object_exist_in_frame(frame_number, object_id):
                raise ObjectInFrameError(
                    f"Object ID {object_name} already has a bbox in frame {frame_number}."
                )

            self._add_object_bbox(
                frame_number=frame_number, bbox_coordinates=coordinates, obj_id=object_id
            )
        except ValidationError as e:
            errors = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
            raise InvalidInputError(f"Invalid input data: {errors}", e)


    def get_active_objects(self, frame_number: int) -> list[dict]:
        """Get a list of active objects for the given frame number.

        Note, filtering of active objects is done at the service level,
        due to being related to the context of the application,
        whereas the model handles what the data is and manipulating itself.
        """
        # Get frame object keys
        frame = self.project_state.annotation_config.frames[str(frame_number)]
        active_object_keys = frame.objects.keys()

        return [
            {"type": obj.type, "name": obj.name}
            for key, obj in self.project_state.annotation_config.objects.items()
            if key in active_object_keys
        ]

    def get_frame_objects(self, frame_limit: int, current_frame: int) -> list[str]:
        """
        Get a list of all objects with bboxes in the frame range between the
        current frame and frame_limit.
        """
        frames = self.project_state.annotation_config.frames
        global_objects = self.project_state.annotation_config.objects

        if frame_limit < 0 or current_frame < 0:
            raise InvalidFrameRangeError("Frame numbers must be non-negative.")

        start_frame = max(0, current_frame - frame_limit)

        # Convert to string keys for consistency
        frame_keys = [str(k) for k in range(start_frame, current_frame + 1)]
        frame_subset = {k: frames[k] for k in frame_keys if k in frames}

        object_ids = set()
        for frame_data in frame_subset.values():  # Use new variable name
            object_ids.update(frame_data.objects.keys())  # Use update() for set

        # Correct list comprehension:
        return [
            global_objects[obj_id].name
            for obj_id in object_ids
            if obj_id in global_objects
        ]

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
        try:
            return openlabel_model.get_bbox(
                frame_key=frame_key,
                object_key=object_key,
                bbox_index=bbox_index,
            )
        except KeyError as e:
            raise BBoxNotFoundError(f"BBox not found for object {object_key} in frame {frame_key}.", e)

    def move_resize_bbox(
        self,
        frame_key: Union[int, str],
        object_key: Union[int, str],
        *,
        bbox_index: int = 0,
        # absolute (optional)
        x_center: Optional[float] = None,
        y_center: Optional[float] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
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
        try:
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
        except ValidationError as e:
            raise InvalidInputError(f"Invalid input data: {[err["msg"] for err in e.errors()]}", e)

    def delete_bbox(
        self, frame_key: int, object_key: str
    ) -> Optional[FrameLevelObject]:
        """No try-except here; let the controller handle exceptions."""

        if not self.project_state.annotation_config:
            return None
        return self.project_state.annotation_config.delete_bbox(frame_key, object_key)

    def restore_bbox(
        self, frame_key: int, object_key: str, frame_obj: FrameLevelObject
    ) -> None:
        if not self.project_state.annotation_config:
            return
        self.project_state.annotation_config.restore_bbox(
            frame_key, object_key, frame_obj
        )


    def _add_new_object(self, obj_type: str, obj_id: str) -> None:
        self.project_state.annotation_config.add_new_object(
            obj_type=obj_type, obj_id=obj_id
        )

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
        raise ObjectNotFoundError(f"Object: {object_name}, does not exist.")

    def _generate_new_object_id(self) -> str:
        return str(
            int(list(self.project_state.annotation_config.objects.keys())[-1]) + 1
        )

    

   