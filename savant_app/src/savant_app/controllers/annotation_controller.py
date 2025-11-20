from typing import Optional, Union

from savant_app.models.OpenLabel import FrameLevelObject, RotatedBBox
from savant_app.services.annotation_service import AnnotationService

from .error_handler_middleware import error_handler


class AnnotationController:
    def __init__(self, annotation_service: AnnotationService) -> None:
        self.annotation_service = annotation_service

    @error_handler
    def get_actor_types(self) -> list[str]:
        """Get the list of all possible actor types."""
        return self.annotation_service.get_actor_types()

    @error_handler
    def create_new_object_bbox(
        self, frame_number: int, bbox_info: dict, annotator: str
    ) -> None:
        self.annotation_service.create_new_object_bbox(
            frame_number=frame_number,
            obj_type=bbox_info["object_type"],
            coordinates=bbox_info["coordinates"],
            annotator=annotator,
        )

    @error_handler
    def get_bbox(
        self,
        frame_key: int | str,
        object_key: int | str,
        bbox_index: int = 0,
    ) -> RotatedBBox:
        """UI-level read of a bbox."""
        return self.annotation_service.get_bbox(
            frame_key=frame_key,
            object_key=object_key,
            bbox_index=bbox_index,
        )

    @error_handler
    def move_resize_bbox(
        self,
        frame_key: Union[int, str],
        object_key: Union[int, str],
        *,
        bbox_index: int = 0,
        x_center: Optional[float] = None,
        y_center: Optional[float] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        rotation: Optional[float] = None,
        delta_x: float = 0.0,
        delta_y: float = 0.0,
        delta_w: float = 0.0,
        delta_h: float = 0.0,
        delta_theta: float = 0.0,
        min_width: float = 1e-6,
        min_height: float = 1e-6,
        annotator: str
    ) -> RotatedBBox:
        """
        UI-level update for bbox geometry; delegates to the service.
        Handles cascade mode if enabled.
        """

        # Update current frame
        return self.annotation_service.move_resize_bbox(
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
            annotator=annotator,
        )

    @error_handler
    def cascade_bbox_edit(
        self,
        frame_start: int,
        object_key: Union[int, str],
        frame_end: Optional[int],
        annotator: str,
        width: Optional[float] = None,
        height: Optional[float] = None,
        rotation: Optional[float] = None,
    ) -> Optional[RotatedBBox]:
        """
        Cascade resize/rotation to all frames containing the object starting from frame_start.
        """
        return self.annotation_service.cascade_bbox_edit(
            frame_start=int(frame_start),  # Start from next frame
            frame_end=frame_end,
            annotator=annotator,
            object_key=object_key,
            width=width,
            height=height,
            rotation=rotation,
        )

    @error_handler
    def create_bbox_existing_object(
        self, frame_number: int, bbox_info: dict, annotator: str
    ) -> None:
        self.annotation_service.create_existing_object_bbox(
            frame_number=frame_number,
            coordinates=bbox_info["coordinates"],
            object_name=bbox_info["object_id"],
            annotator=annotator,
        )

    @error_handler
    def get_active_objects(self, frame_number: int) -> list[dict]:
        """Get a list of active objects for the given frame number."""
        return self.annotation_service.get_active_objects(frame_number)

    @error_handler
    def get_frame_object_ids(self, frame_limit: int, current_frame: int) -> list[str]:
        """
        Get a list of all objects with bboxes in the frame range between the
        current frame and frame_limit.
        """
        return self.annotation_service.get_frame_objects(
            frame_limit=frame_limit, current_frame=current_frame
        )

    @error_handler
    def delete_bbox(
        self, frame_key: int, object_key: str
    ) -> Optional[FrameLevelObject]:
        return self.annotation_service.delete_bbox(frame_key, object_key)

    @error_handler
    def restore_bbox(
        self, frame_key: int, object_key: str, frame_obj: FrameLevelObject
    ) -> None:
        self.annotation_service.restore_bbox(frame_key, object_key, frame_obj)

    @error_handler
    def list_object_ids(self) -> list[str]:
        return self.annotation_service.list_object_ids()

    @error_handler
    def frames_for_object(self, object_id: str) -> list[int]:
        return self.annotation_service.frames_for_object(object_id)

    @error_handler
    def link_object_ids(
        self,
        primary_object_id: str,
        secondary_object_id: str,
    ) -> list[int]:
        return self.annotation_service.link_object_ids(
            primary_object_id,
            secondary_object_id,
        )

    @error_handler
    def mark_confidence_resolved(
        self, frame_number: int, object_id: str, annotator: str
    ) -> None:
        self.annotation_service.mark_confidence_resolved(
            frame_number=frame_number, object_id=object_id, annotator=annotator
        )

    @error_handler
    def allowed_frame_tags(self) -> list[str]:
        """Return the frame tag names from the service."""
        return self.annotation_service.get_frame_tags()

    @error_handler
    def add_frame_tag(self, tag_name: str, frame_start: int, frame_end: int) -> None:
        """
        Add a new frame tag interval via the annotation service.

        Args:
            tag_name: Name of the tag.
            frame_start: Index of the start frame.
            frame_end: Index of the end frame.
        """
        self.annotation_service.add_frame_tag(tag_name, frame_start, frame_end)

    @error_handler
    def active_frame_tags(self, frame_index: int) -> list[tuple[str, int, int]]:
        """
        Fetch frame-tag intervals that are active at the given frame.

        Args:
            frame_index.

        Returns:
            List of (tag_name, start, end) tuples.
        """
        return self.annotation_service.get_active_frame_tags(frame_index)

    @error_handler
    def allowed_bbox_types(self) -> dict[str, list[str]]:
        """
        Return bbox type labels grouped as DynamicObject / StaticObject.
        """
        return self.annotation_service.bbox_types()

    @error_handler
    def remove_frame_tag(self, tag_name: str, frame_start: int, frame_end: int) -> bool:
        return self.annotation_service.remove_frame_tag(
            tag_name, frame_start, frame_end
        )

    @error_handler
    def delete_bboxes_by_object(
        self, object_key: str
    ) -> list[tuple[int, FrameLevelObject]]:
        return self.annotation_service.delete_bboxes_by_object(object_key)

    @error_handler
    def get_object_metadata(self, object_id: str) -> dict:
        return self.annotation_service.get_object_metadata(object_id)

    @error_handler
    def update_object_name(self, object_id: str, new_name: str) -> None:
        self.annotation_service.update_object_name(object_id, new_name)

    @error_handler
    def update_object_type(self, object_id: str, new_type: str) -> None:
        self.annotation_service.update_object_type(object_id, new_type)

    @error_handler
    def interpolate_annotations(
        self,
        object_id: str,
        start_frame: int,
        end_frame: int,
        # control_points: Dict[str, List],
        annotator: str,
    ) -> None:
        """Create interpolated annotations between two keyframes"""
        self.annotation_service.interpolate_annotations(
            object_id, start_frame, end_frame, annotator
        )

    @error_handler
    def add_object_relationship(
        self,
        relationship_name: str,
        relationship_type: str,
        ontology_uid: str,
        subject_object_id: str,
        object_object_id: str,
    ):
        self.annotation_service.add_object_relationship(
            relationship_name,
            relationship_type,
            ontology_uid,
            subject_object_id,
            object_object_id,
        )
