from savant_app.services.annotation_service import AnnotationService
from savant_app.models.OpenLabel import RotatedBBox
from typing import Optional
from savant_app.models.OpenLabel import FrameLevelObject
from .error_handler_middleware import error_handler


class AnnotationController:
    def __init__(self, annotation_service: AnnotationService) -> None:
        self.annotation_service = annotation_service

    @error_handler
    def get_actor_types(self) -> list[str]:
        """Get the list of all possible actor types."""
        return self.annotation_service.get_actor_types()

    @error_handler
    def create_new_object_bbox(self, frame_number: int, bbox_info: dict) -> None:
        self.annotation_service.create_new_object_bbox(
            frame_number=frame_number,
            obj_type=bbox_info["object_type"],
            coordinates=bbox_info["coordinates"],
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
        frame_key: int | str,
        object_key: int | str,
        *,
        bbox_index: int = 0,
        x_center: float | None = None,
        y_center: float | None = None,
        width: float | None = None,
        height: float | None = None,
        rotation: float | None = None,
        delta_x: float = 0.0,
        delta_y: float = 0.0,
        delta_w: float = 0.0,
        delta_h: float = 0.0,
        delta_theta: float = 0.0,
        min_width: float = 1e-6,
        min_height: float = 1e-6,
    ) -> RotatedBBox:
        """
        UI-level update for bbox geometry; delegates to the service.
        """
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
        )

    @error_handler
    def create_bbox_existing_object(self, frame_number: int, bbox_info: dict) -> None:
        self.annotation_service.create_existing_object_bbox(
            frame_number=frame_number,
            coordinates=bbox_info["coordinates"],
            object_name=bbox_info["object_id"],
        )

    @error_handler
    def get_active_objects(self, frame_number: int) -> list[str]:
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

    def allowed_frame_tags(self) -> list[str]:
        """Return the frame tag names from the service."""
        return self.annotation_service.get_frame_tags()

    def add_frame_tag(self, tag_name: str, frame_start: int, frame_end: int) -> None:
        """
        Add a new frame tag interval via the annotation service.

        Args:
            tag_name: Name of the tag.
            frame_start: Index of the start frame.
            frame_end: Index of the end frame.
        """
        self.annotation_service.add_frame_tag(tag_name, frame_start, frame_end)

    def active_frame_tags(self, frame_index: int) -> list[tuple[str, int, int]]:
        """
        Fetch frame-tag intervals that are active at the given frame.

        Args:
            frame_index.

        Returns:
            List of (tag_name, start, end) tuples.
        """
        return self.annotation_service.get_active_frame_tags(frame_index)

    def allowed_bbox_types(self) -> dict[str, list[str]]:
        """
        Return bbox type labels grouped as DynamicObject / StaticObject.
        """
        return self.annotation_service.bbox_types()
