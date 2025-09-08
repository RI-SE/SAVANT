from savant_app.services.annotation_service import AnnotationService
from savant_app.models.OpenLabel import RotatedBBox


class AnnotationController:
    def __init__(self, annotation_service: AnnotationService) -> None:
        self.annotation_service = annotation_service

    def get_actor_types(self) -> list[str]:
        """Get the list of all possible actor types."""
        return self.annotation_service.get_actor_types()

    def add_new_object_annotation(self, frame_number: int, bbox_info: dict) -> None:
        self.annotation_service.add_new_object(obj_type=bbox_info["type"])
        self.annotation_service.add_new_object_bbox(
            frame_number=frame_number, bbox_info=bbox_info
        )

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

    def move_resize_bbox(
        self,
        frame_key: int | str,
        object_key: int | str,
        *,
        bbox_index: int = 0,
        x_center: float | None = None,
        y_center: float | None = None,
        width:    float | None = None,
        height:   float | None = None,
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
