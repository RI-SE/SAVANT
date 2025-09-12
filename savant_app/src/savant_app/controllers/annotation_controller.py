from savant_app.services.annotation_service import AnnotationService


class AnnotationController:
    def __init__(self, annotation_service: AnnotationService) -> None:
        self.annotation_service = annotation_service

    def get_actor_types(self) -> list[str]:
        """Get the list of all possible actor types."""
        return self.annotation_service.get_actor_types()

    def create_new_object_bbox(self, frame_number: int, bbox_info: dict) -> None:
        self.annotation_service.create_new_object_bbox(
            frame_number=frame_number, bbox_info=bbox_info
        )

    def get_active_objects(self, frame_number: int) -> list[str]:
        """Get a list of active objects for the given frame number."""
        return self.annotation_service.get_active_objects(frame_number)
