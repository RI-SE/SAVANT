from savant_app.services.annotation_service import AnnotationService


class AnnotationController:
    def __init__(self, annotation_service: AnnotationService) -> None:
        self.annotation_service = annotation_service

    def get_actor_types(self) -> list[str]:
        """Get the list of all possible actor types."""
        return self.annotation_service.get_actor_types()
