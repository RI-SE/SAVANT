from savant_app.services.annotation_service import AnnotationService


class AnnotationController:
    def __init__(self, annotation_service: AnnotationService) -> None:
        self.annotation_service = annotation_service

    def get_actor_types(self) -> list[str]:
        """Get the list of all possible actor types."""
        return self.annotation_service.get_actor_types()
    
    # TODO: Refactor so add new object is called as a 
    # result of adding a new box.
    def add_new_object(self):
        print("new object")

    def add_annotation(self, frame_number: int, coordinates: tuple) -> None:
        print("frame number: ", frame_number)
        print("coordinates: ", coordinates)