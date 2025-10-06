"""
Class representing the controller to the project state.

The purpose of this controller is to facilitate frontend<->project_state
communication while maintaining separation of concerns, and a decoupled frontend.
"""

from savant_app.services.project_state import ProjectState


class ProjectStateController:
    # TODO: ERROR HANDLING: Catch service level errors, make them UI friendly.
    def __init__(self, project_state: ProjectState):
        self.project_state = project_state

    def load_openlabel_config(self, path: str) -> None:
        self.project_state.load_openlabel_config(path)

    def save_openlabel_config(self) -> None:
        self.project_state.save_openlabel_config()

    def get_actor_types(self) -> list[str]:
        return self.project_state.get_actor_types()

    def boxes_for_frame(self, frame_idx: int):
        return self.project_state.boxes_for_frame(frame_idx)

    def boxes_with_ids_for_frame(self, frame_idx: int):
        """
        UI helper: return [(object_id_str, (cx, cy, w, h, theta)), ...]
        in the same order the overlay will draw them.
        """
        return self.project_state.boxes_with_ids_for_frame(frame_idx)

    def object_id_for_frame_index(self, frame_idx: int, overlay_index: int) -> str:
        """
        UI helper: map overlay row index -> object_id_str for that frame.
        """
        return self.project_state.object_id_for_frame_index(frame_idx, overlay_index)

    def validate_before_save(self) -> None:
        self.project_state.validate_before_save()
