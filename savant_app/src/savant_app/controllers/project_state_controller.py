"""
Class representing the controller to the project state.

The purpose of this controller is to facilitate frontend<->project_state
communication while maintaining separation of concerns, and a decoupled frontend.
"""

from savant_app.services.project_state import ProjectState
from .error_handler_middleware import error_handler


class ProjectStateController:
    # TODO: ERROR HANDLING: Catch service level errors, make them UI friendly.
    def __init__(self, project_state: ProjectState):
        self.project_state = project_state

    @error_handler
    def load_openlabel_config(self, path: str) -> None:
        self.project_state.load_openlabel_config(path)

    @error_handler
    def save_openlabel_config(self) -> None:
        self.project_state.save_openlabel_config()

    @error_handler
    def get_actor_types(self) -> list[str]:
        return self.project_state.get_actor_types()

    @error_handler
    def boxes_for_frame(self, frame_idx: int):
        return self.project_state.boxes_for_frame(frame_idx)

    @error_handler
    def boxes_with_ids_for_frame(self, frame_idx: int):
        """
        UI helper: return [(object_id_str, (cx, cy, w, h, theta)), ...]
        in the same order the overlay will draw them.
        """
        return self.project_state.boxes_with_ids_for_frame(frame_idx)

    @error_handler
    def object_id_for_frame_index(self, frame_idx: int, overlay_index: int) -> str:
        """
        UI helper: map overlay row index -> object_id_str for that frame.
        """
        return self.project_state.object_id_for_frame_index(frame_idx, overlay_index)

    def validate_before_save(self) -> None:
        self.project_state.validate_before_save()

    @error_handler
    def get_video_metadata(self):
        """Return the entire video metadata dictionary"""
        return self.project_state.video_metadata

    @error_handler
    def get_frame_count(self) -> int:
        """Return total number of frames in the loaded video"""
        return self.project_state.video_metadata.frame_count

    @error_handler
    def get_fps(self) -> float:
        """Return frames per second of the loaded video"""
        return self.project_state.video_metadata.fps

    @error_handler
    def get_video_size(self):
        """Return video dimensions as (width, height) tuple"""
        metadata = self.project_state.video_metadata
        return (metadata.width, metadata.height)
