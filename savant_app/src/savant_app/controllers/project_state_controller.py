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
