import sys
from PyQt6.QtWidgets import QApplication
from edit.frontend.main_window import MainWindow
from edit.controllers.project_state_controller import ProjectStateController
from edit.controllers.annotation_controller import AnnotationController
from edit.services.annotation_service import AnnotationService
from edit.services.autosave_service import AutosaveService
from edit.services.project_state import ProjectState
from edit.controllers.video_controller import VideoController
from edit.services.video_reader import VideoReader
from edit.global_exception_handler import exception_hook
from edit.frontend.theme.menu_styler import install_menu_styler
from edit.frontend.utils.project_config import get_active_project_dir, load_project_config
from edit.frontend.utils.settings_store import get_autosave_interval_minutes
from edit.logger_config import setup_logger


def main():
    """Initializes and runs the Savant application."""
    setup_logger()  # Set up logging configuration

    # Initialize centralized state and PYQT widgets
    project_state = ProjectState()
    app = QApplication(sys.argv)
    install_menu_styler(app)

    def _get_project_config_snapshot() -> dict | None:
        project_dir = get_active_project_dir()
        if project_dir is None:
            return None
        config = load_project_config(project_dir)
        return config.to_dict()

    autosave_service = AutosaveService(
        project_state,
        get_project_config_snapshot=_get_project_config_snapshot,
    )
    autosave_service.set_interval(get_autosave_interval_minutes())

    # Initialize services
    video_service = VideoReader(project_state)
    annotation_service = AnnotationService(project_state)

    # Initialize controllers
    project_state_controller = ProjectStateController(
        project_state
    )  # The only controller with project state.
    video_controller = VideoController(video_service)
    annotation_controller = AnnotationController(annotation_service)

    # Setup UI
    window = MainWindow(
        project_name="",
        video_controller=video_controller,
        project_state_controller=project_state_controller,
        annotation_controller=annotation_controller,
        autosave_service=autosave_service,
    )
    window.show()

    sys.excepthook = exception_hook

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
