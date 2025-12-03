import sys
from PyQt6.QtWidgets import QApplication
from savant_app.frontend.main_window import MainWindow
from savant_app.controllers.project_state_controller import ProjectStateController
from savant_app.controllers.annotation_controller import AnnotationController
from savant_app.services.annotation_service import AnnotationService
from savant_app.services.project_state import ProjectState
from savant_app.controllers.video_controller import VideoController
from savant_app.services.video_reader import VideoReader
from savant_app.global_exception_handler import exception_hook
from savant_app.frontend.theme.menu_styler import install_menu_styler
from savant_app.logger_config import setup_logger


if __name__ == "__main__":

    setup_logger()  # Set up logging configuration

    # Initialize centralized state and PYQT widgets
    project_state = ProjectState()
    app = QApplication(sys.argv)
    install_menu_styler(app)

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
    # TODO: Implement passing controllers
    window = MainWindow(
        project_name="",
        video_controller=video_controller,
        project_state_controller=project_state_controller,
        annotation_controller=annotation_controller,
    )
    window.show()

    sys.excepthook = exception_hook

    sys.exit(app.exec())
