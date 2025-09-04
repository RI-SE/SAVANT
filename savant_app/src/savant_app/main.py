import sys
from PyQt6.QtWidgets import QApplication
from .frontend.main_window import MainWindow
from .controllers.project_state_controller import ProjectStateController
from .controllers.annotation_controller import AnnotationController
from .services.annotation_service import AnnotationService
from .services.project_state import ProjectState
from .controllers.video_controller import VideoController
from .services.video_reader import VideoReader

if __name__ == "__main__":

    # Initialize centralized state and PYQT widgets
    project_state = ProjectState()
    app = QApplication(sys.argv)

    # Initialize services
    video_service = VideoReader()
    annotation_service = AnnotationService(project_state)

    # Initialize controllers
    project_state_controller = ProjectStateController(
        project_state
    )  # The only controller with project state.
    video_controller = VideoController(video_service)
    annotation_controller = AnnotationController(annotation_service)

    # Setup UI
    # TODO - Get project name from current project
    # TODO: Implement passing controllers
    window = MainWindow(
        project_name="temp_name",
        video_controller=video_controller,
        project_state_controller=project_state_controller,
        annotation_controller=annotation_controller,
    )
    window.show()
    sys.exit(app.exec())
