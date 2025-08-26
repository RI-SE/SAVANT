import sys
from PyQt6.QtWidgets import QApplication
from frontend.main_window import MainWindow
from .controllers.project_state_controller import ProjectStateController
from .project_state import ProjectState
from .controllers.video_controller import VideoController
from .services.video_reader import VideoReader

if __name__ == "__main__":

    # Initialize centralized state and PYQT widgets
    project_state = ProjectState()
    app = QApplication(sys.argv)

    # Initialize services
    video_service = VideoReader()

    # Initialize controllers
    project_state_controller = ProjectStateController(project_state)
    video_controller = VideoController(video_service)


    # Setup UI
    # TODO - Get project name from current project
    # TODO: Implement passing controllers
    window = MainWindow(
        project_name="temp_name",
        video_controller=video_controller,
        project_state_controller=project_state_controller
    )
    window.show()
    sys.exit(app.exec())
