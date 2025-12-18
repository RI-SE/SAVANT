#!/usr/bin/env python3
"""
trainit-gui - PyQt6 GUI for managing YOLO training datasets and configurations.

Entry point for the application.
"""

import logging
import sys

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def check_dependencies() -> bool:
    """Check that required dependencies are available."""
    missing = []

    try:
        import PyQt6  # noqa: F401
    except ImportError:
        missing.append("PyQt6")

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        missing.append("matplotlib")

    try:
        import yaml  # noqa: F401
    except ImportError:
        missing.append("pyyaml")

    try:
        import pydantic  # noqa: F401
    except ImportError:
        missing.append("pydantic")

    if missing:
        print(f"Error: Missing required dependencies: {', '.join(missing)}")
        print("Install GUI dependencies with: pip install savant[gui]")
        return False

    return True


def exception_hook(exc_type, exc_value, exc_traceback):
    """Global exception handler."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    # Show error dialog if app is running
    app = QApplication.instance()
    if app:
        QMessageBox.critical(
            None,
            "Error",
            f"An unexpected error occurred:\n\n{exc_value}\n\n"
            "See the console for details.",
        )


def main():
    """Main entry point for trainit-gui."""
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)

    # Setup logging
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    setup_logging(verbose)

    logger = logging.getLogger(__name__)
    logger.info("Starting trainit-gui")

    # Import after dependency check
    from .frontend.main_window import MainWindow
    from .frontend.states.app_state import AppState
    from .services.dataset_service import DatasetService
    from .services.project_service import ProjectService
    from .services.config_generator import ConfigGenerator
    from .controllers.project_controller import ProjectController
    from .controllers.dataset_controller import DatasetController
    from .controllers.config_controller import ConfigController

    # Enable high DPI scaling (must be set before QApplication creation)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("trainit-gui")
    app.setOrganizationName("SAVANT")

    # Initialize state
    app_state = AppState()

    # Initialize services
    dataset_service = DatasetService()
    project_service = ProjectService()
    config_generator = ConfigGenerator(dataset_service)

    # Initialize controllers
    project_controller = ProjectController(
        app_state=app_state,
        project_service=project_service,
        dataset_service=dataset_service,
    )
    dataset_controller = DatasetController(
        app_state=app_state, dataset_service=dataset_service
    )
    config_controller = ConfigController(
        app_state=app_state,
        project_service=project_service,
        config_generator=config_generator,
    )

    # Create and show main window
    window = MainWindow(
        app_state=app_state,
        project_controller=project_controller,
        dataset_controller=dataset_controller,
        config_controller=config_controller,
    )
    window.show()

    # Install exception hook
    sys.excepthook = exception_hook

    logger.info("Application started")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
