"""Controller for project management operations."""

import logging
from pathlib import Path
from typing import Optional

from ..frontend.states.app_state import AppState
from ..services.project_service import ProjectService
from ..services.dataset_service import DatasetService
from ..models.project import Project, TrainingConfig

logger = logging.getLogger(__name__)


class ProjectController:
    """Controller for project CRUD operations."""

    def __init__(
        self,
        app_state: AppState,
        project_service: ProjectService,
        dataset_service: DatasetService
    ):
        self.app_state = app_state
        self.project_service = project_service
        self.dataset_service = dataset_service

    def create_project(
        self,
        name: str,
        project_folder: str,
        datasets_root: str,
        description: str = ""
    ) -> bool:
        """Create a new project.

        Returns:
            True if successful, False otherwise
        """
        try:
            project = self.project_service.create_project(
                name=name,
                project_folder=project_folder,
                datasets_root=datasets_root,
                description=description
            )

            # Scan for available datasets
            available = self.dataset_service.scan_directory(datasets_root)
            project.available_datasets = available
            self.project_service.save_project(
                project,
                self.project_service.get_project_file_path(project_folder)
            )

            # Update app state
            self.app_state.project = project
            self.app_state.project_path = self.project_service.get_project_file_path(project_folder)
            self.app_state.datasets_root = datasets_root
            self.app_state.available_datasets = available
            self.app_state.selected_datasets = []
            self.app_state.current_config = None
            self.app_state.clear_dataset_cache()

            self.app_state.status_message.emit(f"Created project: {name}")
            logger.info(f"Created project: {name} at {project_folder}")
            return True

        except FileExistsError as e:
            self.app_state.error_occurred.emit(f"Project already exists: {e}")
            logger.error(f"Failed to create project: {e}")
            return False
        except Exception as e:
            self.app_state.error_occurred.emit(f"Failed to create project: {e}")
            logger.error(f"Failed to create project: {e}")
            return False

    def open_project(self, project_file_path: str) -> bool:
        """Open an existing project.

        Returns:
            True if successful, False otherwise
        """
        try:
            project = self.project_service.load_project(project_file_path)

            # Refresh available datasets
            available = self.dataset_service.scan_directory(project.datasets_root)
            project.available_datasets = available

            # Update app state
            self.app_state.project = project
            self.app_state.project_path = project_file_path
            self.app_state.datasets_root = project.datasets_root
            self.app_state.available_datasets = available
            self.app_state.selected_datasets = []
            self.app_state.current_config = None
            self.app_state.clear_dataset_cache()

            self.app_state.status_message.emit(f"Opened project: {project.name}")
            logger.info(f"Opened project: {project.name}")
            return True

        except FileNotFoundError as e:
            self.app_state.error_occurred.emit(f"Project file not found: {e}")
            logger.error(f"Failed to open project: {e}")
            return False
        except Exception as e:
            self.app_state.error_occurred.emit(f"Failed to open project: {e}")
            logger.error(f"Failed to open project: {e}")
            return False

    def save_project(self) -> bool:
        """Save the current project.

        Returns:
            True if successful, False otherwise
        """
        if not self.app_state.project or not self.app_state.project_path:
            self.app_state.error_occurred.emit("No project to save")
            return False

        try:
            self.project_service.save_project(
                self.app_state.project,
                self.app_state.project_path
            )
            self.app_state.status_message.emit("Project saved")
            logger.info(f"Saved project to {self.app_state.project_path}")
            return True

        except Exception as e:
            self.app_state.error_occurred.emit(f"Failed to save project: {e}")
            logger.error(f"Failed to save project: {e}")
            return False

    def close_project(self) -> None:
        """Close the current project."""
        self.app_state.reset()
        self.app_state.status_message.emit("Project closed")

    def refresh_datasets(self) -> bool:
        """Rescan the datasets root directory.

        Returns:
            True if successful, False otherwise
        """
        if not self.app_state.datasets_root:
            return False

        try:
            available = self.dataset_service.scan_directory(self.app_state.datasets_root)
            self.app_state.available_datasets = available

            if self.app_state.project:
                self.app_state.project.available_datasets = available

            self.app_state.clear_dataset_cache()
            self.app_state.status_message.emit(f"Found {len(available)} datasets")
            return True

        except Exception as e:
            self.app_state.error_occurred.emit(f"Failed to scan datasets: {e}")
            return False

    def set_datasets_root(self, path: str) -> bool:
        """Change the datasets root directory.

        Returns:
            True if successful, False otherwise
        """
        if not Path(path).is_dir():
            self.app_state.error_occurred.emit(f"Not a directory: {path}")
            return False

        self.app_state.datasets_root = path
        self.app_state.clear_dataset_cache()

        if self.app_state.project:
            self.app_state.project.datasets_root = path

        return self.refresh_datasets()
