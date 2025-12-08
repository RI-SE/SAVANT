"""Service for loading and saving project files."""

import json
import logging
from pathlib import Path
from typing import Optional

from ..models.project import Project, TrainingConfig

logger = logging.getLogger(__name__)


class ProjectService:
    """Service for managing project files.

    Projects are stored as JSON files in project folders:
        project_folder/
        └── project.json
    """

    PROJECT_FILE_NAME = "project.json"

    def create_project(
        self,
        name: str,
        project_folder: str,
        datasets_root: str,
        description: str = ""
    ) -> Project:
        """Create a new project and save it.

        Args:
            name: Project name
            project_folder: Directory to store the project file
            datasets_root: Root directory containing training datasets
            description: Optional project description

        Returns:
            The created Project

        Raises:
            ValueError: If project folder doesn't exist or isn't a directory
            FileExistsError: If project.json already exists
        """
        folder = Path(project_folder)
        if not folder.is_dir():
            raise ValueError(f"Project folder doesn't exist: {project_folder}")

        project_file = folder / self.PROJECT_FILE_NAME
        if project_file.exists():
            raise FileExistsError(f"Project already exists: {project_file}")

        project = Project(
            name=name,
            datasets_root=datasets_root,
            description=description
        )

        self.save_project(project, str(project_file))
        return project

    def load_project(self, project_file_path: str) -> Project:
        """Load a project from a JSON file.

        Args:
            project_file_path: Path to project.json file

        Returns:
            The loaded Project

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is invalid JSON or doesn't match schema
        """
        path = Path(project_file_path)
        if not path.exists():
            raise FileNotFoundError(f"Project file not found: {project_file_path}")

        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return Project.model_validate(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in project file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading project: {e}")

    def save_project(self, project: Project, project_file_path: str) -> None:
        """Save a project to a JSON file.

        Args:
            project: The Project to save
            project_file_path: Path to save the project.json file

        Raises:
            OSError: If unable to write to the file
        """
        project.update_modified()

        path = Path(project_file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(path, 'w') as f:
                json.dump(
                    project.model_dump(),
                    f,
                    indent=2,
                    ensure_ascii=False
                )
            logger.info(f"Saved project to {path}")
        except Exception as e:
            logger.error(f"Failed to save project: {e}")
            raise

    def get_project_file_path(self, project_folder: str) -> str:
        """Get the expected project.json path for a folder."""
        return str(Path(project_folder) / self.PROJECT_FILE_NAME)

    def project_exists(self, project_folder: str) -> bool:
        """Check if a project file exists in the given folder."""
        path = Path(project_folder) / self.PROJECT_FILE_NAME
        return path.exists()

    def list_recent_projects(self, paths: list[str]) -> list[tuple[str, str]]:
        """Validate a list of project paths and return valid ones with names.

        Args:
            paths: List of project.json paths to validate

        Returns:
            List of (path, project_name) tuples for valid projects
        """
        valid = []
        for path in paths:
            try:
                project = self.load_project(path)
                valid.append((path, project.name))
            except Exception:
                logger.debug(f"Invalid or missing project: {path}")
        return valid

    def add_config_to_project(
        self,
        project_file_path: str,
        config: TrainingConfig
    ) -> Project:
        """Add a new configuration to an existing project.

        Args:
            project_file_path: Path to project.json
            config: Configuration to add

        Returns:
            Updated Project

        Raises:
            ValueError: If a config with the same name already exists
        """
        project = self.load_project(project_file_path)

        if project.get_config_by_name(config.name):
            raise ValueError(f"Config '{config.name}' already exists in project")

        project.add_config(config)
        self.save_project(project, project_file_path)
        return project

    def update_config_in_project(
        self,
        project_file_path: str,
        config: TrainingConfig
    ) -> Project:
        """Update an existing configuration in a project.

        Args:
            project_file_path: Path to project.json
            config: Configuration to update (matched by name)

        Returns:
            Updated Project

        Raises:
            ValueError: If config with the given name doesn't exist
        """
        project = self.load_project(project_file_path)

        existing = project.get_config_by_name(config.name)
        if not existing:
            raise ValueError(f"Config '{config.name}' not found in project")

        # Replace the config
        project.remove_config(config.name)
        project.add_config(config)
        self.save_project(project, project_file_path)
        return project

    def remove_config_from_project(
        self,
        project_file_path: str,
        config_name: str
    ) -> Project:
        """Remove a configuration from a project.

        Args:
            project_file_path: Path to project.json
            config_name: Name of config to remove

        Returns:
            Updated Project
        """
        project = self.load_project(project_file_path)
        project.remove_config(config_name)
        self.save_project(project, project_file_path)
        return project
