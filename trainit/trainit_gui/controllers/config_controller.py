"""Controller for training configuration operations."""

import logging
from pathlib import Path
from typing import Optional

from ..frontend.states.app_state import AppState
from ..services.project_service import ProjectService
from ..services.config_generator import ConfigGenerator
from ..models.project import TrainingConfig

logger = logging.getLogger(__name__)


class ConfigController:
    """Controller for training configuration management."""

    def __init__(
        self,
        app_state: AppState,
        project_service: ProjectService,
        config_generator: ConfigGenerator
    ):
        self.app_state = app_state
        self.project_service = project_service
        self.config_generator = config_generator

    def create_config(self, name: str, description: str = "") -> bool:
        """Create a new training configuration.

        Uses project defaults if set, otherwise uses trainit defaults.

        Returns:
            True if successful, False otherwise
        """
        if not self.app_state.project:
            self.app_state.error_occurred.emit("No project open")
            return False

        # Check for duplicate name
        if self.app_state.project.get_config_by_name(name):
            self.app_state.error_occurred.emit(f"Config '{name}' already exists")
            return False

        try:
            # Start with trainit defaults
            config = TrainingConfig(
                name=name,
                description=description,
                selected_datasets=list(self.app_state.selected_datasets)
            )

            # Apply project defaults if set
            if self.app_state.project.default_config:
                config = self.app_state.project.default_config.apply_to_config(config)

            self.app_state.project.add_config(config)
            self.app_state.current_config = config
            self.app_state.config_list_changed.emit(self.app_state.project.configs)

            # Save project
            if self.app_state.project_path:
                self.project_service.save_project(
                    self.app_state.project,
                    self.app_state.project_path
                )

            self.app_state.status_message.emit(f"Created config: {name}")
            logger.info(f"Created config: {name}")
            return True

        except Exception as e:
            self.app_state.error_occurred.emit(f"Failed to create config: {e}")
            logger.error(f"Failed to create config: {e}")
            return False

    def select_config(self, name: str) -> bool:
        """Select a configuration for editing.

        Returns:
            True if successful, False otherwise
        """
        if not self.app_state.project:
            return False

        config = self.app_state.project.get_config_by_name(name)
        if not config:
            self.app_state.error_occurred.emit(f"Config '{name}' not found")
            return False

        self.app_state.current_config = config
        self.app_state.selected_datasets = list(config.selected_datasets)
        self.app_state.config_dirty = False  # Fresh load is not dirty
        logger.debug(f"Selected config: {name}")
        return True

    def update_config(self, config: TrainingConfig) -> bool:
        """Update the current configuration.

        Returns:
            True if successful, False otherwise
        """
        if not self.app_state.project:
            self.app_state.error_occurred.emit("No project open")
            return False

        try:
            # Remove old and add updated
            self.app_state.project.remove_config(config.name)
            self.app_state.project.add_config(config)
            self.app_state.current_config = config
            self.app_state.config_list_changed.emit(self.app_state.project.configs)

            # Save project
            if self.app_state.project_path:
                self.project_service.save_project(
                    self.app_state.project,
                    self.app_state.project_path
                )

            self.app_state.status_message.emit(f"Updated config: {config.name}")
            return True

        except Exception as e:
            self.app_state.error_occurred.emit(f"Failed to update config: {e}")
            logger.error(f"Failed to update config: {e}")
            return False

    def delete_config(self, name: str) -> bool:
        """Delete a configuration.

        Returns:
            True if successful, False otherwise
        """
        if not self.app_state.project:
            return False

        try:
            if self.app_state.project.remove_config(name):
                if (self.app_state.current_config and
                    self.app_state.current_config.name == name):
                    self.app_state.current_config = None

                self.app_state.config_list_changed.emit(self.app_state.project.configs)

                # Save project
                if self.app_state.project_path:
                    self.project_service.save_project(
                        self.app_state.project,
                        self.app_state.project_path
                    )

                self.app_state.status_message.emit(f"Deleted config: {name}")
                return True

            return False

        except Exception as e:
            self.app_state.error_occurred.emit(f"Failed to delete config: {e}")
            return False

    def generate_files(
        self,
        output_dir: str,
        copy_images: bool = False,
        generate_manifest: bool = True
    ) -> tuple[bool, str, str, Optional[str]]:
        """Generate training configuration files.

        Args:
            output_dir: Directory to write output files
            copy_images: If True, copy images instead of symlinking
            generate_manifest: If True, generate manifest JSON with file hashes

        Returns:
            Tuple of (success, yaml_path, json_path, manifest_path or None)
        """
        if not self.app_state.project or not self.app_state.current_config:
            self.app_state.error_occurred.emit("No project or config selected")
            return False, "", "", None

        config = self.app_state.current_config

        if not config.selected_datasets:
            self.app_state.error_occurred.emit("No datasets selected in config")
            return False, "", "", None

        try:
            self.app_state.status_message.emit("Generating training files...")

            yaml_path, json_path, manifest_path = self.config_generator.generate_training_files(
                project=self.app_state.project,
                config=config,
                output_dir=output_dir,
                copy_images=copy_images,
                generate_manifest=generate_manifest
            )

            self.app_state.status_message.emit(
                f"Generated files in {output_dir}"
            )
            logger.info(f"Generated training files: {yaml_path}, {json_path}, {manifest_path}")
            return True, yaml_path, json_path, manifest_path

        except Exception as e:
            self.app_state.error_occurred.emit(f"Generation failed: {e}")
            logger.error(f"Generation failed: {e}")
            return False, "", "", None

    def preview_generation(
        self,
        output_dir: str,
        generate_manifest: bool = True
    ) -> dict:
        """Preview what files would be generated.

        Args:
            output_dir: Directory to write output files
            generate_manifest: If True, include manifest in preview

        Returns:
            Preview information dict
        """
        if not self.app_state.project or not self.app_state.current_config:
            return {'valid': False, 'error': 'No project or config selected'}

        return self.config_generator.preview_generation(
            project=self.app_state.project,
            config=self.app_state.current_config,
            output_dir=output_dir,
            generate_manifest=generate_manifest
        )

    def verify_manifest(self, manifest_path: str):
        """Verify source files against a manifest.

        Args:
            manifest_path: Path to manifest JSON file

        Returns:
            VerificationResult with verification details
        """
        from ..services.manifest_service import ManifestService
        from pathlib import Path

        manifest_service = ManifestService()
        return manifest_service.verify_manifest(Path(manifest_path))
