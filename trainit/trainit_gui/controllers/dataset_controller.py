"""Controller for dataset operations."""

import logging
from typing import Optional

from ..frontend.states.app_state import AppState
from ..services.dataset_service import DatasetService
from ..models.dataset import DatasetInfo

logger = logging.getLogger(__name__)


class DatasetController:
    """Controller for dataset browsing and analysis."""

    def __init__(self, app_state: AppState, dataset_service: DatasetService):
        self.app_state = app_state
        self.dataset_service = dataset_service

    def select_datasets(self, dataset_names: list[str]) -> None:
        """Update the selected datasets."""
        self.app_state.selected_datasets = dataset_names
        logger.debug(f"Selected datasets: {dataset_names}")

    def toggle_dataset(self, name: str, selected: bool) -> None:
        """Toggle a single dataset's selection state."""
        current = list(self.app_state.selected_datasets)

        if selected and name not in current:
            current.append(name)
        elif not selected and name in current:
            current.remove(name)

        self.app_state.selected_datasets = current

    def analyze_selected(self) -> bool:
        """Analyze the currently selected datasets.

        Returns:
            True if successful, False otherwise
        """
        if not self.app_state.selected_datasets:
            self.app_state.error_occurred.emit("No datasets selected")
            return False

        if not self.app_state.datasets_root:
            self.app_state.error_occurred.emit("Datasets root not set")
            return False

        try:
            self.app_state.status_message.emit("Analyzing datasets...")

            # Get aggregated stats
            stats = self.dataset_service.analyze_datasets(
                self.app_state.datasets_root, self.app_state.selected_datasets
            )

            if not stats.is_valid:
                self.app_state.error_occurred.emit(stats.error_message)
                self.app_state.analysis = None
                return False

            # Cache individual dataset info
            for name in self.app_state.selected_datasets:
                if not self.app_state.get_dataset_info(name):
                    from pathlib import Path

                    dataset_path = Path(self.app_state.datasets_root) / name
                    info = self.dataset_service.load_dataset_info(str(dataset_path))
                    self.app_state.set_dataset_info(name, info)

            self.app_state.analysis = stats
            self.app_state.status_message.emit(
                f"Analyzed {len(stats.dataset_names)} datasets: "
                f"{stats.total_images} images, {stats.total_objects} objects"
            )
            logger.info(f"Analysis complete: {stats.total_images} images")
            return True

        except Exception as e:
            self.app_state.error_occurred.emit(f"Analysis failed: {e}")
            logger.error(f"Analysis failed: {e}")
            return False

    def get_dataset_info(self, name: str) -> Optional[DatasetInfo]:
        """Get info for a specific dataset, loading if necessary."""
        # Check cache first
        info = self.app_state.get_dataset_info(name)
        if info:
            return info

        if not self.app_state.datasets_root:
            return None

        try:
            from pathlib import Path

            dataset_path = Path(self.app_state.datasets_root) / name
            info = self.dataset_service.load_dataset_info(str(dataset_path))
            self.app_state.set_dataset_info(name, info)
            return info

        except Exception as e:
            logger.error(f"Failed to load dataset info for {name}: {e}")
            return None

    def validate_class_compatibility(
        self, dataset_names: list[str]
    ) -> tuple[bool, str]:
        """Check if all specified datasets have compatible class mappings.

        Returns:
            Tuple of (is_compatible, error_message)
        """
        if len(dataset_names) < 2:
            return True, ""

        infos = []
        for name in dataset_names:
            info = self.get_dataset_info(name)
            if not info or not info.is_valid:
                return False, f"Invalid dataset: {name}"
            infos.append(info)

        first = infos[0]
        for info in infos[1:]:
            if not first.has_matching_classes(info):
                return False, (
                    f"Class mismatch: '{first.name}' and '{info.name}' "
                    f"have different class definitions"
                )

        return True, ""
