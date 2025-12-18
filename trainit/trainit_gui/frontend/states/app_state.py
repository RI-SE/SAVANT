"""Application state management with Qt signals."""

from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal

from ...models.project import Project, TrainingConfig
from ...models.dataset import DatasetInfo, AggregatedStats


class AppState(QObject):
    """Central application state with Qt signals for UI updates.

    Emits signals when state changes so widgets can update accordingly.
    """

    # Signals for state changes
    project_changed = pyqtSignal(object)  # Project or None
    project_path_changed = pyqtSignal(str)  # Path to project.json
    datasets_root_changed = pyqtSignal(str)
    available_datasets_changed = pyqtSignal(list)  # list[str]
    selected_datasets_changed = pyqtSignal(list)  # list[str]
    analysis_updated = pyqtSignal(object)  # AggregatedStats or None
    current_config_changed = pyqtSignal(object)  # TrainingConfig or None
    config_list_changed = pyqtSignal(list)  # list[TrainingConfig]
    config_dirty_changed = pyqtSignal(bool)  # True if config has unsaved changes
    status_message = pyqtSignal(str)  # Status bar messages
    error_occurred = pyqtSignal(str)  # Error messages

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)

        self._project: Optional[Project] = None
        self._project_path: str = ""
        self._datasets_root: str = ""
        self._available_datasets: list[str] = []
        self._selected_datasets: list[str] = []
        self._analysis: Optional[AggregatedStats] = None
        self._current_config: Optional[TrainingConfig] = None
        self._config_dirty: bool = False
        self._dataset_infos: dict[str, DatasetInfo] = {}

    @property
    def project(self) -> Optional[Project]:
        return self._project

    @project.setter
    def project(self, value: Optional[Project]) -> None:
        self._project = value
        self.project_changed.emit(value)
        if value:
            self.config_list_changed.emit(value.configs)
        else:
            self.config_list_changed.emit([])

    @property
    def project_path(self) -> str:
        return self._project_path

    @project_path.setter
    def project_path(self, value: str) -> None:
        self._project_path = value
        self.project_path_changed.emit(value)

    @property
    def datasets_root(self) -> str:
        return self._datasets_root

    @datasets_root.setter
    def datasets_root(self, value: str) -> None:
        self._datasets_root = value
        self.datasets_root_changed.emit(value)

    @property
    def available_datasets(self) -> list[str]:
        return self._available_datasets

    @available_datasets.setter
    def available_datasets(self, value: list[str]) -> None:
        self._available_datasets = value
        self.available_datasets_changed.emit(value)

    @property
    def selected_datasets(self) -> list[str]:
        return self._selected_datasets

    @selected_datasets.setter
    def selected_datasets(self, value: list[str]) -> None:
        self._selected_datasets = value
        self.selected_datasets_changed.emit(value)

    @property
    def analysis(self) -> Optional[AggregatedStats]:
        return self._analysis

    @analysis.setter
    def analysis(self, value: Optional[AggregatedStats]) -> None:
        self._analysis = value
        self.analysis_updated.emit(value)

    @property
    def current_config(self) -> Optional[TrainingConfig]:
        return self._current_config

    @current_config.setter
    def current_config(self, value: Optional[TrainingConfig]) -> None:
        self._current_config = value
        self.current_config_changed.emit(value)

    @property
    def config_dirty(self) -> bool:
        return self._config_dirty

    @config_dirty.setter
    def config_dirty(self, value: bool) -> None:
        self._config_dirty = value
        self.config_dirty_changed.emit(value)

    def get_dataset_info(self, name: str) -> Optional[DatasetInfo]:
        """Get cached DatasetInfo by name."""
        return self._dataset_infos.get(name)

    def set_dataset_info(self, name: str, info: DatasetInfo) -> None:
        """Cache DatasetInfo."""
        self._dataset_infos[name] = info

    def clear_dataset_cache(self) -> None:
        """Clear cached dataset info."""
        self._dataset_infos.clear()

    def has_unsaved_changes(self) -> bool:
        """Check if there are unsaved changes."""
        # TODO: Track modifications
        return False

    def reset(self) -> None:
        """Reset all state."""
        self._project = None
        self._project_path = ""
        self._datasets_root = ""
        self._available_datasets = []
        self._selected_datasets = []
        self._analysis = None
        self._current_config = None
        self._config_dirty = False
        self._dataset_infos.clear()

        self.project_changed.emit(None)
        self.project_path_changed.emit("")
        self.datasets_root_changed.emit("")
        self.available_datasets_changed.emit([])
        self.selected_datasets_changed.emit([])
        self.analysis_updated.emit(None)
        self.current_config_changed.emit(None)
        self.config_dirty_changed.emit(False)
        self.config_list_changed.emit([])
