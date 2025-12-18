"""Dataset selector widget with checkable list."""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QGroupBox,
)
from PyQt6.QtCore import Qt

from ..states.app_state import AppState
from ...controllers.dataset_controller import DatasetController


class DatasetSelector(QWidget):
    """Widget for selecting datasets from the available list."""

    def __init__(
        self, app_state: AppState, dataset_controller: DatasetController, parent=None
    ):
        super().__init__(parent)

        self.app_state = app_state
        self.dataset_controller = dataset_controller
        self._updating = False

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Datasets group
        group = QGroupBox("Datasets")
        group_layout = QVBoxLayout(group)

        # Dataset list
        self.dataset_list = QListWidget()
        self.dataset_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.dataset_list.itemChanged.connect(self._on_item_changed)
        group_layout.addWidget(self.dataset_list)

        # Selection info
        self.selection_label = QLabel("0 datasets selected")
        group_layout.addWidget(self.selection_label)

        # Buttons
        btn_layout = QHBoxLayout()

        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self._on_select_all)
        btn_layout.addWidget(self.select_all_btn)

        self.clear_btn = QPushButton("Deselect All")
        self.clear_btn.clicked.connect(self._on_clear_selection)
        btn_layout.addWidget(self.clear_btn)

        group_layout.addLayout(btn_layout)

        # Analyze button
        self.analyze_btn = QPushButton("Analyze Selected")
        self.analyze_btn.clicked.connect(self._on_analyze)
        self.analyze_btn.setEnabled(False)
        group_layout.addWidget(self.analyze_btn)

        layout.addWidget(group)

    def _connect_signals(self):
        """Connect app state signals."""
        self.app_state.available_datasets_changed.connect(self._on_available_changed)
        self.app_state.selected_datasets_changed.connect(self._on_selection_changed)

    def _on_available_changed(self, datasets: list):
        """Handle available datasets change."""
        self._updating = True
        self.dataset_list.clear()

        for name in datasets:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.dataset_list.addItem(item)

        self._updating = False
        self._update_selection_label()

    def _on_selection_changed(self, selected: list):
        """Handle selection change from external source."""
        self._updating = True

        for i in range(self.dataset_list.count()):
            item = self.dataset_list.item(i)
            if item.text() in selected:
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setCheckState(Qt.CheckState.Unchecked)

        self._updating = False
        self._update_selection_label()

    def _on_item_changed(self, item: QListWidgetItem):
        """Handle item check state change."""
        if self._updating:
            return

        name = item.text()
        is_checked = item.checkState() == Qt.CheckState.Checked
        self.dataset_controller.toggle_dataset(name, is_checked)
        self._update_selection_label()

        # Mark config dirty if one is selected
        if self.app_state.current_config:
            self.app_state.config_dirty = True

    def _on_select_all(self):
        """Select all datasets."""
        all_names = [
            self.dataset_list.item(i).text() for i in range(self.dataset_list.count())
        ]
        self.dataset_controller.select_datasets(all_names)
        if self.app_state.current_config:
            self.app_state.config_dirty = True

    def _on_clear_selection(self):
        """Clear all selections."""
        self.dataset_controller.select_datasets([])
        if self.app_state.current_config:
            self.app_state.config_dirty = True

    def _on_analyze(self):
        """Trigger analysis of selected datasets."""
        self.dataset_controller.analyze_selected()

    def _update_selection_label(self):
        """Update the selection count label."""
        count = sum(
            1
            for i in range(self.dataset_list.count())
            if self.dataset_list.item(i).checkState() == Qt.CheckState.Checked
        )
        self.selection_label.setText(f"{count} datasets selected")
        self.analyze_btn.setEnabled(count > 0)
