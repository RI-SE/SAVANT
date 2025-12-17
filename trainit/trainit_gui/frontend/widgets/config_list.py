"""Config list widget for managing training configurations."""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QGroupBox,
    QInputDialog,
    QMessageBox,
)
from PyQt6.QtCore import Qt

from ..states.app_state import AppState
from ...controllers.config_controller import ConfigController


class ConfigList(QWidget):
    """Widget for listing and managing training configurations."""

    def __init__(
        self, app_state: AppState, config_controller: ConfigController, parent=None
    ):
        super().__init__(parent)

        self.app_state = app_state
        self.config_controller = config_controller

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Configs group
        group = QGroupBox("Training Configurations")
        group_layout = QVBoxLayout(group)

        # Config list
        self.config_list = QListWidget()
        self.config_list.itemClicked.connect(self._on_item_clicked)
        self.config_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        group_layout.addWidget(self.config_list)

        # Buttons
        btn_layout = QHBoxLayout()

        self.new_btn = QPushButton("New Config")
        self.new_btn.clicked.connect(self._on_new_config)
        self.new_btn.setEnabled(False)
        btn_layout.addWidget(self.new_btn)

        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self._on_delete_config)
        self.delete_btn.setEnabled(False)
        btn_layout.addWidget(self.delete_btn)

        group_layout.addLayout(btn_layout)

        layout.addWidget(group)

    def _connect_signals(self):
        """Connect app state signals."""
        self.app_state.config_list_changed.connect(self._on_configs_changed)
        self.app_state.current_config_changed.connect(self._on_current_changed)
        self.app_state.project_changed.connect(self._on_project_changed)
        self.app_state.config_dirty_changed.connect(self._on_dirty_changed)

    def _on_dirty_changed(self, dirty: bool):
        """Update visual indicator for dirty state."""
        self._update_current_display()

    def _on_project_changed(self, project):
        """Handle project change."""
        self.new_btn.setEnabled(project is not None)
        if not project:
            self.config_list.clear()
            self.delete_btn.setEnabled(False)

    def _on_configs_changed(self, configs: list):
        """Handle configs list change."""
        self.config_list.clear()

        for config in configs:
            item = QListWidgetItem(config.name)
            item.setData(Qt.ItemDataRole.UserRole, config)
            if config.description:
                item.setToolTip(config.description)
            self.config_list.addItem(item)

    def _on_current_changed(self, config):
        """Handle current config change."""
        # Highlight current config
        for i in range(self.config_list.count()):
            item = self.config_list.item(i)
            stored_config = item.data(Qt.ItemDataRole.UserRole)
            if config and stored_config.name == config.name:
                item.setSelected(True)
                self.delete_btn.setEnabled(True)
            else:
                item.setSelected(False)
                # Reset any dirty indicator on non-current items
                font = item.font()
                font.setItalic(False)
                item.setFont(font)
                item.setText(stored_config.name)

        if not config:
            self.delete_btn.setEnabled(False)

        self._update_current_display()

    def _update_current_display(self):
        """Update the display of the current config (dirty indicator)."""
        for i in range(self.config_list.count()):
            item = self.config_list.item(i)
            config = item.data(Qt.ItemDataRole.UserRole)
            if (
                config
                and self.app_state.current_config
                and config.name == self.app_state.current_config.name
            ):
                # Set italics + asterisk if dirty
                font = item.font()
                font.setItalic(self.app_state.config_dirty)
                item.setFont(font)
                display_name = (
                    f"{config.name}*" if self.app_state.config_dirty else config.name
                )
                item.setText(display_name)

    def _on_item_clicked(self, item: QListWidgetItem):
        """Handle item click - check for unsaved changes first."""
        config = item.data(Qt.ItemDataRole.UserRole)
        if not config:
            return

        # Check if switching away from dirty config
        if self.app_state.config_dirty and self.app_state.current_config:
            if config.name != self.app_state.current_config.name:
                reply = QMessageBox.question(
                    self,
                    "Unsaved Changes",
                    f"Configuration '{self.app_state.current_config.name}' has unsaved changes.\n"
                    "Do you want to save before switching?",
                    QMessageBox.StandardButton.Save
                    | QMessageBox.StandardButton.Discard
                    | QMessageBox.StandardButton.Cancel,
                )
                if reply == QMessageBox.StandardButton.Save:
                    self._save_current_config()
                elif reply == QMessageBox.StandardButton.Cancel:
                    return  # Don't switch
                # Discard: just continue without saving

        self.config_controller.select_config(config.name)

    def _save_current_config(self):
        """Save the current config via the config editor."""
        window = self.window()
        if hasattr(window, "config_editor"):
            window.config_editor._on_save()

    def _on_item_double_clicked(self, item: QListWidgetItem):
        """Handle item double click - switch to config editor tab."""
        self._on_item_clicked(item)
        # Find parent window and switch tab
        window = self.window()
        if hasattr(window, "tab_widget"):
            window.tab_widget.setCurrentIndex(1)  # Config Editor tab

    def _on_new_config(self):
        """Handle new config button."""
        name, ok = QInputDialog.getText(
            self, "New Configuration", "Configuration name:"
        )
        if ok and name.strip():
            # Check if any datasets are selected
            if not self.app_state.selected_datasets:
                reply = QMessageBox.question(
                    self,
                    "No Datasets Selected",
                    "No datasets are currently selected. "
                    "Create configuration with no datasets?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.No:
                    return

            self.config_controller.create_config(name.strip())

    def _on_delete_config(self):
        """Handle delete config button."""
        current = self.config_list.currentItem()
        if not current:
            return

        config = current.data(Qt.ItemDataRole.UserRole)
        if not config:
            return

        reply = QMessageBox.question(
            self,
            "Delete Configuration",
            f"Delete configuration '{config.name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.config_controller.delete_config(config.name)
