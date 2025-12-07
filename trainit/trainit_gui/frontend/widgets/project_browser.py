"""Project info widget showing current project and datasets root."""

from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel
)
from PyQt6.QtCore import Qt

from ..states.app_state import AppState
from ...controllers.project_controller import ProjectController


class ProjectBrowser(QWidget):
    """Compact widget showing current project info."""

    def __init__(
        self,
        app_state: AppState,
        project_controller: ProjectController,
        parent=None
    ):
        super().__init__(parent)

        self.app_state = app_state
        self.project_controller = project_controller

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(2)

        # Project name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Project:"))
        self.project_name_label = QLabel("No project")
        self.project_name_label.setStyleSheet("font-weight: bold;")
        name_layout.addWidget(self.project_name_label, stretch=1)
        layout.addLayout(name_layout)

        # Datasets root (truncated path)
        root_layout = QHBoxLayout()
        root_layout.addWidget(QLabel("Root:"))
        self.datasets_root_label = QLabel("-")
        self.datasets_root_label.setStyleSheet("color: gray; font-size: 11px;")
        self.datasets_root_label.setToolTip("")
        root_layout.addWidget(self.datasets_root_label, stretch=1)
        layout.addLayout(root_layout)

    def _connect_signals(self):
        """Connect app state signals."""
        self.app_state.project_changed.connect(self._on_project_changed)
        self.app_state.datasets_root_changed.connect(self._on_datasets_root_changed)

    def _on_project_changed(self, project):
        """Handle project change."""
        if project:
            self.project_name_label.setText(project.name)
        else:
            self.project_name_label.setText("No project")
            self.datasets_root_label.setText("-")
            self.datasets_root_label.setToolTip("")

    def _on_datasets_root_changed(self, path: str):
        """Handle datasets root change."""
        if path:
            # Show truncated path
            p = Path(path)
            display = f".../{p.parent.name}/{p.name}" if len(path) > 40 else path
            self.datasets_root_label.setText(display)
            self.datasets_root_label.setToolTip(path)
        else:
            self.datasets_root_label.setText("-")
            self.datasets_root_label.setToolTip("")
