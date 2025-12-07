"""Main application window with split panel layout."""

import logging
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QMenuBar, QMenu, QStatusBar, QFileDialog, QMessageBox,
    QTabWidget, QLabel, QDialog, QDialogButtonBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence, QPixmap

from .states.app_state import AppState
from .widgets.project_browser import ProjectBrowser
from .widgets.dataset_selector import DatasetSelector
from .widgets.config_list import ConfigList
from .widgets.analysis_panel import AnalysisPanel
from .widgets.config_editor import ConfigEditor
from .dialogs.generate_dialog import GenerateDialog
from ..controllers.project_controller import ProjectController
from ..controllers.dataset_controller import DatasetController
from ..controllers.config_controller import ConfigController

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window with split panel layout.

    Layout:
    - Left panel (300px): Project browser, dataset selector, config list
    - Right panel: Tabbed view with Analysis and Config Editor
    """

    def __init__(
        self,
        app_state: AppState,
        project_controller: ProjectController,
        dataset_controller: DatasetController,
        config_controller: ConfigController,
        parent=None
    ):
        super().__init__(parent)

        self.app_state = app_state
        self.project_controller = project_controller
        self.dataset_controller = dataset_controller
        self.config_controller = config_controller

        self.setWindowTitle("trainit-gui - YOLO Training Manager")
        self.setMinimumSize(1200, 800)

        self._setup_ui()
        self._setup_menu()
        self._setup_statusbar()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the main UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Create splitter for left/right panels
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.splitter)

        # Left panel
        left_panel = self._create_left_panel()
        self.splitter.addWidget(left_panel)

        # Right panel
        right_panel = self._create_right_panel()
        self.splitter.addWidget(right_panel)

        # Set initial splitter sizes (300px left, rest for right)
        self.splitter.setSizes([300, 900])
        self.splitter.setStretchFactor(0, 0)  # Left panel doesn't stretch
        self.splitter.setStretchFactor(1, 1)  # Right panel stretches

    def _create_left_panel(self) -> QWidget:
        """Create the left panel with project/dataset/config widgets."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Project info (compact - just name and root path)
        self.project_browser = ProjectBrowser(
            self.app_state,
            self.project_controller
        )
        layout.addWidget(self.project_browser)

        # Dataset selector (gets most space)
        self.dataset_selector = DatasetSelector(
            self.app_state,
            self.dataset_controller
        )
        layout.addWidget(self.dataset_selector, stretch=2)

        # Config list
        self.config_list = ConfigList(
            self.app_state,
            self.config_controller
        )
        layout.addWidget(self.config_list, stretch=1)

        return panel

    def _create_right_panel(self) -> QWidget:
        """Create the right panel with tabbed content."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        # Tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Analysis tab
        self.analysis_panel = AnalysisPanel(self.app_state)
        self.tab_widget.addTab(self.analysis_panel, "Analysis")

        # Config Editor tab
        self.config_editor = ConfigEditor(
            self.app_state,
            self.config_controller
        )
        self.tab_widget.addTab(self.config_editor, "Configuration")

        return panel

    def _setup_menu(self):
        """Setup the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        new_action = QAction("&New Project...", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self._on_new_project)
        file_menu.addAction(new_action)

        open_action = QAction("&Open Project...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._on_open_project)
        file_menu.addAction(open_action)

        save_action = QAction("&Save Project", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._on_save_project)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        generate_action = QAction("&Generate Training Files...", self)
        generate_action.setShortcut("Ctrl+G")
        generate_action.triggered.connect(self._on_generate)
        file_menu.addAction(generate_action)

        verify_action = QAction("&Verify Manifest...", self)
        verify_action.setShortcut("Ctrl+Shift+V")
        verify_action.triggered.connect(self._on_verify_manifest)
        file_menu.addAction(verify_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Project menu
        project_menu = menubar.addMenu("&Project")

        refresh_action = QAction("&Refresh Datasets", self)
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self._on_refresh_datasets)
        project_menu.addAction(refresh_action)

        project_menu.addSeparator()

        change_root_action = QAction("Change &Datasets Root...", self)
        change_root_action.triggered.connect(self._on_change_datasets_root)
        project_menu.addAction(change_root_action)

        defaults_action = QAction("Change Project &Defaults...", self)
        defaults_action.triggered.connect(self._on_change_defaults)
        project_menu.addAction(defaults_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _setup_statusbar(self):
        """Setup the status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Ready")

    def _connect_signals(self):
        """Connect app state signals to UI updates."""
        self.app_state.status_message.connect(self._on_status_message)
        self.app_state.error_occurred.connect(self._on_error)
        self.app_state.project_changed.connect(self._on_project_changed)

    def _on_status_message(self, message: str):
        """Handle status message."""
        self.statusbar.showMessage(message, 5000)

    def _on_error(self, message: str):
        """Handle error message."""
        self.statusbar.showMessage(f"Error: {message}", 10000)
        logger.error(message)

    def _on_project_changed(self, project):
        """Handle project change."""
        if project:
            self.setWindowTitle(f"trainit-gui - {project.name}")
        else:
            self.setWindowTitle("trainit-gui - YOLO Training Manager")

    def _on_new_project(self):
        """Handle new project action."""
        from .dialogs.new_project_dialog import NewProjectDialog

        dialog = NewProjectDialog(self)
        if dialog.exec():
            name, folder, datasets_root, description = dialog.get_values()
            self.project_controller.create_project(
                name=name,
                project_folder=folder,
                datasets_root=datasets_root,
                description=description
            )

    def _on_open_project(self):
        """Handle open project action."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            "",
            "Project files (project.json);;All files (*)"
        )
        if file_path:
            self.project_controller.open_project(file_path)

    def _on_save_project(self):
        """Handle save project action."""
        self.project_controller.save_project()

    def _on_generate(self):
        """Handle generate action."""
        if not self.app_state.current_config:
            QMessageBox.warning(
                self,
                "No Configuration",
                "Please select or create a training configuration first."
            )
            return

        dialog = GenerateDialog(
            self.app_state,
            self.config_controller,
            self
        )
        dialog.exec()

    def _on_refresh_datasets(self):
        """Handle refresh datasets action."""
        self.project_controller.refresh_datasets()

    def _on_change_datasets_root(self):
        """Handle change datasets root action."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Datasets Root Directory",
            self.app_state.datasets_root or ""
        )
        if folder:
            self.project_controller.set_datasets_root(folder)

    def _on_change_defaults(self):
        """Handle change project defaults action."""
        if not self.app_state.project:
            QMessageBox.warning(
                self,
                "No Project",
                "Please open or create a project first."
            )
            return

        from .dialogs.project_defaults_dialog import ProjectDefaultsDialog

        dialog = ProjectDefaultsDialog(
            self.app_state.project.default_config,
            self
        )
        if dialog.exec():
            self.app_state.project.default_config = dialog.get_defaults()
            self.project_controller.save_project()
            # Notify listeners that project (defaults) changed so they can update
            self.app_state.project_changed.emit(self.app_state.project)
            self.app_state.status_message.emit("Project defaults updated")

    def _on_about(self):
        """Show about dialog with logo."""
        dialog = QDialog(self)
        dialog.setWindowTitle("About trainit-gui")

        layout = QHBoxLayout(dialog)

        # Logo on the left
        logo_label = QLabel()
        logo_path = Path(__file__).parent.parent.parent.parent / "docs" / "savant_logo.png"
        if logo_path.exists():
            pixmap = QPixmap(str(logo_path))
            # Scale to reasonable size (max 128px height)
            scaled = pixmap.scaledToHeight(128, Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(scaled)
        layout.addWidget(logo_label)

        # Text and button on the right
        right_layout = QVBoxLayout()

        text_label = QLabel(
            "<h2>trainit-gui</h2>"
            "<p>Version 1.0.0</p>"
            "<p>GUI for managing YOLO training datasets "
            "and generating training configurations.</p>"
            "<p>Part of the <b>SAVANT</b> toolkit.</p>"
        )
        text_label.setWordWrap(True)
        right_layout.addWidget(text_label)

        right_layout.addStretch()

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        button_box.accepted.connect(dialog.accept)
        right_layout.addWidget(button_box)

        layout.addLayout(right_layout)

        dialog.exec()

    def _on_verify_manifest(self):
        """Handle verify manifest action."""
        from .dialogs.verify_manifest_dialog import VerifyManifestDialog

        dialog = VerifyManifestDialog(self)
        dialog.exec()

    def closeEvent(self, event):
        """Handle window close."""
        if self.app_state.has_unsaved_changes():
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "There are unsaved changes. Save before closing?",
                QMessageBox.StandardButton.Save |
                QMessageBox.StandardButton.Discard |
                QMessageBox.StandardButton.Cancel
            )

            if reply == QMessageBox.StandardButton.Save:
                self.project_controller.save_project()
                event.accept()
            elif reply == QMessageBox.StandardButton.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
