"""Dialog for creating a new project."""

from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QLineEdit, QTextEdit, QPushButton, QFileDialog, QMessageBox
)


class NewProjectDialog(QDialog):
    """Dialog for creating a new training project."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("New Project")
        self.setMinimumWidth(500)

        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)

        # Form
        form = QFormLayout()

        # Project name
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("My Training Project")
        form.addRow("Project Name:", self.name_edit)

        # Project folder
        folder_layout = QHBoxLayout()
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Select project folder...")
        folder_layout.addWidget(self.folder_edit)

        browse_folder_btn = QPushButton("Browse...")
        browse_folder_btn.clicked.connect(self._browse_folder)
        folder_layout.addWidget(browse_folder_btn)

        form.addRow("Project Folder:", folder_layout)

        # Datasets root
        datasets_layout = QHBoxLayout()
        self.datasets_edit = QLineEdit()
        self.datasets_edit.setPlaceholderText("Select datasets root directory...")
        datasets_layout.addWidget(self.datasets_edit)

        browse_datasets_btn = QPushButton("Browse...")
        browse_datasets_btn.clicked.connect(self._browse_datasets)
        datasets_layout.addWidget(browse_datasets_btn)

        form.addRow("Datasets Root:", datasets_layout)

        # Description
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(80)
        self.description_edit.setPlaceholderText("Optional project description...")
        form.addRow("Description:", self.description_edit)

        layout.addLayout(form)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        create_btn = QPushButton("Create Project")
        create_btn.clicked.connect(self._on_create)
        create_btn.setDefault(True)
        btn_layout.addWidget(create_btn)

        layout.addLayout(btn_layout)

    def _browse_folder(self):
        """Browse for project folder."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Project Folder",
            ""
        )
        if folder:
            self.folder_edit.setText(folder)

    def _browse_datasets(self):
        """Browse for datasets root."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Datasets Root Directory",
            ""
        )
        if folder:
            self.datasets_edit.setText(folder)

    def _on_create(self):
        """Handle create button click."""
        name = self.name_edit.text().strip()
        folder = self.folder_edit.text().strip()
        datasets_root = self.datasets_edit.text().strip()

        # Validate
        if not name:
            QMessageBox.warning(self, "Validation Error", "Project name is required.")
            return

        if not folder:
            QMessageBox.warning(self, "Validation Error", "Project folder is required.")
            return

        if not Path(folder).is_dir():
            QMessageBox.warning(
                self,
                "Validation Error",
                f"Project folder doesn't exist: {folder}"
            )
            return

        if not datasets_root:
            QMessageBox.warning(
                self,
                "Validation Error",
                "Datasets root directory is required."
            )
            return

        if not Path(datasets_root).is_dir():
            QMessageBox.warning(
                self,
                "Validation Error",
                f"Datasets root doesn't exist: {datasets_root}"
            )
            return

        # Check if project already exists
        project_file = Path(folder) / "project.json"
        if project_file.exists():
            reply = QMessageBox.question(
                self,
                "Project Exists",
                f"A project already exists in {folder}. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return

        self.accept()

    def get_values(self) -> tuple[str, str, str, str]:
        """Get the dialog values.

        Returns:
            Tuple of (name, folder, datasets_root, description)
        """
        return (
            self.name_edit.text().strip(),
            self.folder_edit.text().strip(),
            self.datasets_edit.text().strip(),
            self.description_edit.toPlainText().strip()
        )
