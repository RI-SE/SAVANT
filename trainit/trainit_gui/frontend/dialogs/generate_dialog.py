"""Dialog for generating training configuration files."""

from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QMessageBox, QGroupBox,
    QCheckBox, QTextEdit
)

from ..states.app_state import AppState
from ...controllers.config_controller import ConfigController


class GenerateDialog(QDialog):
    """Dialog for generating training files."""

    def __init__(
        self,
        app_state: AppState,
        config_controller: ConfigController,
        parent=None
    ):
        super().__init__(parent)

        self.app_state = app_state
        self.config_controller = config_controller

        self.setWindowTitle("Generate Training Files")
        self.setMinimumWidth(600)

        self._setup_ui()
        self._update_preview()

    def _setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)

        # Config info
        info_group = QGroupBox("Configuration")
        info_layout = QFormLayout(info_group)

        config = self.app_state.current_config
        if config:
            info_layout.addRow("Config Name:", QLabel(config.name))
            info_layout.addRow(
                "Selected Datasets:",
                QLabel(", ".join(config.selected_datasets) or "None")
            )

        layout.addWidget(info_group)

        # Output settings
        output_group = QGroupBox("Output Settings")
        output_layout = QFormLayout(output_group)

        # Output directory
        dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Select output directory...")
        self.output_dir_edit.textChanged.connect(self._update_preview)
        dir_layout.addWidget(self.output_dir_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_output)
        dir_layout.addWidget(browse_btn)

        output_layout.addRow("Output Directory:", dir_layout)

        # Options
        self.copy_images_cb = QCheckBox("Copy images (instead of symlinks)")
        self.copy_images_cb.setToolTip(
            "If checked, images will be copied to output directory.\n"
            "Otherwise, symbolic links will be created (uses less disk space)."
        )
        output_layout.addRow("", self.copy_images_cb)

        layout.addWidget(output_group)

        # Preview
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)

        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(150)
        preview_layout.addWidget(self.preview_text)

        layout.addWidget(preview_group)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        self.generate_btn = QPushButton("Generate Files")
        self.generate_btn.clicked.connect(self._on_generate)
        self.generate_btn.setDefault(True)
        self.generate_btn.setEnabled(False)
        btn_layout.addWidget(self.generate_btn)

        layout.addLayout(btn_layout)

    def _browse_output(self):
        """Browse for output directory."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self.output_dir_edit.text() or ""
        )
        if folder:
            self.output_dir_edit.setText(folder)

    def _update_preview(self):
        """Update the preview text."""
        output_dir = self.output_dir_edit.text().strip()

        if not output_dir:
            self.preview_text.setText("Select an output directory...")
            self.generate_btn.setEnabled(False)
            return

        preview = self.config_controller.preview_generation(output_dir)

        if not preview.get('valid', False):
            self.preview_text.setText(f"Error: {preview.get('error', 'Unknown error')}")
            self.generate_btn.setEnabled(False)
            return

        text = f"""Output Directory: {preview['output_dir']}

Files to generate:
  - {Path(preview['dataset_yaml']).name}
  - {Path(preview['params_json']).name}

Summary:
  - Datasets: {', '.join(preview['datasets'])}
  - Training images: {preview['total_train_images']}
  - Validation images: {preview['total_val_images']}
  - Total objects: {preview['total_objects']}
  - Classes: {preview['num_classes']}
"""
        self.preview_text.setText(text)
        self.generate_btn.setEnabled(True)

    def _on_generate(self):
        """Handle generate button click."""
        output_dir = self.output_dir_edit.text().strip()

        if not output_dir:
            QMessageBox.warning(
                self,
                "Validation Error",
                "Output directory is required."
            )
            return

        # Create directory if needed
        output_path = Path(output_dir)
        if not output_path.exists():
            try:
                output_path.mkdir(parents=True)
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Error",
                    f"Failed to create output directory: {e}"
                )
                return

        # Generate files
        success, yaml_path, json_path = self.config_controller.generate_files(
            output_dir=output_dir,
            copy_images=self.copy_images_cb.isChecked()
        )

        if success:
            QMessageBox.information(
                self,
                "Success",
                f"Training files generated successfully!\n\n"
                f"Dataset YAML: {yaml_path}\n"
                f"Params JSON: {json_path}\n\n"
                f"To train, run:\n"
                f"train-yolo-obb --config {json_path}"
            )
            self.accept()
        else:
            # Error message already shown via app_state.error_occurred
            pass
