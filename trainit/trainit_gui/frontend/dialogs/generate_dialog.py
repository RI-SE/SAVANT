"""Dialog for generating training configuration files."""

from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QGroupBox,
    QCheckBox,
    QTextEdit,
    QDoubleSpinBox,
    QSpinBox,
    QWidget,
    QScrollArea,
)
from PyQt6.QtCore import Qt

from ..states.app_state import AppState
from ...controllers.config_controller import ConfigController


class GenerateDialog(QDialog):
    """Dialog for generating training files."""

    def __init__(
        self, app_state: AppState, config_controller: ConfigController, parent=None
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
                QLabel(", ".join(config.selected_datasets) or "None"),
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

        # Manifest option
        self.generate_manifest_cb = QCheckBox("Generate manifest with file hashes")
        self.generate_manifest_cb.setChecked(True)
        self.generate_manifest_cb.setToolTip(
            "Generate a JSON manifest with SHA256 hashes of all source files.\n"
            "Useful for validating that source files haven't changed."
        )
        self.generate_manifest_cb.stateChanged.connect(self._update_preview)
        output_layout.addRow("", self.generate_manifest_cb)

        layout.addWidget(output_group)

        # Split settings group
        split_group = QGroupBox("Train/Val Split Settings")
        split_layout = QVBoxLayout(split_group)

        # Enable checkbox
        self.split_enabled_cb = QCheckBox("Re-map train/val split during generation")
        self.split_enabled_cb.setToolTip(
            "When enabled, source files will be pooled and re-split.\n"
            "Uses stratified sampling by sequence when possible."
        )
        self.split_enabled_cb.stateChanged.connect(self._on_split_enabled_changed)
        split_layout.addWidget(self.split_enabled_cb)

        # Split parameters (initially disabled)
        self.split_params_widget = QWidget()
        split_params_layout = QFormLayout(self.split_params_widget)
        split_params_layout.setContentsMargins(20, 5, 0, 0)

        # Get default values from project defaults if set
        default_ratio = 0.9
        default_seed = 42
        if self.app_state.project and self.app_state.project.default_config:
            proj_defaults = self.app_state.project.default_config
            if proj_defaults.split_ratio is not None:
                default_ratio = proj_defaults.split_ratio
            if proj_defaults.split_seed is not None:
                default_seed = proj_defaults.split_seed

        self.split_ratio_spin = QDoubleSpinBox()
        self.split_ratio_spin.setRange(0.5, 0.99)
        self.split_ratio_spin.setSingleStep(0.05)
        self.split_ratio_spin.setValue(default_ratio)
        self.split_ratio_spin.setToolTip(
            "Fraction of data for training (e.g., 0.9 = 90% train)"
        )
        self.split_ratio_spin.valueChanged.connect(self._update_preview)
        split_params_layout.addRow("Train Ratio:", self.split_ratio_spin)

        self.split_seed_spin = QSpinBox()
        self.split_seed_spin.setRange(0, 999999)
        self.split_seed_spin.setValue(default_seed)
        self.split_seed_spin.setToolTip("Random seed for reproducible splits")
        self.split_seed_spin.valueChanged.connect(self._update_preview)
        split_params_layout.addRow("Seed:", self.split_seed_spin)

        # Per-dataset val inclusion (with scroll area for many datasets)
        self.val_pool_group = QGroupBox("Include val folder in pool")
        self.val_pool_group.setToolTip(
            "Select which datasets' validation folders should be included\n"
            "in the pool of files to be redistributed."
        )
        val_pool_outer = QVBoxLayout(self.val_pool_group)

        # Scroll area for checkboxes
        val_pool_scroll = QScrollArea()
        val_pool_scroll.setWidgetResizable(True)
        val_pool_scroll.setMaximumHeight(120)  # Limit height, scroll if needed
        val_pool_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        val_pool_content = QWidget()
        self.val_pool_layout = QVBoxLayout(val_pool_content)
        self.val_pool_layout.setContentsMargins(0, 0, 0, 0)
        self.val_pool_checkboxes: dict[str, QCheckBox] = {}

        # Populate checkboxes based on selected datasets
        config = self.app_state.current_config
        if config:
            for ds_name in config.selected_datasets:
                cb = QCheckBox(ds_name)
                cb.setChecked(config.include_val_in_pool.get(ds_name, False))
                cb.stateChanged.connect(self._update_preview)
                self.val_pool_checkboxes[ds_name] = cb
                self.val_pool_layout.addWidget(cb)

        self.val_pool_layout.addStretch()
        val_pool_scroll.setWidget(val_pool_content)
        val_pool_outer.addWidget(val_pool_scroll)

        # Select all / Deselect all buttons for val pool
        val_pool_btn_layout = QHBoxLayout()
        val_pool_select_all = QPushButton("Select All")
        val_pool_select_all.clicked.connect(self._on_val_pool_select_all)
        val_pool_btn_layout.addWidget(val_pool_select_all)

        val_pool_deselect_all = QPushButton("Deselect All")
        val_pool_deselect_all.clicked.connect(self._on_val_pool_deselect_all)
        val_pool_btn_layout.addWidget(val_pool_deselect_all)

        val_pool_outer.addLayout(val_pool_btn_layout)

        split_params_layout.addRow(self.val_pool_group)

        self.split_params_widget.setEnabled(False)
        split_layout.addWidget(self.split_params_widget)

        # Initialize from config if split is enabled
        if config and config.split_enabled:
            self.split_enabled_cb.setChecked(True)
            self.split_ratio_spin.setValue(config.split_ratio)
            self.split_seed_spin.setValue(config.split_seed)

        layout.addWidget(split_group)

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
            self, "Select Output Directory", self.output_dir_edit.text() or ""
        )
        if folder:
            self.output_dir_edit.setText(folder)

    def _on_split_enabled_changed(self, state):
        """Handle split enabled checkbox change."""
        enabled = state == 2  # Qt.CheckState.Checked
        self.split_params_widget.setEnabled(enabled)
        # Guard against being called before UI is fully set up
        if hasattr(self, "preview_text"):
            self._update_preview()

    def _on_val_pool_select_all(self):
        """Select all val pool checkboxes."""
        for cb in self.val_pool_checkboxes.values():
            cb.setChecked(True)

    def _on_val_pool_deselect_all(self):
        """Deselect all val pool checkboxes."""
        for cb in self.val_pool_checkboxes.values():
            cb.setChecked(False)

    def _update_preview(self):
        """Update the preview text."""
        output_dir = self.output_dir_edit.text().strip()

        if not output_dir:
            self.preview_text.setText("Select an output directory...")
            self.generate_btn.setEnabled(False)
            return

        # Update config with current UI settings before preview
        self._apply_ui_to_config()

        generate_manifest = self.generate_manifest_cb.isChecked()
        preview = self.config_controller.preview_generation(
            output_dir, generate_manifest=generate_manifest
        )

        if not preview.get("valid", False):
            self.preview_text.setText(f"Error: {preview.get('error', 'Unknown error')}")
            self.generate_btn.setEnabled(False)
            return

        files_list = [
            f"  - {Path(preview['dataset_yaml']).name}",
            f"  - {Path(preview['params_json']).name}",
        ]
        if generate_manifest and "manifest_json" in preview:
            files_list.append(f"  - {Path(preview['manifest_json']).name}")

        split_info = ""
        if preview.get("split_enabled"):
            split_info = f"\n  - Split: {preview['split_ratio']:.0%} train, seed={preview['split_seed']}"

        text = f"""Output Directory: {preview['output_dir']}

Files to generate:
{chr(10).join(files_list)}

Summary:
  - Datasets: {', '.join(preview['datasets'])}
  - Training images: {preview['total_train_images']}
  - Validation images: {preview['total_val_images']}
  - Total objects: {preview['total_objects']}
  - Classes: {preview['num_classes']}{split_info}
"""
        self.preview_text.setText(text)
        self.generate_btn.setEnabled(True)

    def _apply_ui_to_config(self):
        """Apply UI settings to the current config for preview/generation."""
        config = self.app_state.current_config
        if not config:
            return

        # Update split settings
        config.split_enabled = self.split_enabled_cb.isChecked()
        config.split_ratio = self.split_ratio_spin.value()
        config.split_seed = self.split_seed_spin.value()

        # Update val pool settings
        for ds_name, cb in self.val_pool_checkboxes.items():
            config.include_val_in_pool[ds_name] = cb.isChecked()

    def _on_generate(self):
        """Handle generate button click."""
        output_dir = self.output_dir_edit.text().strip()

        if not output_dir:
            QMessageBox.warning(
                self, "Validation Error", "Output directory is required."
            )
            return

        # Create directory if needed
        output_path = Path(output_dir)
        if not output_path.exists():
            try:
                output_path.mkdir(parents=True)
            except Exception as e:
                QMessageBox.warning(
                    self, "Error", f"Failed to create output directory: {e}"
                )
                return

        # Apply UI settings to config
        self._apply_ui_to_config()

        # Generate files
        generate_manifest = self.generate_manifest_cb.isChecked()
        result = self.config_controller.generate_files(
            output_dir=output_dir,
            copy_images=self.copy_images_cb.isChecked(),
            generate_manifest=generate_manifest,
        )

        if result[0]:  # success
            _, yaml_path, json_path, manifest_path = result

            manifest_msg = ""
            if manifest_path:
                manifest_msg = f"Manifest JSON: {manifest_path}\n"

            QMessageBox.information(
                self,
                "Success",
                f"Training files generated successfully!\n\n"
                f"Dataset YAML: {yaml_path}\n"
                f"Params JSON: {json_path}\n"
                f"{manifest_msg}\n"
                f"To train, run:\n"
                f"train-yolo-obb --config {json_path}",
            )
            self.accept()
        else:
            # Error message already shown via app_state.error_occurred
            pass
