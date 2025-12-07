"""Dialog for editing project default configuration parameters."""

import logging
from typing import Optional, Any

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QGroupBox, QScrollArea, QPushButton, QDialogButtonBox, QWidget
)
from PyQt6.QtCore import Qt

from ...models.project import ProjectDefaults

logger = logging.getLogger(__name__)


class ProjectDefaultsDialog(QDialog):
    """Dialog for editing project default training parameters."""

    def __init__(self, defaults: Optional[ProjectDefaults] = None, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Project Default Parameters")
        self.setMinimumSize(500, 600)

        self._defaults = defaults or ProjectDefaults()
        self._widgets = {}

        self._setup_ui()
        self._load_values()

    def _setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)

        # Info label
        info = QLabel(
            "Set default parameters for new configurations in this project.\n"
            "Unchecked parameters will use trainit defaults."
        )
        info.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(info)

        # Scroll area for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Core parameters
        core_group = QGroupBox("Core Parameters")
        core_layout = QFormLayout(core_group)

        self._add_optional_line(core_layout, "model", "Model:", "yolo11s-obb.pt")
        self._add_optional_spin(core_layout, "epochs", "Epochs:", 1, 1000, 50)
        self._add_optional_spin(core_layout, "imgsz", "Image Size:", 32, 2048, 640)
        self._add_optional_spin(core_layout, "batch", "Batch Size:", 1, 256, 30)
        self._add_optional_combo(core_layout, "device", "Device:",
                                 ["auto", "cuda", "mps", "cpu"])
        self._add_optional_line(core_layout, "project", "Project:", "runs/obb")

        scroll_layout.addWidget(core_group)

        # Advanced parameters
        advanced_group = QGroupBox("Advanced Training Parameters")
        advanced_layout = QFormLayout(advanced_group)

        self._add_optional_double(advanced_layout, "lr0", "Initial LR:", 0.0001, 1.0, 0.01)
        self._add_optional_double(advanced_layout, "lrf", "Final LR:", 0.0, 1.0, 0.01)
        self._add_optional_combo(advanced_layout, "optimizer", "Optimizer:",
                                 ["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"])
        self._add_optional_double(advanced_layout, "warmup_epochs", "Warmup Epochs:", 0, 10, 3.0)
        self._add_optional_double(advanced_layout, "warmup_momentum", "Warmup Momentum:", 0, 1, 0.8)
        self._add_optional_spin(advanced_layout, "patience", "Patience:", 1, 500, 50)
        self._add_optional_spin(advanced_layout, "save_period", "Save Period:", -1, 100, -1)
        self._add_optional_combo(advanced_layout, "cache", "Cache:",
                                 ["false", "true", "ram", "disk"])
        self._add_optional_spin(advanced_layout, "workers", "Workers:", 0, 32, 8)
        self._add_optional_spin(advanced_layout, "close_mosaic", "Close Mosaic:", 0, 100, 10)
        self._add_optional_spin(advanced_layout, "freeze", "Freeze Layers:", 0, 100, 0)

        scroll_layout.addWidget(advanced_group)

        # Loss weights
        loss_group = QGroupBox("Loss Weights")
        loss_layout = QFormLayout(loss_group)

        self._add_optional_double(loss_layout, "box", "Box Loss:", 0.1, 20.0, 7.5)
        self._add_optional_double(loss_layout, "cls", "Class Loss:", 0.1, 5.0, 0.5)
        self._add_optional_double(loss_layout, "dfl", "DFL Loss:", 0.1, 5.0, 1.5)

        scroll_layout.addWidget(loss_group)

        # Augmentation parameters
        aug_group = QGroupBox("Augmentation Parameters")
        aug_layout = QFormLayout(aug_group)

        self._add_optional_double(aug_layout, "hsv_h", "HSV Hue:", 0, 1, 0.015)
        self._add_optional_double(aug_layout, "hsv_s", "HSV Saturation:", 0, 1, 0.7)
        self._add_optional_double(aug_layout, "hsv_v", "HSV Value:", 0, 1, 0.4)
        self._add_optional_double(aug_layout, "degrees", "Rotation:", 0, 180, 0.0)
        self._add_optional_double(aug_layout, "translate", "Translation:", 0, 1, 0.1)
        self._add_optional_double(aug_layout, "scale", "Scale:", 0, 2, 0.5)
        self._add_optional_double(aug_layout, "shear", "Shear:", 0, 90, 0.0)
        self._add_optional_double(aug_layout, "perspective", "Perspective:", 0, 0.01, 0.0)
        self._add_optional_double(aug_layout, "fliplr", "Horizontal Flip:", 0, 1, 0.5)
        self._add_optional_double(aug_layout, "flipud", "Vertical Flip:", 0, 1, 0.0)
        self._add_optional_double(aug_layout, "mosaic", "Mosaic:", 0, 1, 1.0)
        self._add_optional_double(aug_layout, "mixup", "Mixup:", 0, 1, 0.0)

        scroll_layout.addWidget(aug_group)
        scroll_layout.addStretch()

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        # Add clear all button
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._clear_all)
        button_box.addButton(clear_btn, QDialogButtonBox.ButtonRole.ResetRole)

        layout.addWidget(button_box)

    def _add_optional_line(
        self,
        layout: QFormLayout,
        name: str,
        label: str,
        default: str
    ):
        """Add an optional line edit with checkbox."""
        container = QWidget()
        h_layout = QHBoxLayout(container)
        h_layout.setContentsMargins(0, 0, 0, 0)

        checkbox = QCheckBox()
        h_layout.addWidget(checkbox)

        edit = QLineEdit()
        edit.setText(default)
        edit.setEnabled(False)
        checkbox.stateChanged.connect(lambda state: edit.setEnabled(state == Qt.CheckState.Checked.value))
        h_layout.addWidget(edit, stretch=1)

        layout.addRow(label, container)
        self._widgets[name] = (checkbox, edit)

    def _add_optional_spin(
        self,
        layout: QFormLayout,
        name: str,
        label: str,
        min_val: int,
        max_val: int,
        default: int
    ):
        """Add an optional spin box with checkbox."""
        container = QWidget()
        h_layout = QHBoxLayout(container)
        h_layout.setContentsMargins(0, 0, 0, 0)

        checkbox = QCheckBox()
        h_layout.addWidget(checkbox)

        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        spin.setEnabled(False)
        checkbox.stateChanged.connect(lambda state: spin.setEnabled(state == Qt.CheckState.Checked.value))
        h_layout.addWidget(spin, stretch=1)

        layout.addRow(label, container)
        self._widgets[name] = (checkbox, spin)

    def _add_optional_double(
        self,
        layout: QFormLayout,
        name: str,
        label: str,
        min_val: float,
        max_val: float,
        default: float
    ):
        """Add an optional double spin box with checkbox."""
        container = QWidget()
        h_layout = QHBoxLayout(container)
        h_layout.setContentsMargins(0, 0, 0, 0)

        checkbox = QCheckBox()
        h_layout.addWidget(checkbox)

        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setDecimals(4)
        spin.setSingleStep(0.01)
        spin.setValue(default)
        spin.setEnabled(False)
        checkbox.stateChanged.connect(lambda state: spin.setEnabled(state == Qt.CheckState.Checked.value))
        h_layout.addWidget(spin, stretch=1)

        layout.addRow(label, container)
        self._widgets[name] = (checkbox, spin)

    def _add_optional_combo(
        self,
        layout: QFormLayout,
        name: str,
        label: str,
        options: list
    ):
        """Add an optional combo box with checkbox."""
        container = QWidget()
        h_layout = QHBoxLayout(container)
        h_layout.setContentsMargins(0, 0, 0, 0)

        checkbox = QCheckBox()
        h_layout.addWidget(checkbox)

        combo = QComboBox()
        combo.addItems(options)
        combo.setEnabled(False)
        checkbox.stateChanged.connect(lambda state: combo.setEnabled(state == Qt.CheckState.Checked.value))
        h_layout.addWidget(combo, stretch=1)

        layout.addRow(label, container)
        self._widgets[name] = (checkbox, combo)

    def _load_values(self):
        """Load current defaults into widgets."""
        for name, widget_tuple in self._widgets.items():
            checkbox, widget = widget_tuple
            value = getattr(self._defaults, name, None)

            if value is not None:
                checkbox.setChecked(True)
                widget.setEnabled(True)

                if isinstance(widget, QLineEdit):
                    widget.setText(str(value))
                elif isinstance(widget, QSpinBox):
                    widget.setValue(int(value))
                elif isinstance(widget, QDoubleSpinBox):
                    widget.setValue(float(value))
                elif isinstance(widget, QComboBox):
                    idx = widget.findText(str(value))
                    if idx >= 0:
                        widget.setCurrentIndex(idx)

    def _clear_all(self):
        """Clear all checkboxes."""
        for checkbox, widget in self._widgets.values():
            checkbox.setChecked(False)

    def get_defaults(self) -> ProjectDefaults:
        """Get the configured defaults."""
        data = {}

        for name, widget_tuple in self._widgets.items():
            checkbox, widget = widget_tuple

            if checkbox.isChecked():
                if isinstance(widget, QLineEdit):
                    data[name] = widget.text()
                elif isinstance(widget, QSpinBox):
                    data[name] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    data[name] = widget.value()
                elif isinstance(widget, QComboBox):
                    data[name] = widget.currentText()
            else:
                data[name] = None

        return ProjectDefaults(**data)
