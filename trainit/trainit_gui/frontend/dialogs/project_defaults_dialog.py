"""Dialog for editing project default configuration parameters."""

import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QCheckBox,
    QGroupBox,
    QScrollArea,
    QPushButton,
    QDialogButtonBox,
    QWidget,
    QMessageBox,
)
from PyQt6.QtCore import Qt

from ..help_texts import PARAMETER_HELP, GROUP_HELP
from ...models.project import ProjectDefaults

logger = logging.getLogger(__name__)


def _create_info_button(parent: QWidget, param_name: str) -> QPushButton:
    """Create an info button that shows help for the given parameter."""
    btn = QPushButton("\u24d8")  # Circled i character
    btn.setFixedSize(22, 22)
    btn.setToolTip("Click for help")
    btn.setStyleSheet(
        "QPushButton { border: none; color: #0066cc; font-size: 14px; }"
        "QPushButton:hover { color: #0044aa; }"
    )
    btn.clicked.connect(lambda: _show_help_dialog(parent, param_name))
    return btn


def _create_group_info_button(parent: QWidget, group_name: str) -> QPushButton:
    """Create an info button for a group box."""
    btn = QPushButton("\u24d8")
    btn.setFixedSize(22, 22)
    btn.setToolTip("About this section")
    btn.setStyleSheet(
        "QPushButton { border: none; color: #0066cc; font-size: 14px; }"
        "QPushButton:hover { color: #0044aa; }"
    )
    btn.clicked.connect(lambda: _show_group_help_dialog(parent, group_name))
    return btn


def _show_help_dialog(parent: QWidget, param_name: str) -> None:
    """Show a help dialog for a parameter."""
    help_info = PARAMETER_HELP.get(param_name)
    if help_info:
        msg = QMessageBox(parent)
        msg.setWindowTitle(f"Help: {help_info['title']}")
        msg.setText(help_info["text"])
        msg.setIcon(QMessageBox.Icon.Information)
        msg.exec()


def _show_group_help_dialog(parent: QWidget, group_name: str) -> None:
    """Show a help dialog for a group."""
    help_info = GROUP_HELP.get(group_name)
    if help_info:
        msg = QMessageBox(parent)
        msg.setWindowTitle(help_info["title"])
        msg.setText(help_info["text"])
        msg.setIcon(QMessageBox.Icon.Information)
        msg.exec()


def _get_tooltip(param_name: str) -> str:
    """Get tooltip text for a parameter."""
    help_info = PARAMETER_HELP.get(param_name, {})
    return help_info.get("tooltip", "")


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
        core_header = QHBoxLayout()
        core_header.addWidget(QLabel("Core Parameters"))
        core_header.addWidget(_create_group_info_button(self, "core"))
        core_header.addStretch()
        core_group.setTitle("")
        core_inner = QVBoxLayout()
        core_inner.addLayout(core_header)
        core_layout = QFormLayout()
        core_inner.addLayout(core_layout)
        core_group.setLayout(core_inner)

        self._add_optional_line(core_layout, "model", "Model:", "yolo11s-obb.pt")
        self._add_optional_spin(core_layout, "epochs", "Epochs:", 1, 1000, 50)
        self._add_optional_spin(core_layout, "imgsz", "Image Size:", 32, 2048, 640)
        self._add_optional_spin(core_layout, "batch", "Batch Size:", 1, 256, 30)
        self._add_optional_combo(
            core_layout, "device", "Device:", ["auto", "cuda", "mps", "cpu"]
        )
        self._add_optional_line(core_layout, "project", "Project:", "runs/obb")

        scroll_layout.addWidget(core_group)

        # Advanced parameters
        advanced_group = QGroupBox()
        advanced_header = QHBoxLayout()
        advanced_header.addWidget(QLabel("Advanced Training Parameters"))
        advanced_header.addWidget(_create_group_info_button(self, "advanced"))
        advanced_header.addStretch()
        advanced_inner = QVBoxLayout()
        advanced_inner.addLayout(advanced_header)
        advanced_layout = QFormLayout()
        advanced_inner.addLayout(advanced_layout)
        advanced_group.setLayout(advanced_inner)

        self._add_optional_double(
            advanced_layout, "lr0", "Initial LR:", 0.0001, 1.0, 0.01
        )
        self._add_optional_double(advanced_layout, "lrf", "Final LR:", 0.0, 1.0, 0.01)
        self._add_optional_combo(
            advanced_layout,
            "optimizer",
            "Optimizer:",
            ["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"],
        )
        self._add_optional_double(
            advanced_layout, "warmup_epochs", "Warmup Epochs:", 0, 10, 3.0
        )
        self._add_optional_double(
            advanced_layout, "warmup_momentum", "Warmup Momentum:", 0, 1, 0.8
        )
        self._add_optional_spin(advanced_layout, "patience", "Patience:", 1, 500, 50)
        self._add_optional_spin(
            advanced_layout, "save_period", "Save Period:", -1, 100, -1
        )
        self._add_optional_combo(
            advanced_layout, "cache", "Cache:", ["false", "true", "ram", "disk"]
        )
        self._add_optional_spin(advanced_layout, "workers", "Workers:", 0, 32, 8)
        self._add_optional_spin(
            advanced_layout, "close_mosaic", "Close Mosaic:", 0, 100, 10
        )
        self._add_optional_spin(advanced_layout, "freeze", "Freeze Layers:", 0, 100, 0)

        scroll_layout.addWidget(advanced_group)

        # Loss weights
        loss_group = QGroupBox()
        loss_header = QHBoxLayout()
        loss_header.addWidget(QLabel("Loss Weights"))
        loss_header.addWidget(_create_group_info_button(self, "loss"))
        loss_header.addStretch()
        loss_inner = QVBoxLayout()
        loss_inner.addLayout(loss_header)
        loss_layout = QFormLayout()
        loss_inner.addLayout(loss_layout)
        loss_group.setLayout(loss_inner)

        self._add_optional_double(loss_layout, "box", "Box Loss:", 0.1, 20.0, 7.5)
        self._add_optional_double(loss_layout, "cls", "Class Loss:", 0.1, 5.0, 0.5)
        self._add_optional_double(loss_layout, "dfl", "DFL Loss:", 0.1, 5.0, 1.5)

        scroll_layout.addWidget(loss_group)

        # Augmentation parameters
        aug_group = QGroupBox()
        aug_header = QHBoxLayout()
        aug_header.addWidget(QLabel("Augmentation Parameters"))
        aug_header.addWidget(_create_group_info_button(self, "augmentation"))
        aug_header.addStretch()
        aug_inner = QVBoxLayout()
        aug_inner.addLayout(aug_header)
        aug_layout = QFormLayout()
        aug_inner.addLayout(aug_layout)
        aug_group.setLayout(aug_inner)

        self._add_optional_double(aug_layout, "hsv_h", "HSV Hue:", 0, 1, 0.015)
        self._add_optional_double(aug_layout, "hsv_s", "HSV Saturation:", 0, 1, 0.7)
        self._add_optional_double(aug_layout, "hsv_v", "HSV Value:", 0, 1, 0.4)
        self._add_optional_double(aug_layout, "degrees", "Rotation:", 0, 180, 0.0)
        self._add_optional_double(aug_layout, "translate", "Translation:", 0, 1, 0.1)
        self._add_optional_double(aug_layout, "scale", "Scale:", 0, 2, 0.5)
        self._add_optional_double(aug_layout, "shear", "Shear:", 0, 90, 0.0)
        self._add_optional_double(
            aug_layout, "perspective", "Perspective:", 0, 0.01, 0.0
        )
        self._add_optional_double(aug_layout, "fliplr", "Horizontal Flip:", 0, 1, 0.5)
        self._add_optional_double(aug_layout, "flipud", "Vertical Flip:", 0, 1, 0.0)
        self._add_optional_double(aug_layout, "mosaic", "Mosaic:", 0, 1, 1.0)
        self._add_optional_double(aug_layout, "mixup", "Mixup:", 0, 1, 0.0)

        scroll_layout.addWidget(aug_group)

        # Train/Val Split defaults (just ratio and seed - enable/disable is in Generate dialog)
        split_group = QGroupBox()
        split_header = QHBoxLayout()
        split_header.addWidget(QLabel("Train/Val Split Defaults"))
        split_header.addWidget(_create_group_info_button(self, "split"))
        split_header.addStretch()
        split_inner = QVBoxLayout()
        split_inner.addLayout(split_header)
        split_layout = QFormLayout()
        split_inner.addLayout(split_layout)
        split_group.setLayout(split_inner)

        self._add_optional_double(
            split_layout, "split_ratio", "Train Ratio:", 0.5, 0.99, 0.9
        )
        self._add_optional_spin(split_layout, "split_seed", "Seed:", 0, 999999, 42)

        scroll_layout.addWidget(split_group)
        scroll_layout.addStretch()

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        # Add clear all button
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._clear_all)
        button_box.addButton(clear_btn, QDialogButtonBox.ButtonRole.ResetRole)

        layout.addWidget(button_box)

    def _add_optional_line(
        self, layout: QFormLayout, name: str, label: str, default: str
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
        edit.setToolTip(_get_tooltip(name))
        checkbox.stateChanged.connect(
            lambda state: edit.setEnabled(state == Qt.CheckState.Checked.value)
        )
        h_layout.addWidget(edit, stretch=1)

        info_btn = _create_info_button(self, name)
        h_layout.addWidget(info_btn)

        layout.addRow(label, container)
        self._widgets[name] = (checkbox, edit)

    def _add_optional_spin(
        self,
        layout: QFormLayout,
        name: str,
        label: str,
        min_val: int,
        max_val: int,
        default: int,
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
        spin.setToolTip(
            _get_tooltip(name) or f"Range: {min_val}-{max_val}, Default: {default}"
        )
        checkbox.stateChanged.connect(
            lambda state: spin.setEnabled(state == Qt.CheckState.Checked.value)
        )
        h_layout.addWidget(spin, stretch=1)

        info_btn = _create_info_button(self, name)
        h_layout.addWidget(info_btn)

        layout.addRow(label, container)
        self._widgets[name] = (checkbox, spin)

    def _add_optional_double(
        self,
        layout: QFormLayout,
        name: str,
        label: str,
        min_val: float,
        max_val: float,
        default: float,
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
        spin.setToolTip(
            _get_tooltip(name) or f"Range: {min_val}-{max_val}, Default: {default}"
        )
        checkbox.stateChanged.connect(
            lambda state: spin.setEnabled(state == Qt.CheckState.Checked.value)
        )
        h_layout.addWidget(spin, stretch=1)

        info_btn = _create_info_button(self, name)
        h_layout.addWidget(info_btn)

        layout.addRow(label, container)
        self._widgets[name] = (checkbox, spin)

    def _add_optional_combo(
        self, layout: QFormLayout, name: str, label: str, options: list
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
        combo.setToolTip(_get_tooltip(name))
        checkbox.stateChanged.connect(
            lambda state: combo.setEnabled(state == Qt.CheckState.Checked.value)
        )
        h_layout.addWidget(combo, stretch=1)

        info_btn = _create_info_button(self, name)
        h_layout.addWidget(info_btn)

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
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))

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
                elif isinstance(widget, QCheckBox):
                    data[name] = widget.isChecked()
            else:
                data[name] = None

        return ProjectDefaults(**data)
