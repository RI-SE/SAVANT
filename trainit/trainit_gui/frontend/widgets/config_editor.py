"""Config editor widget with grouped parameter forms."""

import logging
from typing import Optional, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QGroupBox, QScrollArea, QPushButton, QTextEdit
)
from PyQt6.QtCore import Qt

from ..states.app_state import AppState
from ...controllers.config_controller import ConfigController
from ...models.project import TrainingConfig

logger = logging.getLogger(__name__)


class ConfigEditor(QWidget):
    """Widget for editing training configuration parameters."""

    def __init__(
        self,
        app_state: AppState,
        config_controller: ConfigController,
        parent=None
    ):
        super().__init__(parent)

        self.app_state = app_state
        self.config_controller = config_controller
        self._updating = False
        self._widgets = {}  # field_name -> widget

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area for all parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Config info section
        info_group = QGroupBox("Configuration")
        info_layout = QFormLayout(info_group)

        self.name_label = QLabel("No configuration selected")
        self.name_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        info_layout.addRow("Name:", self.name_label)

        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(60)
        self.description_edit.setPlaceholderText("Optional description...")
        self.description_edit.textChanged.connect(self._on_value_changed)
        info_layout.addRow("Description:", self.description_edit)

        scroll_layout.addWidget(info_group)

        # Core parameters
        core_group = QGroupBox("Core Parameters")
        core_layout = QFormLayout(core_group)

        self._add_line_edit(core_layout, "model", "Model:", "yolo11s-obb.pt")
        self._add_spin_box(core_layout, "epochs", "Epochs:", 1, 1000, 50)
        self._add_spin_box(core_layout, "imgsz", "Image Size:", 32, 2048, 640)
        self._add_spin_box(core_layout, "batch", "Batch Size:", 1, 256, 30)
        self._add_combo_box(core_layout, "device", "Device:",
                           ["auto", "cuda", "mps", "cpu"])
        self._add_line_edit(core_layout, "project", "Project:", "runs/obb")

        scroll_layout.addWidget(core_group)

        # Advanced parameters (YOLO defaults shown as greyed values)
        advanced_group = QGroupBox("Advanced Training Parameters")
        advanced_layout = QFormLayout(advanced_group)

        self._add_optional_double(advanced_layout, "lr0", "Initial LR:", 0.0001, 1.0, 0.01)
        self._add_optional_double(advanced_layout, "lrf", "Final LR (fraction):", 0.0, 1.0, 0.01)
        self._add_combo_box(advanced_layout, "optimizer", "Optimizer:",
                           ["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"],
                           optional=True)
        self._add_optional_double(advanced_layout, "warmup_epochs", "Warmup Epochs:", 0, 10, 3.0)
        self._add_optional_double(advanced_layout, "warmup_momentum", "Warmup Momentum:", 0, 1, 0.8)
        self._add_optional_spin(advanced_layout, "patience", "Patience:", 1, 500, 50)
        self._add_optional_spin(advanced_layout, "save_period", "Save Period:", -1, 100, -1)
        self._add_combo_box(advanced_layout, "cache", "Cache:",
                           ["false", "true", "ram", "disk"],
                           optional=True)
        self._add_optional_spin(advanced_layout, "workers", "Workers:", 0, 32, 8)
        self._add_optional_spin(advanced_layout, "close_mosaic", "Close Mosaic:", 0, 100, 10)
        self._add_optional_spin(advanced_layout, "freeze", "Freeze Layers:", 0, 100, 0)

        scroll_layout.addWidget(advanced_group)

        # Loss weights (YOLO defaults)
        loss_group = QGroupBox("Loss Weights")
        loss_layout = QFormLayout(loss_group)

        self._add_optional_double(loss_layout, "box", "Box Loss:", 0.1, 20.0, 7.5)
        self._add_optional_double(loss_layout, "cls", "Class Loss:", 0.1, 5.0, 0.5)
        self._add_optional_double(loss_layout, "dfl", "DFL Loss:", 0.1, 5.0, 1.5)

        scroll_layout.addWidget(loss_group)

        # Augmentation parameters (YOLO defaults)
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

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.revert_btn = QPushButton("Revert")
        self.revert_btn.clicked.connect(self._on_revert)
        self.revert_btn.setEnabled(False)
        btn_layout.addWidget(self.revert_btn)

        self.save_btn = QPushButton("Save Configuration")
        self.save_btn.clicked.connect(self._on_save)
        self.save_btn.setEnabled(False)
        btn_layout.addWidget(self.save_btn)

        layout.addLayout(btn_layout)

    def _add_line_edit(
        self,
        layout: QFormLayout,
        name: str,
        label: str,
        default: str = ""
    ):
        """Add a line edit field."""
        widget = QLineEdit()
        widget.setText(default)
        widget.textChanged.connect(self._on_value_changed)
        layout.addRow(label, widget)
        self._widgets[name] = widget

    def _add_spin_box(
        self,
        layout: QFormLayout,
        name: str,
        label: str,
        min_val: int,
        max_val: int,
        default: int
    ):
        """Add a spin box field."""
        widget = QSpinBox()
        widget.setRange(min_val, max_val)
        widget.setValue(default)
        widget.valueChanged.connect(self._on_value_changed)
        layout.addRow(label, widget)
        self._widgets[name] = widget

    def _add_combo_box(
        self,
        layout: QFormLayout,
        name: str,
        label: str,
        options: list,
        optional: bool = False
    ):
        """Add a combo box field."""
        widget = QComboBox()
        if optional:
            widget.addItem("(default)")
        widget.addItems(options if not optional else options[1:] if options[0] == "auto" else options)
        widget.currentIndexChanged.connect(self._on_value_changed)
        layout.addRow(label, widget)
        self._widgets[name] = widget

    def _add_optional_spin(
        self,
        layout: QFormLayout,
        name: str,
        label: str,
        min_val: int,
        max_val: int,
        default: int = 0
    ):
        """Add an optional spin box with checkbox."""
        container = QWidget()
        h_layout = QHBoxLayout(container)
        h_layout.setContentsMargins(0, 0, 0, 0)

        checkbox = QCheckBox()
        checkbox.stateChanged.connect(self._on_value_changed)
        h_layout.addWidget(checkbox)

        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        spin.setEnabled(False)
        spin.valueChanged.connect(self._on_value_changed)
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
        default: float = 0.0
    ):
        """Add an optional double spin box with checkbox."""
        container = QWidget()
        h_layout = QHBoxLayout(container)
        h_layout.setContentsMargins(0, 0, 0, 0)

        checkbox = QCheckBox()
        checkbox.stateChanged.connect(self._on_value_changed)
        h_layout.addWidget(checkbox)

        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setDecimals(4)
        spin.setSingleStep(0.01)
        spin.setValue(default)
        spin.setEnabled(False)
        spin.valueChanged.connect(self._on_value_changed)
        checkbox.stateChanged.connect(lambda state: spin.setEnabled(state == Qt.CheckState.Checked.value))
        h_layout.addWidget(spin, stretch=1)

        layout.addRow(label, container)
        self._widgets[name] = (checkbox, spin)

    def _connect_signals(self):
        """Connect app state signals."""
        self.app_state.current_config_changed.connect(self._on_config_changed)
        self.app_state.config_dirty_changed.connect(self._on_dirty_changed)

    def _on_dirty_changed(self, dirty: bool):
        """Handle dirty state change."""
        self.revert_btn.setEnabled(dirty and self.app_state.current_config is not None)

    def _on_config_changed(self, config: Optional[TrainingConfig]):
        """Handle config change."""
        self._updating = True

        if config:
            self.name_label.setText(config.name)
            self.description_edit.setText(config.description)
            self.save_btn.setEnabled(True)

            # Load all values
            self._set_value("model", config.model)
            self._set_value("epochs", config.epochs)
            self._set_value("imgsz", config.imgsz)
            self._set_value("batch", config.batch)
            self._set_value("device", config.device)
            self._set_value("project", config.project)

            # Optional values
            self._set_optional_value("lr0", config.lr0)
            self._set_optional_value("lrf", config.lrf)
            self._set_optional_value("optimizer", config.optimizer)
            self._set_optional_value("warmup_epochs", config.warmup_epochs)
            self._set_optional_value("warmup_momentum", config.warmup_momentum)
            self._set_optional_value("patience", config.patience)
            self._set_optional_value("save_period", config.save_period)
            self._set_optional_value("cache", config.cache)
            self._set_optional_value("workers", config.workers)
            self._set_optional_value("close_mosaic", config.close_mosaic)
            self._set_optional_value("freeze", config.freeze)
            self._set_optional_value("box", config.box)
            self._set_optional_value("cls", config.cls)
            self._set_optional_value("dfl", config.dfl)
            self._set_optional_value("hsv_h", config.hsv_h)
            self._set_optional_value("hsv_s", config.hsv_s)
            self._set_optional_value("hsv_v", config.hsv_v)
            self._set_optional_value("degrees", config.degrees)
            self._set_optional_value("translate", config.translate)
            self._set_optional_value("scale", config.scale)
            self._set_optional_value("shear", config.shear)
            self._set_optional_value("perspective", config.perspective)
            self._set_optional_value("fliplr", config.fliplr)
            self._set_optional_value("flipud", config.flipud)
            self._set_optional_value("mosaic", config.mosaic)
            self._set_optional_value("mixup", config.mixup)

        else:
            self.name_label.setText("No configuration selected")
            self.description_edit.clear()
            self.save_btn.setEnabled(False)

        self._updating = False

    def _set_value(self, name: str, value: Any):
        """Set a widget value."""
        widget = self._widgets.get(name)
        if not widget:
            return

        if isinstance(widget, QLineEdit):
            widget.setText(str(value) if value else "")
        elif isinstance(widget, QSpinBox):
            widget.setValue(int(value) if value else 0)
        elif isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(value) if value else 0.0)
        elif isinstance(widget, QComboBox):
            idx = widget.findText(str(value))
            if idx >= 0:
                widget.setCurrentIndex(idx)

    def _set_optional_value(self, name: str, value: Any):
        """Set an optional widget value."""
        widget = self._widgets.get(name)
        if not widget:
            return

        if isinstance(widget, tuple):
            checkbox, spin = widget
            if value is not None:
                checkbox.setChecked(True)
                spin.setEnabled(True)
                if isinstance(spin, QSpinBox):
                    spin.setValue(int(value))
                else:
                    spin.setValue(float(value))
            else:
                checkbox.setChecked(False)
                spin.setEnabled(False)
        elif isinstance(widget, QComboBox):
            if value is not None:
                idx = widget.findText(str(value))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            else:
                widget.setCurrentIndex(0)  # (default)

    def _get_value(self, name: str) -> Any:
        """Get a widget value."""
        widget = self._widgets.get(name)
        if not widget:
            return None

        if isinstance(widget, QLineEdit):
            return widget.text()
        elif isinstance(widget, QSpinBox):
            return widget.value()
        elif isinstance(widget, QDoubleSpinBox):
            return widget.value()
        elif isinstance(widget, QComboBox):
            text = widget.currentText()
            if text == "(default)" or text == "auto":
                return "auto" if name == "device" else None
            return text

        return None

    def _get_optional_value(self, name: str) -> Optional[Any]:
        """Get an optional widget value."""
        widget = self._widgets.get(name)
        if not widget:
            return None

        if isinstance(widget, tuple):
            checkbox, spin = widget
            if checkbox.isChecked():
                if isinstance(spin, QSpinBox):
                    return spin.value()
                else:
                    return spin.value()
            return None
        elif isinstance(widget, QComboBox):
            text = widget.currentText()
            if text == "(default)":
                return None
            return text

        return None

    def _on_value_changed(self):
        """Handle value change in any widget."""
        if self._updating:
            return
        # Mark config as dirty
        if self.app_state.current_config:
            self.app_state.config_dirty = True

    def _on_save(self):
        """Save the current configuration."""
        if not self.app_state.current_config:
            return

        # Build updated config
        config = TrainingConfig(
            name=self.app_state.current_config.name,
            description=self.description_edit.toPlainText(),
            selected_datasets=list(self.app_state.selected_datasets),
            model=self._get_value("model"),
            epochs=self._get_value("epochs"),
            imgsz=self._get_value("imgsz"),
            batch=self._get_value("batch"),
            device=self._get_value("device") or "auto",
            project=self._get_value("project"),
            lr0=self._get_optional_value("lr0"),
            lrf=self._get_optional_value("lrf"),
            optimizer=self._get_optional_value("optimizer"),
            warmup_epochs=self._get_optional_value("warmup_epochs"),
            warmup_momentum=self._get_optional_value("warmup_momentum"),
            patience=self._get_optional_value("patience"),
            save_period=self._get_optional_value("save_period"),
            cache=self._get_optional_value("cache"),
            workers=self._get_optional_value("workers"),
            close_mosaic=self._get_optional_value("close_mosaic"),
            freeze=self._get_optional_value("freeze"),
            box=self._get_optional_value("box"),
            cls=self._get_optional_value("cls"),
            dfl=self._get_optional_value("dfl"),
            hsv_h=self._get_optional_value("hsv_h"),
            hsv_s=self._get_optional_value("hsv_s"),
            hsv_v=self._get_optional_value("hsv_v"),
            degrees=self._get_optional_value("degrees"),
            translate=self._get_optional_value("translate"),
            scale=self._get_optional_value("scale"),
            shear=self._get_optional_value("shear"),
            perspective=self._get_optional_value("perspective"),
            fliplr=self._get_optional_value("fliplr"),
            flipud=self._get_optional_value("flipud"),
            mosaic=self._get_optional_value("mosaic"),
            mixup=self._get_optional_value("mixup"),
        )

        self.config_controller.update_config(config)
        self.app_state.config_dirty = False

    def _on_revert(self):
        """Revert to saved config state."""
        if self.app_state.current_config:
            # Reload the current config from project (not from app_state which has unsaved changes)
            if self.app_state.project:
                saved_config = self.app_state.project.get_config_by_name(
                    self.app_state.current_config.name
                )
                if saved_config:
                    # Reload UI from saved config
                    self._on_config_changed(saved_config)
                    # Restore selected datasets from saved config
                    self.app_state.selected_datasets = list(saved_config.selected_datasets)
            self.app_state.config_dirty = False
