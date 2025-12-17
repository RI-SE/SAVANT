# settings.py

# manual ontology selection imports:
# from pathlib import Path
# from PyQt6.QtCore import Qt, pyqtSignal

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    # QFileDialog,  # Manual ontology picker
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QMenu,
    QWidgetAction,
)

from edit.frontend.theme.forms import style_checkbox
from edit.frontend.utils.settings_store import (
    get_action_interval_offset,
    get_error_range,
    get_movement_sensitivity,
    get_ontology_namespace,
    # get_ontology_path,  # manual ontology picker
    get_rotation_sensitivity,
    get_zoom_rate,
    get_show_errors,
    get_show_warnings,
    get_warning_range,
    set_action_interval_offset,
    set_movement_sensitivity,
    set_ontology_namespace,
    # set_ontology_path,  # Manual ontology picker
    set_rotation_sensitivity,
)


class SettingsDialog(QDialog):
    # Manual ontology selection support:
    # ontology_path_selected = pyqtSignal(str)
    def __init__(
        self,
        *,
        theme="System",
        zoom_rate: float | None = None,
        frame_count=100,
        # ontology_path: Path, # manual ontology parameter:
        action_interval_offset: int,
        tag_options: dict[str, dict[str, bool]] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setMinimumWidth(700)

        page = QWidget(self)
        form = QFormLayout(page)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        # General group
        general_group = QGroupBox("General", self)
        general_form = QFormLayout(general_group)

        self.zoom_spin = QDoubleSpinBox()
        self.zoom_spin.setRange(0.1, 10.0)
        self.zoom_spin.setDecimals(1)
        self.zoom_spin.setSingleStep(0.1)
        self.zoom_spin.setSuffix(" X")
        if zoom_rate is None:
            zoom_rate = get_zoom_rate()
        self.zoom_spin.setValue(float(zoom_rate))
        general_form.addRow("Zoom rate:", self.zoom_spin)

        self.frame_count_spin = QSpinBox()
        self.frame_count_spin.setRange(1, 100000)
        self.frame_count_spin.setSingleStep(1)
        self.frame_count_spin.setSuffix(" frames")
        self.frame_count_spin.setValue(int(frame_count))
        self.frame_count_spin.setToolTip(
            "Number of recent frames to analyze for object detection"
        )
        general_form.addRow("Frame history:", self.frame_count_spin)

        # Movement sensitivity
        self.movement_sensitivity_spin = QDoubleSpinBox()
        self.movement_sensitivity_spin.setRange(0.1, 10.0)
        self.movement_sensitivity_spin.setDecimals(2)
        self.movement_sensitivity_spin.setSingleStep(0.1)
        self.movement_sensitivity_spin.setValue(float(get_movement_sensitivity()))
        self.movement_sensitivity_spin.valueChanged.connect(
            self._on_movement_sensitivity_changed
        )
        general_form.addRow(
            "Annotation Movement sensitivity:", self.movement_sensitivity_spin
        )

        # Rotation sensitivity
        self.rotation_sensitivity_spin = QDoubleSpinBox()
        self.rotation_sensitivity_spin.setRange(0.1, 10.0)
        self.rotation_sensitivity_spin.setDecimals(2)
        self.rotation_sensitivity_spin.setSingleStep(0.1)
        self.rotation_sensitivity_spin.setValue(float(get_rotation_sensitivity()))
        self.rotation_sensitivity_spin.valueChanged.connect(
            self._on_rotation_sensitivity_changed
        )
        general_form.addRow(
            "Annotation Rotation sensitivity:", self.rotation_sensitivity_spin
        )

        form.addRow(general_group)

        # Annotators group
        annotators_group = QGroupBox("Annotators", self)
        annotators_form = QFormLayout(annotators_group)

        self.annotator_table = QTableWidget(0, 3, self)
        self.annotator_table.setHorizontalHeaderLabels(["Name", "Enabled", "Colour"])
        self.annotator_table.horizontalHeader().setStretchLastSection(True)
        self.annotator_table.verticalHeader().setVisible(False)
        self.annotator_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.annotator_table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        self.annotator_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self._annotator = [
            {"name": "Chris", "enabled": True, "colour": "#ff6666"},
            {"name": "Younis", "enabled": False, "colour": "#3aa3ff"},
            {"name": "Fredrik", "enabled": False, "colour": "#63ff5e"},
            {"name": "Thanh", "enabled": False, "colour": "#fff347"},
        ]

        annotators_form.addRow(self.annotator_table)
        form.addRow(annotators_group)

        self.annotator_table.clearContents()
        self.annotator_table.setRowCount(0)
        for p in self._annotator:
            self._add_annotator_row(p)
        self.annotator_table.resizeRowsToContents()
        header_height = self.annotator_table.horizontalHeader().height()
        rows_height = sum(
            self.annotator_table.rowHeight(row)
            for row in range(self.annotator_table.rowCount())
        )
        frame_height = 2 * self.annotator_table.frameWidth()
        total_height = header_height + rows_height + frame_height
        self.annotator_table.setFixedHeight(total_height)

        # Annotations & Ontology group
        ontology_group = QGroupBox("Annotations && Ontology", self)
        ontology_form = QFormLayout(ontology_group)

        # manual ontology selector:
        # self._ontology_edit = QLineEdit(self)
        # self._ontology_edit.setReadOnly(True)
        # browse_btn = QPushButton("Browse…", self)
        # browse_btn.clicked.connect(self._on_browse_ontology_clicked)
        # ontology_form.addRow("Frame Tag Ontology:", self._ontology_edit)
        # ontology_form.addRow("", browse_btn)

        self._namespace_edit = QLineEdit(self)
        self._namespace_edit.setPlaceholderText(get_ontology_namespace())
        self._namespace_edit.setText(get_ontology_namespace())
        self._namespace_edit.editingFinished.connect(
            self._on_namespace_editing_finished
        )
        ontology_form.addRow("Ontology namespace:", self._namespace_edit)

        self._offset_spin = QSpinBox(self)
        self._offset_spin.setRange(0, 1_000_000)
        self._offset_spin.setSingleStep(1)
        self._offset_spin.setValue(int(get_action_interval_offset()))
        self._offset_spin.valueChanged.connect(self._on_offset_changed)
        ontology_form.addRow("Action interval offset (frames):", self._offset_spin)

        form.addRow(ontology_group)

        # Frame & Object Tags group
        option_map = tag_options or {}
        self._frame_tag_states: dict[str, bool] = dict(option_map.get("frame", {}))
        self._object_tag_states: dict[str, bool] = dict(option_map.get("object", {}))

        tag_group = QGroupBox("Frame && Object Tags", self)
        tag_layout = QHBoxLayout(tag_group)
        tag_layout.setSpacing(24)
        tag_layout.addWidget(
            self._create_tag_dropdown(
                title="Frame Tags",
                states=self._frame_tag_states,
                empty_message="No frame tags found",
            ),
        )
        tag_layout.addWidget(
            self._create_tag_dropdown(
                title="Object Tags",
                states=self._object_tag_states,
                empty_message="No object tags found",
            ),
        )

        form.addRow(tag_group)

        # Confidence group
        confidence_group = QGroupBox("Confidence Issues", self)
        confidence_form = QFormLayout(confidence_group)

        warning_min, warning_max = get_warning_range()
        error_min, error_max = get_error_range()

        self.warning_min_spin = QDoubleSpinBox()
        self.warning_min_spin.setRange(0.0, 1.0)
        self.warning_min_spin.setDecimals(2)
        self.warning_min_spin.setSingleStep(0.01)
        self.warning_min_spin.setValue(float(warning_min))
        self.warning_min_spin.setMinimumWidth(80)

        self.warning_max_spin = QDoubleSpinBox()
        self.warning_max_spin.setRange(0.0, 1.0)
        self.warning_max_spin.setDecimals(2)
        self.warning_max_spin.setSingleStep(0.01)
        self.warning_max_spin.setValue(float(warning_max))
        self.warning_max_spin.setMinimumWidth(80)

        warning_row = QWidget(self)
        warning_layout = QHBoxLayout(warning_row)
        warning_layout.setContentsMargins(0, 0, 0, 0)
        warning_layout.addWidget(QLabel("Min:"))
        warning_layout.addWidget(self.warning_min_spin)
        warning_layout.addSpacing(12)
        warning_layout.addWidget(QLabel("Max:"))
        warning_layout.addWidget(self.warning_max_spin)
        self.warning_toggle_cb = QCheckBox()
        self.warning_toggle_cb.setChecked(bool(get_show_warnings()))
        style_checkbox(self.warning_toggle_cb)
        warning_toggle_label = QLabel("Display warnings:  ")
        warning_toggle_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        warning_layout.addSpacing(24)
        warning_layout.addWidget(warning_toggle_label)
        warning_layout.addWidget(self.warning_toggle_cb)
        warning_layout.addStretch(1)
        confidence_form.addRow("Warning range:", warning_row)

        self.error_min_spin = QDoubleSpinBox()
        self.error_min_spin.setRange(0.0, 1.0)
        self.error_min_spin.setDecimals(2)
        self.error_min_spin.setSingleStep(0.01)
        self.error_min_spin.setValue(float(error_min))
        self.error_min_spin.setMinimumWidth(80)

        self.error_max_spin = QDoubleSpinBox()
        self.error_max_spin.setRange(0.0, 1.0)
        self.error_max_spin.setDecimals(2)
        self.error_max_spin.setSingleStep(0.01)
        self.error_max_spin.setValue(float(error_max))
        self.error_max_spin.setMinimumWidth(80)

        error_row = QWidget(self)
        error_layout = QHBoxLayout(error_row)
        error_layout.setContentsMargins(0, 0, 0, 0)
        error_layout.addWidget(QLabel("Min:"))
        error_layout.addWidget(self.error_min_spin)
        error_layout.addSpacing(12)
        error_layout.addWidget(QLabel("Max:"))
        error_layout.addWidget(self.error_max_spin)
        self.error_toggle_cb = QCheckBox()
        self.error_toggle_cb.setChecked(bool(get_show_errors()))
        style_checkbox(self.error_toggle_cb)
        error_toggle_label = QLabel("Display errors:        ")
        error_toggle_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        error_layout.addSpacing(24)
        error_layout.addWidget(error_toggle_label)
        error_layout.addWidget(self.error_toggle_cb)
        error_layout.addStretch(1)
        confidence_form.addRow("Error range:", error_row)

        self.warning_toggle_cb.stateChanged.connect(self._on_warning_toggle_changed)
        self.error_toggle_cb.stateChanged.connect(self._on_error_toggle_changed)

        self._previous_warning_range = (float(warning_min), float(warning_max))
        self._previous_error_range = (float(error_min), float(error_max))
        self._normalize_ranges()
        form.addRow(confidence_group)

        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addWidget(page)
        lay.addWidget(buttons)

    def _add_annotator_row(self, annotator: dict):
        row = self.annotator_table.rowCount()
        self.annotator_table.insertRow(row)

        it = QTableWidgetItem(annotator["name"])
        it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.annotator_table.setItem(row, 0, it)

        cb = QCheckBox()
        cb.setChecked(bool(annotator.get("enabled", False)))
        style_checkbox(cb)
        cb_container = QWidget()
        cb_layout = QHBoxLayout(cb_container)
        cb_layout.setContentsMargins(0, 0, 0, 0)
        cb_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cb_layout.addWidget(cb)
        cb_container.setProperty("checkbox", cb)
        self.annotator_table.setCellWidget(row, 1, cb_container)

        btn = QPushButton()
        btn.setText(annotator.get("colour", "#000000"))
        btn.setStyleSheet(f"background-color: {annotator.get('colour', '#000000')};")
        btn.clicked.connect(lambda _=False, r=row: self._pick_color_for_row(r))
        self.annotator_table.setCellWidget(row, 2, btn)

    def values(self) -> dict:
        annotators = []
        rows = self.annotator_table.rowCount()
        for r in range(rows):
            name = self.annotator_table.item(r, 0).text()
            enabled_widget = self.annotator_table.cellWidget(r, 1)
            checkbox = None
            if isinstance(enabled_widget, QCheckBox):
                checkbox = enabled_widget
            elif enabled_widget is not None:
                checkbox = enabled_widget.property("checkbox")
                if checkbox is None:
                    checkbox = enabled_widget.findChild(QCheckBox)
            enabled = checkbox.isChecked() if isinstance(checkbox, QCheckBox) else False
            color_btn = self.annotator_table.cellWidget(r, 2)
            color_hex = color_btn.text()
            annotators.append({"name": name, "enabled": enabled, "color": color_hex})

        tag_options = {
            "frame": dict(self._frame_tag_states),
            "object": dict(self._object_tag_states),
        }

        return {
            "zoom_rate": float(self.zoom_spin.value()),
            "previous_frame_count": int(self.frame_count_spin.value()),
            "movement_sensitivity": float(self.movement_sensitivity_spin.value()),
            "rotation_sensitivity": float(self.rotation_sensitivity_spin.value()),
            "warning_range": (
                float(self.warning_min_spin.value()),
                float(self.warning_max_spin.value()),
            ),
            "error_range": (
                float(self.error_min_spin.value()),
                float(self.error_max_spin.value()),
            ),
            "show_warnings": bool(self.warning_toggle_cb.isChecked()),
            "show_errors": bool(self.error_toggle_cb.isChecked()),
            "Annotators": annotators,
            "tag_options": tag_options,
        }

    def _set_spin_value(self, spin: QDoubleSpinBox, value: float) -> None:
        spin.blockSignals(True)
        spin.setValue(float(value))
        spin.blockSignals(False)

    def _create_tag_dropdown(
        self, *, title: str, states: dict[str, bool], empty_message: str
    ) -> QWidget:
        container = QWidget(self)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        heading = QLabel(f"{title}:", self)
        heading.setStyleSheet("font-weight: bold;")
        layout.addWidget(heading)

        if not states:
            placeholder = QLabel(empty_message, self)
            placeholder.setEnabled(False)
            layout.addWidget(placeholder)
            return container

        dropdown = QToolButton(self)
        dropdown.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(dropdown)
        dropdown.setMenu(menu)

        def update_text():
            selected = [name for name, enabled in states.items() if enabled]
            dropdown.setText(
                "Select tags to display…   "
                if not selected
                else f"{len(selected)} selected to display"
            )

        update_text()

        for name in sorted(states.keys(), key=lambda n: n.lower()):
            widget = QWidget()
            widget_layout = QHBoxLayout(widget)
            widget_layout.setContentsMargins(5, 2, 5, 2)
            checkbox = QCheckBox(name, self)
            style_checkbox(checkbox)
            checkbox.setChecked(bool(states[name]))

            def handle_toggle(checked, key=name, box=checkbox):
                states[key] = bool(checked)
                box.blockSignals(True)
                box.setChecked(bool(checked))
                box.blockSignals(False)
                update_text()

            checkbox.toggled.connect(handle_toggle)
            widget_layout.addWidget(checkbox)
            action = QWidgetAction(menu)
            action.setDefaultWidget(widget)
            menu.addAction(action)

        layout.addWidget(dropdown)
        return container

    def _ranges_overlap(
        self, warning_range: tuple[float, float], error_range: tuple[float, float]
    ) -> bool:
        warning_min, warning_max = warning_range
        error_min, error_max = error_range
        return (warning_min < error_max) and (error_min < warning_max)

    def _normalize_ranges(self) -> None:
        warning_min = float(self.warning_min_spin.value())
        warning_max = float(self.warning_max_spin.value())
        error_min = float(self.error_min_spin.value())
        error_max = float(self.error_max_spin.value())

        if warning_min > warning_max:
            self._set_spin_value(self.warning_max_spin, warning_min)
            warning_max = warning_min
            QMessageBox.warning(
                self,
                "Invalid Warning Range",
                "The warning minimum value can not be greater than the maximum value."
                "\n\nThe minimum value will be set to the maximum value.",
            )
        if error_min > error_max:
            self._set_spin_value(self.error_min_spin, error_max)
            error_min = error_max
            QMessageBox.warning(
                self,
                "Invalid Error Range",
                "The error minimum value can not be greater than the maximum value."
                "\n\nThe minimum value will be set to the maximum value.",
            )

        warning_range = (warning_min, warning_max)
        error_range = (error_min, error_max)

        if (
            self.warning_toggle_cb.isChecked()
            and self.error_toggle_cb.isChecked()
            and self._ranges_overlap(warning_range, error_range)
        ):
            QMessageBox.warning(
                self,
                "Invalid Ranges",
                "Warning and error ranges must not overlap when both markers are enabled.",
            )
            self._set_spin_value(self.warning_min_spin, self._previous_warning_range[0])
            self._set_spin_value(self.warning_max_spin, self._previous_warning_range[1])
            self._set_spin_value(self.error_min_spin, self._previous_error_range[0])
            self._set_spin_value(self.error_max_spin, self._previous_error_range[1])
            return

        self._previous_warning_range = (
            float(self.warning_min_spin.value()),
            float(self.warning_max_spin.value()),
        )
        self._previous_error_range = (
            float(self.error_min_spin.value()),
            float(self.error_max_spin.value()),
        )

    def _on_warning_toggle_changed(self, state: int):
        if state == Qt.CheckState.Checked.value and self.error_toggle_cb.isChecked():
            warning_range = (
                float(self.warning_min_spin.value()),
                float(self.warning_max_spin.value()),
            )
            error_range = (
                float(self.error_min_spin.value()),
                float(self.error_max_spin.value()),
            )
            if self._ranges_overlap(warning_range, error_range):
                QMessageBox.warning(
                    self,
                    "Invalid Ranges",
                    "Warning and error ranges must not overlap when both markers are enabled.",
                )
                self.warning_toggle_cb.blockSignals(True)
                self.warning_toggle_cb.setChecked(False)
                self.warning_toggle_cb.blockSignals(False)
                self._normalize_ranges()
                return
        self._normalize_ranges()

    def _on_error_toggle_changed(self, state: int):
        if state == Qt.CheckState.Checked.value and self.warning_toggle_cb.isChecked():
            warning_range = (
                float(self.warning_min_spin.value()),
                float(self.warning_max_spin.value()),
            )
            error_range = (
                float(self.error_min_spin.value()),
                float(self.error_max_spin.value()),
            )
            if self._ranges_overlap(warning_range, error_range):
                QMessageBox.warning(
                    self,
                    "Invalid Ranges",
                    "Warning and error ranges must not overlap when both markers are enabled.",
                )
                self.error_toggle_cb.blockSignals(True)
                self.error_toggle_cb.setChecked(False)
                self.error_toggle_cb.blockSignals(False)
                self._normalize_ranges()
                return
        self._normalize_ranges()

    def _pick_color_for_row(self, row: int):
        current_btn = self.annotator_table.cellWidget(row, 2)
        col = QColorDialog.getColor()
        if col.isValid():
            hex_str = col.name()
            current_btn.setText(hex_str)
            current_btn.setStyleSheet(f"background-color: {hex_str};")

    # Manual ontology picker handler:
    # def _on_browse_ontology_clicked(self) -> None:
    #     path, _ = QFileDialog.getOpenFileName(
    #         self,
    #         "Select Ontology (.ttl)",
    #         str(Path.home()),
    #         "Turtle files (*.ttl)",
    #     )
    #     if not path:
    #         return
    #     set_ontology_path(path)
    #     self._ontology_edit.setText(str(get_ontology_path()))
    #     QMessageBox.information(
    #         self, "Ontology Updated", "Frame tag ontology file has been updated."
    #     )
    #     self.ontology_path_selected.emit(str(get_ontology_path()))

    def _on_offset_changed(self, value: int) -> None:
        try:
            set_action_interval_offset(int(value))
        except Exception as ex:
            QMessageBox.critical(self, "Invalid Offset", str(ex))
            self._offset_spin.blockSignals(True)
            self._offset_spin.setValue(get_action_interval_offset())
            self._offset_spin.blockSignals(False)

    def _on_movement_sensitivity_changed(self, value: float) -> None:
        set_movement_sensitivity(float(value))
        self.movement_sensitivity_spin.blockSignals(True)
        self.movement_sensitivity_spin.setValue(get_movement_sensitivity())
        self.movement_sensitivity_spin.blockSignals(False)

    def _on_rotation_sensitivity_changed(self, value: float) -> None:
        set_rotation_sensitivity(float(value))
        self.rotation_sensitivity_spin.blockSignals(True)
        self.rotation_sensitivity_spin.setValue(get_rotation_sensitivity())
        self.rotation_sensitivity_spin.blockSignals(False)

    def _section_label(self, title: str) -> QLabel:
        lbl = QLabel(f"<b>{title}</b>", self)
        lbl.setTextFormat(Qt.TextFormat.RichText)
        return lbl

    def _on_namespace_editing_finished(self) -> None:
        namespace = (self._namespace_edit.text() or "").strip()
        try:
            set_ontology_namespace(namespace)
        except Exception as e:
            QMessageBox.critical(self, "Invalid Namespace", str(e))
            self._namespace_edit.blockSignals(True)
            self._namespace_edit.setText(get_ontology_namespace())
            self._namespace_edit.blockSignals(False)

    def accept(self):
        self._normalize_ranges()
        super().accept()
