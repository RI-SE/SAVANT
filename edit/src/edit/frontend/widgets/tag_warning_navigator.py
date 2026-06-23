# edit/frontend/widgets/tag_warning_navigator.py
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QMenu,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
    QMessageBox,
)
from edit.frontend.theme.forms import style_checkbox

class TagWarningNavigator(QDialog):
    """Dialog to activate the Object Tag & Warnings explorer"""

    _warning_threshold: float = 0.4

    def __init__(self, frame_count: int = 0, parent=None, tag_options: dict[str, dict[str, bool]] | None = None):
        super().__init__(parent)
        self.setWindowTitle("Object Tag & Warnings Explorer")
        self.setModal(True)
        self.setMinimumWidth(600)

        page = QWidget(self)
        form = QFormLayout(page)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        
        # Object Tags group
        option_map = tag_options or {}
        self._object_tag_states: dict[str, bool] = dict(option_map.get("object", {}))

        tag_group = QGroupBox("Object Tags", self)
        tag_layout = QHBoxLayout(tag_group)
        tag_layout.setSpacing(24)
        tag_layout.addWidget(
            self._create_tag_dropdown(
                title="Object Tags",
                states=self._object_tag_states,
                empty_message="No object tags found",
            ),
        )

        form.addRow(tag_group)

        # Confidence group
        confidence_group = QGroupBox("Confidence Level Warnings", self)
        confidence_form = QFormLayout(confidence_group)

        self.warning_max_spin = QDoubleSpinBox()
        self.warning_max_spin.setRange(0.0, 1.0)
        self.warning_max_spin.setDecimals(2)
        self.warning_max_spin.setSingleStep(0.01)
        self.warning_max_spin.setValue(self._warning_threshold)
        self.warning_max_spin.setMinimumWidth(80)

        warning_row = QWidget(self)
        warning_layout = QHBoxLayout(warning_row)
        warning_layout.setContentsMargins(0, 0, 0, 0)
        warning_layout.addWidget(QLabel("Threshold:"))
        warning_layout.addWidget(self.warning_max_spin)
        self.warning_toggle_cb = QCheckBox()
        warning_toggle_label = QLabel("Display confidence warnings:  ")
        warning_toggle_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        warning_layout.addSpacing(24)
        warning_layout.addWidget(warning_toggle_label)
        warning_layout.addWidget(self.warning_toggle_cb)
        warning_layout.addStretch(1)
        confidence_form.addRow("", warning_row)

        self._previous_warning_range = (0, self._warning_threshold)
        self._normalize_ranges()

        form.addRow(confidence_group)

        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setText("Start Navigator View")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)


        lay = QVBoxLayout(self)
        lay.addWidget(page)
        lay.addWidget(buttons)


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


    def _normalize_ranges(self) -> None:
 
        self._previous_warning_range = (
            0.0,
            float(self.warning_max_spin.value()),
        )


    def values(self) -> dict:

        tag_options = {
            "object": dict(self._object_tag_states),
        }

        return {
            "warning_range": (
                float(0.0),
                float(self.warning_max_spin.value()),
            ),
            "show_warnings": bool(self.warning_toggle_cb.isChecked()),
            "tag_options": tag_options,
        }


    def accept(self) -> None:
        super().accept()


    def reject(self) -> None:
        super().reject()