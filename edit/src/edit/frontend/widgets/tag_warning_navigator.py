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
from edit.frontend.utils.settings_store import get_tag_options

class TagWarningNavigator(QDialog):
    """Dialog to activate the Object Tag & Warnings explorer"""

    _warning_threshold: float = 0.4

    def __init__(self, frame_count: int = 0, parent=None, tag_options: dict[str, dict[str, bool]] | None = None, on_generate_tags=None):
        super().__init__(parent)
        self.setWindowTitle("Object Tag & Warnings Explorer")
        self.setModal(True)
        self.setMinimumWidth(600)
        self._on_generate_tags = on_generate_tags
        self._tag_menu = None
        self._tag_dropdown = None
        self._update_tag_text = None

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

        # Generate additional tags button
        if self._on_generate_tags is not None:
            gen_group = QGroupBox("Automatic Tag Generation", self)
            gen_layout = QHBoxLayout(gen_group)
            gen_btn = QPushButton("Generate additional tags")
            gen_btn.setToolTip(
                "Run ghost/double-detection and store results as object tags"
            )
            gen_btn.clicked.connect(self._run_generate_tags)
            gen_layout.addWidget(gen_btn)
            gen_layout.addStretch(1)
            form.addRow(gen_group)

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
            # Use a placeholder but still set up menu infrastructure for later additions
            self._tag_dropdown = QToolButton(self)
            self._tag_dropdown.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            self._tag_menu = QMenu(self._tag_dropdown)
            self._tag_dropdown.setMenu(self._tag_menu)
            self._tag_dropdown.setText("Select tags to display…   ")
            layout.addWidget(self._tag_dropdown)
            self._update_tag_text = lambda: self._tag_dropdown.setText(
                "Select tags to display…   "
                if not any(states.values())
                else f"{sum(1 for v in states.values() if v)} selected to display"
            )
            return container

        self._tag_dropdown = QToolButton(self)
        self._tag_dropdown.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._tag_menu = QMenu(self._tag_dropdown)
        self._tag_dropdown.setMenu(self._tag_menu)

        def update_text():
            selected = [name for name, enabled in states.items() if enabled]
            self._tag_dropdown.setText(
                "Select tags to display…   "
                if not selected
                else f"{len(selected)} selected to display"
            )

        self._update_tag_text = update_text
        update_text()
        self._add_tag_menu_entries(sorted(states.keys(), key=lambda n: n.lower()))

        layout.addWidget(self._tag_dropdown)
        return container

    def _add_tag_menu_entries(self, names):
        """Add checkbox menu entries for the given tag names."""
        states = self._object_tag_states
        for name in names:
            widget = QWidget()
            widget_layout = QHBoxLayout(widget)
            widget_layout.setContentsMargins(5, 2, 5, 2)
            checkbox = QCheckBox(name, self)
            style_checkbox(checkbox)
            checkbox.setChecked(bool(states.get(name, False)))

            def handle_toggle(checked, key=name, box=checkbox):
                states[key] = bool(checked)
                box.blockSignals(True)
                box.setChecked(bool(checked))
                box.blockSignals(False)
                if self._update_tag_text:
                    self._update_tag_text()

            checkbox.toggled.connect(handle_toggle)
            widget_layout.addWidget(checkbox)
            action = QWidgetAction(self._tag_menu)
            action.setDefaultWidget(widget)
            self._tag_menu.addAction(action)

    def _run_generate_tags(self):
        """Call the generate tags callback, then sync new tags into the dropdown."""
        if self._on_generate_tags is not None:
            self._on_generate_tags()
        self._sync_new_object_tags()

    def _sync_new_object_tags(self):
        """Add any newly discovered object tags to the dropdown."""
        fresh = get_tag_options().get("object", {})
        new_names = sorted(
            [n for n in fresh if n not in self._object_tag_states],
            key=lambda n: n.lower(),
        )
        if not new_names:
            return
        for name in new_names:
            self._object_tag_states[name] = fresh.get(name, False)
        self._add_tag_menu_entries(new_names)
        if self._update_tag_text:
            self._update_tag_text()


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