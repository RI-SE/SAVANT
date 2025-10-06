# settings.py
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QDialogButtonBox,
    QWidget,
    QDoubleSpinBox,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QCheckBox,
    QPushButton,
    QColorDialog,
    QLabel,
    QLineEdit,
    QFileDialog,
    QMessageBox,
)
from PyQt6.QtCore import Qt
from pathlib import Path
from PyQt6.QtWidgets import QGroupBox


_DEFAULT_ONTOLOGY = (Path(__file__).resolve().parents[5] / "Tools" / "markit" /
                     "savant_ontology_1.0.0.ttl")
_ONTOLOGY_PATH: Path = _DEFAULT_ONTOLOGY

_ACTION_INTERVAL_OFFSET: int = 0

_DEFAULT_ONTOLOGY_NAMESPACE = "http://savant.ri.se/ontology#"

_ONTOLOGY_NAMESPACE: str = _DEFAULT_ONTOLOGY_NAMESPACE


def get_ontology_path() -> Path:
    """Return the current Turtle (.ttl) ontology path used for frame tags."""
    return _ONTOLOGY_PATH


def set_ontology_path(path: str | Path) -> None:
    """
    Update the ontology (.ttl) file path used for frame tags.

    Raises:
        ValueError if the file is invalid.
    """
    global _ONTOLOGY_PATH
    path = Path(path)
    if not path.is_file() or path.suffix.lower() != ".ttl":
        raise ValueError(f"Invalid ontology file: {path}")
    _ONTOLOGY_PATH = path


def get_action_interval_offset() -> int:
    """
    Return the default action interval offset (in frames).
    Used to prefill start/end around the current frame when adding a frame tag.
    """
    return int(_ACTION_INTERVAL_OFFSET)


def set_action_interval_offset(value: int) -> None:
    """
    Update the default action interval offset (in frames).

    Args:
        value: Non-negative integer number of frames.

    Raises:
        ValueError: If value is negative.
    """
    global _ACTION_INTERVAL_OFFSET
    interval = int(value)
    if interval < 0:
        raise ValueError("Action interval offset must be >= 0.")
    _ACTION_INTERVAL_OFFSET = interval


def get_ontology_namespace() -> str:
    """Return the base namespace IRI for the ontology."""
    return _ONTOLOGY_NAMESPACE


def set_ontology_namespace(ns: str) -> None:
    """
    Set the base namespace IRI for the ontology.

    Args:
        ns: A valid namespace URI ending with '#', '/' or ':'.
    """
    global _ONTOLOGY_NAMESPACE
    ns = str(ns).strip()
    if not ns:
        raise ValueError("Ontology namespace cannot be empty.")
    if not (ns.endswith("#") or ns.endswith("/") or ns.endswith(":")):
        raise ValueError(
            f"Ontology namespace '{ns}' must end with '#', '/' or ':'."
        )
    _ONTOLOGY_NAMESPACE = ns


class SettingsDialog(QDialog):
    def __init__(self, *, theme="System", zoom_rate=1.2, frame_count=100,
                 ontology_path: Path, action_interval_offset: int, parent=None):
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
        self.zoom_spin.setRange(1.0, 10.0)
        self.zoom_spin.setDecimals(1)
        self.zoom_spin.setSingleStep(0.1)
        self.zoom_spin.setSuffix(" X")
        self.zoom_spin.setValue(float(zoom_rate))
        general_form.addRow("Zoom rate:", self.zoom_spin)

        self.frame_count_spin = QSpinBox()
        self.frame_count_spin.setRange(1, 100000)
        self.frame_count_spin.setSingleStep(1)
        self.frame_count_spin.setSuffix(" frames")
        self.frame_count_spin.setValue(int(frame_count))
        self.frame_count_spin.setToolTip("Number of recent frames to analyze for object detection")
        general_form.addRow("Frame history:", self.frame_count_spin)

        form.addRow(general_group)

        # Annotators group
        annotators_group = QGroupBox("Annotators", self)
        annotators_form = QFormLayout(annotators_group)

        self.annotator_table = QTableWidget(0, 3, self)
        self.annotator_table.setHorizontalHeaderLabels(["Name", "Enabled", "Colour"])
        self.annotator_table.horizontalHeader().setStretchLastSection(True)
        self.annotator_table.verticalHeader().setVisible(False)
        self.annotator_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        # TODO: Change to retrieve names from config
        self._annotator = [
            {"name": "Chris", "enabled": True,  "colour": "#ff6666"},
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
        rows_height = sum(self.annotator_table.rowHeight(row)
                          for row in range(self.annotator_table.rowCount()))
        frame_height = 2 * self.annotator_table.frameWidth()
        total_height = header_height + rows_height + frame_height
        self.annotator_table.setFixedHeight(total_height)

        # Annotations & Ontology group
        ontology_group = QGroupBox("Annotations & Ontology", self)
        ontology_form = QFormLayout(ontology_group)

        self._ontology_edit = QLineEdit(self)
        self._ontology_edit.setReadOnly(True)
        self._ontology_edit.setText(str(get_ontology_path()))
        browse_btn = QPushButton("Browse…", self)
        browse_btn.clicked.connect(self._on_browse_ontology_clicked)
        ontology_form.addRow("Frame Tag Ontology:", self._ontology_edit)
        ontology_form.addRow("", browse_btn)

        self._namespace_edit = QLineEdit(self)
        self._namespace_edit.setPlaceholderText(_DEFAULT_ONTOLOGY_NAMESPACE)
        self._namespace_edit.setText(get_ontology_namespace())
        self._namespace_edit.editingFinished.connect(self._on_namespace_editing_finished)
        ontology_form.addRow("Ontology namespace:", self._namespace_edit)

        self._offset_spin = QSpinBox(self)
        self._offset_spin.setRange(0, 1_000_000)
        self._offset_spin.setSingleStep(1)
        self._offset_spin.setValue(int(get_action_interval_offset()))
        self._offset_spin.valueChanged.connect(self._on_offset_changed)
        ontology_form.addRow("Action interval offset (frames):", self._offset_spin)

        form.addRow(ontology_group)

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
        """
        Insert a new row into the annotator table for the given annotator.

        Args:
            annotator (dict): A dictionary containing:
                - "name" (str): Annotator's name.
                - "enabled" (bool): Whether the annotator is enabled.
                - "colour" (str): Hex color code for the annotator.

        This creates a read-only name cell, a checkbox in column 1,
        and a color button in column 2 that opens a color picker when clicked.
        """

        row = self.annotator_table.rowCount()
        self.annotator_table.insertRow(row)

        it = QTableWidgetItem(annotator["name"])
        it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.annotator_table.setItem(row, 0, it)

        cb = QCheckBox()
        cb.setChecked(bool(annotator.get("enabled", False)))
        self.annotator_table.setCellWidget(row, 1, cb)

        btn = QPushButton()
        btn.setText(annotator.get("colour", "#000000"))
        btn.setStyleSheet(f"background-color: {annotator.get('colour', '#000000')};")
        btn.clicked.connect(lambda _=False, r=row: self._pick_color_for_row(r))
        self.annotator_table.setCellWidget(row, 2, btn)

    def values(self) -> dict:
        """
        Collect the current settings from the dialog.

        Returns:
            dict: A dictionary with:
                - "zoom_rate" (float): The current zoom factor from the spinbox.
                - "frame_count" (int): The number of frames to look back for object IDs.
                - "Annotators" (list of dict): List of annotators, where each entry has:
                    * "name" (str): Annotator's name.
                    * "enabled" (bool): Checkbox state.
                    * "color" (str): Hex color chosen from the color button.
        """
        annotators = []
        rows = self.annotator_table.rowCount()
        for r in range(rows):
            name = self.annotator_table.item(r, 0).text()
            enabled = self.annotator_table.cellWidget(r, 1).isChecked()
            color_btn = self.annotator_table.cellWidget(r, 2)
            color_hex = color_btn.text()
            annotators.append({"name": name, "enabled": enabled, "color": color_hex})

        return {
            "zoom_rate": float(self.zoom_spin.value()),
            "previous_frame_count": int(self.frame_count_spin.value()),
            "Annotators": annotators,
        }

    def _pick_color_for_row(self, row: int):
        """
        Open a QColorDialog to select a new color for a specific row.

        Args:
            row (int): The row index of the annotator whose color is being updated.

        If the user picks a valid color, the corresponding button text and
        background color are updated to reflect the new selection.
        """
        current_btn = self.annotator_table.cellWidget(row, 2)
        col = QColorDialog.getColor()
        if col.isValid():
            hex_str = col.name()
            current_btn.setText(hex_str)
            current_btn.setStyleSheet(f"background-color: {hex_str};")

    def _on_browse_ontology_clicked(self) -> None:
        """
        Let the user pick a .ttl ontology; update module-level setting immediately.
        """
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Ontology (.ttl)",
            str(get_ontology_path().parent),
            "Turtle files (*.ttl)",
        )
        if not path:
            return
        set_ontology_path(path)
        self._ontology_edit.setText(str(get_ontology_path()))
        QMessageBox.information(self, "Ontology Updated",
                                "Frame tag ontology file has been updated.")

    def _on_offset_changed(self, value: int) -> None:
        try:
            set_action_interval_offset(int(value))
        except Exception as ex:
            QMessageBox.critical(self, "Invalid Offset", str(ex))
            self._offset_spin.blockSignals(True)
            self._offset_spin.setValue(get_action_interval_offset())
            self._offset_spin.blockSignals(False)

    def _section_label(self, title: str) -> QLabel:
        """
        Create a bold section header label for grouping settings.
        """
        lbl = QLabel(f"<b>{title}</b>", self)
        lbl.setTextFormat(Qt.TextFormat.RichText)
        return lbl

    def _on_namespace_editing_finished(self) -> None:
        """
        Validate and persist the ontology namespace when the user edits the field.
        """
        namespace = (self._namespace_edit.text() or "").strip()
        try:
            set_ontology_namespace(namespace)
        except Exception as e:
            QMessageBox.critical(self, "Invalid Namespace", str(e))
            self._namespace_edit.blockSignals(True)
            self._namespace_edit.setText(get_ontology_namespace())
            self._namespace_edit.blockSignals(False)
