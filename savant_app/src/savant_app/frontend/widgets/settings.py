# settings.py
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QDialogButtonBox,
    QWidget, QDoubleSpinBox, QSpinBox, QTableWidget, QTableWidgetItem,
    QCheckBox, QPushButton, QColorDialog, QLabel
)
from PyQt6.QtCore import Qt


class SettingsDialog(QDialog):
    def __init__(self, *, theme="System", zoom_rate=1.2, frame_count=100, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setMinimumWidth(380)

        page = QWidget(self)
        form = QFormLayout(page)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.annotator_table = QTableWidget(0, 3, self)
        self.annotator_table.setHorizontalHeaderLabels(["Name", "Enabled", "Colour"])
        self.annotator_table.horizontalHeader().setStretchLastSection(True)
        self.annotator_table.verticalHeader().setVisible(False)
        self.annotator_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        anno_tbl_lbl = QLabel("Project annotators")
        form.addRow(anno_tbl_lbl)
        form.addRow(self.annotator_table)

        # TODO: Change to retrieve names from config
        self._annotator = [
            {"name": "Chris", "enabled": True,  "colour": "#ff6666"},
            {"name": "Younis",   "enabled": False, "colour": "#3aa3ff"},
            {"name": "Fredrik",   "enabled": False, "colour": "#63ff5e"},
            {"name": "Thanh",   "enabled": False, "colour": "#fff347"},
        ]

        self.zoom_spin = QDoubleSpinBox()
        self.zoom_spin.setRange(1.0, 10.0)
        self.zoom_spin.setDecimals(1)
        self.zoom_spin.setSingleStep(0.1)
        self.zoom_spin.setSuffix(" X")
        self.zoom_spin.setValue(float(zoom_rate))
        form.addRow("Zoom rate:", self.zoom_spin)

        # Frame count input
        self.frame_count_spin = QSpinBox()
        self.frame_count_spin.setRange(1, 100000)
        self.frame_count_spin.setSingleStep(1)
        self.frame_count_spin.setSuffix(" frames")
        self.frame_count_spin.setValue(int(frame_count))
        self.frame_count_spin.setToolTip("Number of recent frames to analyze for object detection")
        form.addRow("Frame history:", self.frame_count_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addWidget(page)
        lay.addWidget(buttons)

        self.annotator_table.clearContents()
        self.annotator_table.setRowCount(0)

        for p in self._annotator:
            self._add_annotator_row(p)

        self.annotator_table.resizeRowsToContents()

        header_height = self.annotator_table.horizontalHeader().height()
        rows_height = sum(self.annotator_table.rowHeight(r)
                          for r in range(self.annotator_table.rowCount()))
        frame_height = 2 * self.annotator_table.frameWidth()

        total_height = header_height + rows_height + frame_height

        self.annotator_table.setFixedHeight(total_height)

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
            "Annotators": annotators
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