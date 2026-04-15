"""Dialog for configuring spline-based angle interpolation."""

from typing import Callable, Optional

from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)


class SplineAngleDialog(QDialog):
    """Let the user configure and trigger spline angle interpolation."""

    def __init__(
        self,
        parent,
        object_id: str,
        total_frames: int,
        current_frame: int,
        on_apply: Callable[
            [str, float, Optional[int], Optional[int]], None
        ],
    ):
        super().__init__(parent)
        self.setWindowTitle("Spline Angle Interpolation")
        self.setMinimumSize(420, 280)
        self.object_id = object_id
        self.on_apply = on_apply

        layout = QVBoxLayout()
        form = QFormLayout()

        # Object ID (read-only)
        id_label = QLabel(object_id)
        id_label.setStyleSheet("font-weight: bold;")
        form.addRow(QLabel("Object ID:"), id_label)

        # Mode selection
        self.radio_entire = QRadioButton("Entire trajectory")
        self.radio_range = QRadioButton("Frame range")
        self.radio_entire.setChecked(True)

        mode_group = QButtonGroup(self)
        mode_group.addButton(self.radio_entire)
        mode_group.addButton(self.radio_range)
        self.radio_range.toggled.connect(self._toggle_range_inputs)

        form.addRow(QLabel("Mode:"), self.radio_entire)
        form.addRow(QLabel(""), self.radio_range)

        # Frame range inputs
        self.start_frame_spin = QSpinBox()
        self.start_frame_spin.setRange(0, total_frames - 1)
        self.start_frame_spin.setValue(current_frame)
        self.start_frame_spin.setEnabled(False)

        self.end_frame_spin = QSpinBox()
        self.end_frame_spin.setRange(0, total_frames - 1)
        self.end_frame_spin.setValue(min(current_frame + 100, total_frames - 1))
        self.end_frame_spin.setEnabled(False)

        form.addRow(QLabel("Start frame:"), self.start_frame_spin)
        form.addRow(QLabel("End frame:"), self.end_frame_spin)

        # Smoothing factor
        self.smoothing_spin = QDoubleSpinBox()
        self.smoothing_spin.setRange(0.0, 100000.0)
        self.smoothing_spin.setDecimals(2)
        self.smoothing_spin.setValue(0.0)
        self.smoothing_spin.setToolTip(
            "Smoothing parameter (s) for scipy.interpolate.splprep.\n"
            "0 = exact interpolation through all points.\n"
            "Larger values give smoother curves."
        )
        form.addRow(QLabel("Smoothing factor:"), self.smoothing_spin)

        # Warning
        warning_label = QLabel(
            "\u26a0 This will overwrite all existing bounding box "
            "angles for this object in the selected range."
        )
        warning_label.setStyleSheet("color: #cc6600; font-style: italic;")
        warning_label.setWordWrap(True)
        form.addRow(warning_label)

        layout.addLayout(form)

        # Buttons
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self._on_apply)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        layout.addWidget(self.apply_btn)
        layout.addWidget(cancel_btn)
        self.setLayout(layout)

    def _toggle_range_inputs(self, checked: bool) -> None:
        self.start_frame_spin.setEnabled(checked)
        self.end_frame_spin.setEnabled(checked)

    def _on_apply(self) -> None:
        smoothing = self.smoothing_spin.value()

        if self.radio_range.isChecked():
            start = self.start_frame_spin.value()
            end = self.end_frame_spin.value()
            if start >= end:
                QMessageBox.warning(
                    self,
                    "Invalid Range",
                    "Start frame must be less than end frame.",
                )
                return
            range_desc = f"frames {start}\u2013{end}"
        else:
            start = None
            end = None
            range_desc = "entire trajectory"

        reply = QMessageBox.warning(
            self,
            "Confirm Angle Overwrite",
            f"This will overwrite all bounding box angles for "
            f"object '{self.object_id}' ({range_desc}).\n\n"
            f"Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.on_apply(self.object_id, smoothing, start, end)
        self.accept()
