# edit/frontend/widgets/inspection_widgets.py
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class InspectionParamsDialog(QDialog):
    """Dialog to configure inspection parameters before starting."""

    def __init__(self, frame_count: int = 0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Perform Inspection")
        self.setMinimumWidth(380)
        self._frame_count = max(0, frame_count)
        last_frame = max(0, self._frame_count - 1)

        self._ghost_spin = QSpinBox()
        self._ghost_spin.setMinimum(1)
        self._ghost_spin.setMaximum(9999)
        self._ghost_spin.setValue(5)
        self._ghost_spin.setToolTip(
            "Objects annotated in at most this many frames are flagged as ghosts."
        )

        self._overlap_spin = QDoubleSpinBox()
        self._overlap_spin.setMinimum(0.0)
        self._overlap_spin.setMaximum(100.0)
        self._overlap_spin.setValue(50.0)
        self._overlap_spin.setSuffix(" %")
        self._overlap_spin.setDecimals(1)
        self._overlap_spin.setToolTip(
            "Flag frames where two bounding boxes overlap by at least this fraction."
        )

        # Frame range controls
        self._all_frames_cb = QCheckBox("Cover all frames")
        self._all_frames_cb.setChecked(True)

        self._start_spin = QSpinBox()
        self._start_spin.setMinimum(0)
        self._start_spin.setMaximum(last_frame)
        self._start_spin.setValue(0)
        self._start_spin.setEnabled(False)

        self._end_spin = QSpinBox()
        self._end_spin.setMinimum(0)
        self._end_spin.setMaximum(last_frame)
        self._end_spin.setValue(last_frame)
        self._end_spin.setEnabled(False)

        self._all_frames_cb.toggled.connect(self._on_all_frames_toggled)

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Start:"))
        range_row.addWidget(self._start_spin)
        range_row.addSpacing(12)
        range_row.addWidget(QLabel("End:"))
        range_row.addWidget(self._end_spin)
        range_row.addStretch()

        range_group = QGroupBox("Frame range")
        range_layout = QVBoxLayout(range_group)
        range_layout.addWidget(self._all_frames_cb)
        range_layout.addLayout(range_row)

        form = QFormLayout()
        form.addRow("Ghost detection – max frames:", self._ghost_spin)
        form.addRow("Double detection – min overlap:", self._overlap_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        ok_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_btn is not None:
            ok_btn.setText("Start inspection")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(range_group)
        layout.addWidget(buttons)

    def _on_all_frames_toggled(self, checked: bool) -> None:
        self._start_spin.setEnabled(not checked)
        self._end_spin.setEnabled(not checked)

    @property
    def max_ghost_frames(self) -> int:
        return self._ghost_spin.value()

    @property
    def overlap_percent(self) -> float:
        return self._overlap_spin.value()

    @property
    def start_frame(self) -> int:
        if self._all_frames_cb.isChecked():
            return 0
        return self._start_spin.value()

    @property
    def end_frame(self) -> int:
        if self._all_frames_cb.isChecked():
            return max(0, self._frame_count - 1)
        return self._end_spin.value()


class InspectionBar(QWidget):
    """Horizontal bar shown during inspection mode with navigation buttons."""

    prev_clicked = pyqtSignal()
    next_clicked = pyqtSignal()
    end_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            "InspectionBar { background-color: #3a2000; border-top: 1px solid #ff8800; }"
        )

        self._label = QLabel("Inspection mode")
        self._label.setStyleSheet("color: #ff8800; font-weight: bold; padding: 0 8px;")

        self._btn_prev = QPushButton("◀ Previous problem")
        self._btn_next = QPushButton("Next problem ▶")
        self._btn_end = QPushButton("End inspection")
        self._btn_end.setStyleSheet(
            "QPushButton { color: #ff4444; border: 1px solid #ff4444; "
            "border-radius: 4px; padding: 2px 8px; } "
            "QPushButton:hover { background-color: #4a0000; }"
        )

        self._btn_prev.clicked.connect(self.prev_clicked)
        self._btn_next.clicked.connect(self.next_clicked)
        self._btn_end.clicked.connect(self.end_clicked)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.addWidget(self._label)
        layout.addStretch()
        layout.addWidget(self._btn_prev)
        layout.addWidget(self._btn_next)
        layout.addSpacing(16)
        layout.addWidget(self._btn_end)

    def set_counts(self, ghost: int, double: int, total: int) -> None:
        """Update the summary label."""
        self._label.setText(
            f"Inspection mode – {total} problem frame(s)  "
            f"({ghost} ghost, {double} double detection)"
        )
