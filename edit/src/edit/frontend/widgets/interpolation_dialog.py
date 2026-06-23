from typing import Callable, Dict, List, Optional

from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)

from edit.frontend.utils.formats import format_object_identity


class InterpolationDialog(QDialog):
    def __init__(
        self,
        parent,
        object_ids: List[str],
        current_frame: int,
        total_frames: int,
        on_interpolate: Callable,
        preselect_object_id: Optional[str] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Interpolate / Re-track Annotations")
        self.setMinimumSize(400, 200)
        self.control_points: Dict[int, Dict] = {}  # {frame: bbox_data}
        self.on_interpolate = on_interpolate

        layout = QVBoxLayout()

        # Object selection
        form = QFormLayout()
        self.object_combo = QComboBox()
        self.object_combo.setEditable(True)
        self.object_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.object_combo.lineEdit().setPlaceholderText("Select or type object ID")

        # Get recent object IDs from multiple frames
        if hasattr(parent, "state") and hasattr(
            parent.state, "historic_obj_frame_count"
        ):
            frame_limit = parent.state.historic_obj_frame_count
            current_frame = parent.video_controller.current_index()
            recent_identities = (
                parent.annotation_controller.get_frame_object_identities(
                    frame_limit=frame_limit, current_frame=current_frame
                )
            )
            for identity in sorted(recent_identities, key=lambda x: x["id"]):
                display_text = format_object_identity(identity)
                self.object_combo.addItem(display_text, userData=identity["id"])
        else:
            for identity in object_ids:
                if isinstance(identity, dict):
                    display_text = format_object_identity(identity)
                    self.object_combo.addItem(display_text, userData=identity["id"])
                else:
                    self.object_combo.addItem(str(identity), userData=str(identity))

        self.object_combo.setCurrentIndex(-1)  # No pre-selected item

        # Pre-select the requested object if provided
        if preselect_object_id:
            for i in range(self.object_combo.count()):
                if self.object_combo.itemData(i) == preselect_object_id:
                    self.object_combo.setCurrentIndex(i)
                    break

        # Method selection
        self.method_combo = QComboBox()
        self.method_combo.addItem("Linear interpolation", userData="linear")
        self.method_combo.addItem("Re-track forward (start → end)", userData="retrack_forward")
        self.method_combo.addItem("Re-track backward (end → start)", userData="retrack_backward")
        self.method_combo.currentIndexChanged.connect(self._update_help_text)

        # Helper text
        self.help_label = QLabel()
        self.help_label.setStyleSheet("font-style: italic; color: #666;")
        self.help_label.setWordWrap(True)

        form.addRow(QLabel("Object ID:"), self.object_combo)
        form.addRow(QLabel("Method:"), self.method_combo)
        form.addRow(self.help_label)

        # Range mode selection
        self.radio_entire = QRadioButton("Entire range")
        self.radio_range = QRadioButton("Frame range")
        self.radio_entire.setChecked(True)

        mode_group = QButtonGroup(self)
        mode_group.addButton(self.radio_entire)
        mode_group.addButton(self.radio_range)
        self.radio_range.toggled.connect(self._toggle_range_inputs)

        form.addRow(QLabel("Range:"), self.radio_entire)
        form.addRow(QLabel(""), self.radio_range)

        # Start frame: spinbox + "From first occurrence" button
        self.start_frame_spin = QSpinBox()
        self.start_frame_spin.setRange(0, total_frames - 1)
        self.start_frame_spin.setValue(current_frame)
        self.start_frame_spin.valueChanged.connect(self._validate_frames)
        self.start_frame_spin.setEnabled(False)

        self.first_frame_btn = QPushButton("↑ First")
        self.first_frame_btn.setToolTip("Set to the first frame this object appears in")
        self.first_frame_btn.setEnabled(False)
        self.first_frame_btn.clicked.connect(self._fill_first_frame)

        start_row = QHBoxLayout()
        start_row.addWidget(self.start_frame_spin)
        start_row.addWidget(self.first_frame_btn)
        form.addRow(QLabel("Start Frame:"), start_row)

        # End frame: spinbox + "To last occurrence" button
        self.end_frame_spin = QSpinBox()
        self.end_frame_spin.setRange(0, total_frames - 1)
        self.end_frame_spin.setValue(min(current_frame + 30, total_frames - 1))
        self.end_frame_spin.valueChanged.connect(self._validate_frames)
        self.end_frame_spin.setEnabled(False)

        self.last_frame_btn = QPushButton("↓ Last")
        self.last_frame_btn.setToolTip("Set to the last frame this object appears in")
        self.last_frame_btn.setEnabled(False)
        self.last_frame_btn.clicked.connect(self._fill_last_frame)

        end_row = QHBoxLayout()
        end_row.addWidget(self.end_frame_spin)
        end_row.addWidget(self.last_frame_btn)
        form.addRow(QLabel("End Frame:"), end_row)

        self.object_combo.currentIndexChanged.connect(self._update_first_last_btn_state)

        # Interpolate and cancel button
        self.interpolate_btn = QPushButton("Apply")
        self.interpolate_btn.clicked.connect(self._interpolate)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        layout.addLayout(form)
        layout.addWidget(self.interpolate_btn)
        layout.addWidget(cancel_btn)
        self.setLayout(layout)

        self._validate_frames()
        self._update_help_text()

    def _update_help_text(self):
        method = self.method_combo.currentData()
        if method == "linear":
            self.help_label.setText("Linearly interpolate bbox position/size between the two frames.")
        elif method == "retrack_forward":
            self.help_label.setText(
                "Run the tracker forward from the lower frame to the upper frame, overwriting intermediate frames."
            )
        elif method == "retrack_backward":
            self.help_label.setText(
                "Run the tracker backward from the upper frame toward the lower frame, overwriting intermediate frames."
            )

    def _validate_frames(self):
        if self.radio_entire.isChecked():
            self.interpolate_btn.setEnabled(True)
            return
        start = self.start_frame_spin.value()
        end = self.end_frame_spin.value()
        self.interpolate_btn.setEnabled(start != end)

    def _update_first_last_btn_state(self):
        """Enable first/last buttons only in frame-range mode with a valid object."""
        in_range_mode = self.radio_range.isChecked()
        has_object = bool(
            self.object_combo.currentData() or self.object_combo.currentText().strip()
        )
        enabled = in_range_mode and has_object
        self.first_frame_btn.setEnabled(enabled)
        self.last_frame_btn.setEnabled(enabled)

    def _toggle_range_inputs(self, checked: bool):
        self.start_frame_spin.setEnabled(checked)
        self.end_frame_spin.setEnabled(checked)
        self._update_first_last_btn_state()
        self._validate_frames()

    def _current_object_frames(self):
        """Return sorted frame list for the currently selected object, or []."""
        object_id = self.object_combo.currentData()
        if not object_id:
            object_id = self.object_combo.currentText().strip()
        if not object_id:
            return []
        try:
            return self.parent().annotation_controller.frames_for_object(object_id)
        except Exception:
            return []

    def _fill_first_frame(self):
        frames = self._current_object_frames()
        if frames:
            self.start_frame_spin.setValue(int(frames[0]))

    def _fill_last_frame(self):
        frames = self._current_object_frames()
        if frames:
            self.end_frame_spin.setValue(int(frames[-1]))

    def _interpolate(self):
        object_id = self.object_combo.currentData()
        if not object_id:
            object_id = self.object_combo.currentText()
            if not object_id:
                QMessageBox.warning(
                    self, "Invalid Input", "Please select or enter an Object ID."
                )
                return

        method = self.method_combo.currentData()
        if self.radio_entire.isChecked():
            try:
                object_frames = self.parent().annotation_controller.frames_for_object(
                    object_id
                )
            except Exception as exc:
                QMessageBox.warning(
                    self,
                    "Object Not Found",
                    str(exc),
                )
                return
            if not object_frames:
                QMessageBox.warning(
                    self,
                    "Object Not Found",
                    f"Object {object_id} has no annotated frames.",
                )
                return
            start_frame = int(object_frames[0])
            end_frame = int(object_frames[-1])
        else:
            frame_a = self.start_frame_spin.value()
            frame_b = self.end_frame_spin.value()

            # Normalize so start_frame <= end_frame for all methods
            start_frame = min(frame_a, frame_b)
            end_frame = max(frame_a, frame_b)

            # Verify object exists in both boundary frames
            for check_frame in (start_frame, end_frame):
                active_objs = self.parent().annotation_controller.get_active_objects(
                    check_frame
                )
                if not any(obj["id"] == object_id for obj in active_objs):
                    QMessageBox.warning(
                        self,
                        "Object Not Found",
                        f"Object {object_id} not found in frame {check_frame}",
                    )
                    return

        if end_frame - start_frame <= 1:
            QMessageBox.warning(
                self,
                "Invalid Range",
                "The selected range must include at least one intermediate frame.",
            )
            return

        self.on_interpolate(object_id, start_frame, end_frame, method)
        # For re-track methods the sidebar provides its own messaging;
        # only show the completion dialog for linear interpolation.
        if method == "linear":
            QMessageBox.information(
                self,
                "Interpolation Complete",
                f"Interpolated {object_id} from frame {start_frame} to {end_frame}.",
            )
        self.accept()
