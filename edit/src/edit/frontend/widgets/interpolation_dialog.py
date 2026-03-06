from typing import Callable, Dict, List

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QLabel,
    QMessageBox,
    QPushButton,
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

        # Frame selection
        self.start_frame_spin = QSpinBox()
        self.start_frame_spin.setRange(0, total_frames - 1)
        self.start_frame_spin.setValue(current_frame)
        self.start_frame_spin.valueChanged.connect(self._validate_frames)
        form.addRow(QLabel("Start Frame:"), self.start_frame_spin)

        self.end_frame_spin = QSpinBox()
        self.end_frame_spin.setRange(0, total_frames - 1)
        self.end_frame_spin.setValue(min(current_frame + 30, total_frames - 1))
        self.end_frame_spin.valueChanged.connect(self._validate_frames)
        form.addRow(QLabel("End Frame:"), self.end_frame_spin)

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
        start = self.start_frame_spin.value()
        end = self.end_frame_spin.value()
        self.interpolate_btn.setEnabled(start != end)

    def _interpolate(self):
        object_id = self.object_combo.currentData()
        if not object_id:
            object_id = self.object_combo.currentText()
            if not object_id:
                QMessageBox.warning(
                    self, "Invalid Input", "Please select or enter an Object ID."
                )
                return

        frame_a = self.start_frame_spin.value()
        frame_b = self.end_frame_spin.value()
        method = self.method_combo.currentData()

        # Normalize so start_frame <= end_frame for all methods
        start_frame = min(frame_a, frame_b)
        end_frame = max(frame_a, frame_b)

        # Verify object exists in both boundary frames
        for check_frame in (start_frame, end_frame):
            active_objs = self.parent().annotation_controller.get_active_objects(check_frame)
            if not any(obj["id"] == object_id for obj in active_objs):
                QMessageBox.warning(
                    self,
                    "Object Not Found",
                    f"Object {object_id} not found in frame {check_frame}",
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
