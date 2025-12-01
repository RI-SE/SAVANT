from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QLabel,
    QSpinBox,
    QComboBox,
    QPushButton,
    QMessageBox,
)
from typing import List, Dict, Callable


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
        self.setWindowTitle("Interpolate Annotations")
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
            recent_ids = parent.annotation_controller.get_frame_object_ids(
                frame_limit=frame_limit, current_frame=current_frame
            )
            self.object_combo.addItems(sorted(set(recent_ids)))
        else:
            self.object_combo.addItems(object_ids)

        self.object_combo.setCurrentIndex(-1)  # No pre-selected item

        # Add helper text
        help_label = QLabel("Interpolate annotations between start and end frames")
        help_label.setStyleSheet("font-style: italic; color: #666;")
        help_label.setWordWrap(True)

        form.addRow(QLabel("Object ID:"), self.object_combo)
        form.addRow(help_label)

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
        self.interpolate_btn = QPushButton("Interpolate")
        self.interpolate_btn.clicked.connect(self._interpolate)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        layout.addLayout(form)
        layout.addWidget(self.interpolate_btn)
        layout.addWidget(cancel_btn)
        self.setLayout(layout)

    def _validate_frames(self):
        start = self.start_frame_spin.value()
        end = self.end_frame_spin.value()

        if start >= end:
            self.interpolate_btn.setEnabled(False)
        else:
            self.interpolate_btn.setEnabled(True)

    def _interpolate(self):
        object_id = self.object_combo.currentText()
        start_frame = self.start_frame_spin.value()
        end_frame = self.end_frame_spin.value()

        # Verify object exists in start frame
        active_objs = self.parent().annotation_controller.get_active_objects(
            start_frame
        )
        if not any(obj["id"] == object_id for obj in active_objs):
            QMessageBox.warning(
                self,
                "Object Not Found",
                f"Object {object_id} not found in frame {start_frame}",
            )
            return

        self.on_interpolate(object_id, start_frame, end_frame)

        # self.annotation_controller.interpolate_annotations(
        #    object_id,
        #    start_frame,
        #    end_frame,
        #    formatted_ctrl,
        #    "auto"  # annotator
        # )

        QMessageBox.information(
            self,
            "Interpolation Complete",
            f"Interpolated {object_id} from frame {start_frame} to {end_frame}",
        )
        self.accept()


#        except Exception as e:
#            QMessageBox.critical(
#                self,
#                "Interpolation Error",
#                f"Failed to interpolate: {str(e)}"
#            )
