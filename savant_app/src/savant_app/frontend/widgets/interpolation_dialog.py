from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QLabel,
    QSpinBox,
    QComboBox,
    QPushButton,
    QGroupBox,
    QMessageBox
)
from PyQt6.QtCore import Qt
from typing import List, Dict
from savant_app.controllers.annotation_controller import AnnotationController

class InterpolationDialog(QDialog):
    def __init__(
        self, 
        parent,
        object_ids: List[str],
        current_frame: int,
        total_frames: int,
        annotation_controller: AnnotationController
    ):
        super().__init__(parent)
        self.setWindowTitle("Interpolate Annotations")
        self.setMinimumSize(400, 200)
        self.annotation_controller = annotation_controller
        self.control_points: Dict[int, Dict] = {}  # {frame: bbox_data}
        
        layout = QVBoxLayout()
        
        # Object selection
        form = QFormLayout()
        self.object_combo = QComboBox()
        self.object_combo.addItems(object_ids)
        form.addRow(QLabel("Object ID:"), self.object_combo)
        
        # Frame selection
        self.start_frame_spin = QSpinBox()
        self.start_frame_spin.setRange(0, total_frames-1)
        self.start_frame_spin.setValue(current_frame)
        self.start_frame_spin.valueChanged.connect(self._validate_frames)
        form.addRow(QLabel("Start Frame:"), self.start_frame_spin)
        
        self.end_frame_spin = QSpinBox()
        self.end_frame_spin.setRange(0, total_frames-1)
        self.end_frame_spin.setValue(min(current_frame + 30, total_frames-1))
        self.end_frame_spin.valueChanged.connect(self._validate_frames)
        form.addRow(QLabel("End Frame:"), self.end_frame_spin)
        
        # Buttons
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
    
    """
    def _add_control_point(self):
        frame = self.control_frame_spin.value()
        start = self.start_frame_spin.value()
        end = self.end_frame_spin.value()
        
        if frame <= start or frame >= end:
            QMessageBox.warning(self, "Invalid Frame", 
                                "Control frame must be between start and end frames")
            return

        # Placeholder values - will be replaced with actual bbox data
        bbox_data = {
            'center_x': 0.0,
            'center_y': 0.0,
            'width': 0.0,
            'height': 0.0,
            'rotation': 0.0
        }
        self.control_points[frame] = bbox_data
        self.ctrl_points_list.addItem(f"Control at frame {frame}")
        self.del_ctrl_btn.setEnabled(True)
    """
        
    """
    def _remove_control_point(self):
        idx = self.ctrl_points_list.currentIndex()
        if idx < 0:
            return
            
        frame = list(self.control_points.keys())[idx]
        del self.control_points[frame]
        self.ctrl_points_list.removeItem(idx)
        if not self.control_points:
            self.del_ctrl_btn.setEnabled(False)
    """
    
    def _interpolate(self):
        object_id = self.object_combo.currentText()
        start_frame = self.start_frame_spin.value()
        end_frame = self.end_frame_spin.value()
        
        #try:
            # Format control points for backend
        formatted_ctrl = {}
        for prop in ['center_x', 'center_y', 'width', 'height', 'rotation']:
            # Collect property values from control points in frame order
            ctrl_vals = []
            for frame in sorted(self.control_points.keys()):
                ctrl_vals.append(self.control_points[frame][prop])
            formatted_ctrl[prop] = ctrl_vals
        
        # Call controller
        self.annotation_controller.interpolate_annotations(
            object_id,
            start_frame,
            end_frame,
            formatted_ctrl,
            "auto"  # annotator
        )
        
        QMessageBox.information(
            self, 
            "Interpolation Complete", 
            f"Interpolated {object_id} from frame {start_frame} to {end_frame}"
        )
        self.accept()
            
#        except Exception as e:
#            QMessageBox.critical(
#                self, 
#                "Interpolation Error", 
#                f"Failed to interpolate: {str(e)}"
#            )
