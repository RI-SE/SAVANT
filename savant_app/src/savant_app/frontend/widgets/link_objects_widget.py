from typing import List, Optional, Dict

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QDialog,
)

from savant_app.frontend.utils.ontology_utils import get_relation_labels
from savant_app.frontend.utils.settings_store import get_ontology_path


class RelationLinkerWidget(QDialog):
    # Signal emitted when user wants to create a relationship
    relationship_created = pyqtSignal(str, str, str)  # subject_id, object_id, relationship_type
    
    def __init__(self, current_objects: Optional[List[Dict]] = None, main_window=None, parent=None):
        super().__init__(parent)

        # Use current frame objects or empty list
        self.current_objects = current_objects if current_objects is not None else []
        self.main_window = main_window

        self.setup_ui()
        self.setWindowTitle("Create Relationship")
        self.resize(400, 250)

    def _format_object_display_text(self, obj: Dict) -> str:
        """Format object information for display in the combo boxes."""
        obj_id = obj.get("id", "")
        obj_name = obj.get("name", "")
        obj_type = obj.get("type", "")
        return f"{obj_name} ({obj_type}) [ID: {obj_id}]"

    def setup_ui(self):
        # Main Layout
        layout = QVBoxLayout(self)

        # Form Layout (Standard "Label: Input" rows)
        form_layout = QFormLayout()

        # 1. Subject Selection
        self.combo_subject = QComboBox()
        self.combo_subject.setEditable(True)
        self.combo_subject.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        placeholder_text = "Type or select object"
        self.combo_subject.lineEdit().setPlaceholderText(placeholder_text)
        self.combo_subject.setMinimumWidth(len(placeholder_text) * 10)
        
        # Add current objects to the combo box
        object_display_texts = [self._format_object_display_text(obj) for obj in self.current_objects]
        if object_display_texts:
            self.combo_subject.addItems(object_display_texts)
        self.combo_subject.setCurrentIndex(-1)

        # 2. Object Selection
        self.combo_object = QComboBox()
        self.combo_object.setEditable(True)
        self.combo_object.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.combo_object.lineEdit().setPlaceholderText(placeholder_text)
        self.combo_object.setMinimumWidth(len(placeholder_text) * 10)
        
        # Add current objects to the combo box
        if object_display_texts:
            self.combo_object.addItems(object_display_texts)
        self.combo_object.setCurrentIndex(-1)

        # 3. Type Selection
        self.combo_type = QComboBox()
        # Get relationship types from ontology, fallback to hardcoded values if needed
        if get_ontology_path():
            relation_types = get_relation_labels()
            if relation_types:
                self.combo_type.addItems(relation_types)
            else:
                self.combo_type.addItems([])
        form_layout.addRow("Subject (Active):", self.combo_subject)
        form_layout.addRow("Object (Passive):", self.combo_object)
        form_layout.addRow("Relation Type:", self.combo_type)

        layout.addLayout(form_layout)

        # 4. Buttons
        btn_layout = QHBoxLayout()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_create = QPushButton("Create Link")

        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_create)

        layout.addLayout(btn_layout)

        # Connections
        self.btn_create.clicked.connect(self.on_create)
        self.btn_cancel.clicked.connect(self.close)

    def _extract_object_id(self, display_text: str) -> str:
        """Extract object ID from display text."""
        # Display text format: "name (type) [ID: id]"
        # Extract the ID part between [ID: and ]
        start = display_text.find("[ID: ")
        if start == -1:
            return ""
        start += 5  # Length of "[ID: "
        end = display_text.find("]", start)
        if end == -1:
            return ""
        return display_text[start:end]

    def on_create(self):
        """Emit signal with relationship details"""
        subj_text = self.combo_subject.currentText()
        obj_text = self.combo_object.currentText()
        relationship_type = self.combo_type.currentText()
        
        # Extract object IDs from the display text
        subject_id = self._extract_object_id(subj_text)
        object_id = self._extract_object_id(obj_text)

        if not subject_id or not object_id:
            QMessageBox.warning(
                self, "Invalid Selection", "Please select valid objects for both subject and object."
            )
            return

        if subject_id == object_id:
            QMessageBox.warning(
                self, "Invalid Link", "Subject and Object cannot be the same."
            )
            return

        if not relationship_type:
            QMessageBox.warning(
                self, "Invalid Selection", "Please select a relationship type."
            )
            return

        # Emit signal with relationship details
        self.relationship_created.emit(subject_id, object_id, relationship_type)
        self.close()