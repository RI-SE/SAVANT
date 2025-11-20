import sys
from dataclasses import dataclass
from typing import List, Optional

from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


# Keeping the data structure for compatibility with your data
@dataclass
class MockObject:
    uid: str
    name: str
    obj_type: str
    frame_start: int
    frame_end: int

    @property
    def display_text(self):
        return f"{self.name} ({self.obj_type}) [ID: {self.uid}]"


class RelationLinkerWidget(QWidget):
    def __init__(self, mock_objects: Optional[List[MockObject]] = None, parent=None):
        super().__init__(parent)

        # --- HARDCODED MOCK OBJECTS ---
        if not mock_objects:
            self.mock_objects = [
                MockObject("101", "Volvo FH16", "truck", 0, 500),
                MockObject("102", "Trailer Box", "trailer", 0, 500),
                MockObject("205", "Broken Sedan", "car", 200, 400),
                MockObject("300", "Car Carrier", "truck", 0, 600),
                MockObject("301", "Cargo Audi", "car", 0, 600),
            ]
        else:
            self.mock_objects = mock_objects

        self.setup_ui()
        self.setWindowTitle("Create Relationship")
        self.resize(400, 250)

    def setup_ui(self):
        # Main Layout
        layout = QVBoxLayout(self)

        # Form Layout (Standard "Label: Input" rows)
        form_layout = QFormLayout()

        # 1. Subject Selection
        self.combo_subject = QComboBox()
        self.combo_subject.addItems([o.display_text for o in self.mock_objects])

        # 2. Object Selection
        self.combo_object = QComboBox()
        self.combo_object.addItems([o.display_text for o in self.mock_objects])

        # 3. Type Selection
        self.combo_type = QComboBox()
        self.combo_type.addItems(["towing", "carrying"])

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

    def on_create(self):
        """Minimal validation on click"""
        subj_idx = self.combo_subject.currentIndex()
        obj_idx = self.combo_object.currentIndex()

        if subj_idx == -1 or obj_idx == -1:
            return

        if subj_idx == obj_idx:
            QMessageBox.warning(
                self, "Invalid Link", "Subject and Object cannot be the same."
            )
            return

        # Success - in real app, emit signal here
        QMessageBox.information(
            self, "Link Created", "The relationship has been created."
        )
        self.close()

    ### NEW FEATURE ###
    @staticmethod
    def open_linker_menu(parent_window, objects: Optional[List[MockObject]] = None):
        """
        Static helper to instantiate, attach, and show the linker widget.
        Attaching to parent_window ensures it persists (prevents garbage collection).
        """
        # Close existing instance if open
        if (
            hasattr(parent_window, "relation_linker_window")
            and parent_window.relation_linker_window
        ):
            parent_window.relation_linker_window.close()

        # Create new instance (will use hardcoded defaults if objects is None)
        widget = RelationLinkerWidget(objects)

        # Attach to parent to persist memory
        parent_window.relation_linker_window = widget

        # Show
        widget.show()
        return widget
