from typing import Dict, List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from savant_app.frontend.types import Relationship


class RelationDeleterWidget(QDialog):
    # Emits a list of the relationships with relation ids to delete.
    relationships_deleted = pyqtSignal(list)

    def __init__(
        self,
        current_relationships: Optional[list[Relationship]] = None,
        parent=None,
    ):
        super().__init__(parent)

        self.current_relationships = (
            current_relationships if current_relationships is not None else []
        )

        self.setup_ui()
        self.setWindowTitle("Delete Relationships")
        self.resize(500, 400)  # Slightly larger to accommodate list view

    def _format_relationship_display_text(self, relation: Relationship) -> str:
        """
        Format relationship information for display in the list.
        Tries to use names if available, falls back to IDs.
        """
        # Extract  names if available, otherwise use IDs
        subject = relation.subject
        relation_type = relation.relationship_type
        obj = relation.object

        return f"{subject} --[{relation_type}]--> {obj}"

    def setup_ui(self):
        """Create the UI for the relationship deletion widget"""
        # Main Layout
        layout = QVBoxLayout(self)

        lbl_instruction = QLabel("Select the relationships you wish to remove:")
        layout.addWidget(lbl_instruction)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.list_widget.setAlternatingRowColors(True)

        # Populate the list
        for relationship in self.current_relationships:
            display_text = self._format_relationship_display_text(relationship)
            item = QListWidgetItem(display_text)

            item.setData(Qt.ItemDataRole.UserRole, relationship)

            # Make it checkable for multiple selection
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)

            self.list_widget.addItem(item)

        layout.addWidget(self.list_widget)

        # Action Buttons
        btn_layout = QHBoxLayout()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_delete = QPushButton("Delete Selected")

        self.btn_delete.setStyleSheet("QPushButton { color: #d32f2f; }")

        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_delete)

        layout.addLayout(btn_layout)

        # Connections
        self.btn_delete.clicked.connect(self.on_delete)
        self.btn_cancel.clicked.connect(self.close)

        # Optional: Allow clicking the row to toggle the checkbox
        self.list_widget.itemClicked.connect(self._on_item_clicked)

    def _on_item_clicked(self, item):
        """Handle relationship list item selection state when clicks are made."""
        current_state = item.checkState()
        new_state = (
            Qt.CheckState.Checked
            if current_state == Qt.CheckState.Unchecked
            else Qt.CheckState.Unchecked
        )
        item.setCheckState(new_state)

    def on_delete(self):
        """Collect selected items and emit signal"""
        items_to_delete = []

        # Iterate through list to find checked items
        for index in range(self.list_widget.count()):
            item = self.list_widget.item(index)
            if item.checkState() == Qt.CheckState.Checked:
                # Retrieve the original dict stored in UserRole
                relationship_data = item.data(Qt.ItemDataRole.UserRole)
                items_to_delete.append(relationship_data.id)

        if not items_to_delete:
            QMessageBox.information(
                self,
                "No Selection",
                "Please select at least one relationship to delete.",
            )
            return

        confirm = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete {len(items_to_delete)} relationship(s)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if confirm == QMessageBox.StandardButton.Yes:
            self.relationships_deleted.emit(items_to_delete)
            self.close()
