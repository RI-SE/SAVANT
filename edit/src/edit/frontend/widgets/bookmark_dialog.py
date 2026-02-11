from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from edit.frontend.utils.settings_store import (
    get_bookmark_notes,
    get_bookmarks,
    set_bookmark_note,
    set_bookmarks,
)


class BookmarkManagerDialog(QDialog):
    """Dialog for viewing, jumping to, editing notes, and deleting bookmarked frames."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bookmarks")
        self.setModal(True)
        self.resize(400, 420)

        self._selected_frame: int | None = None

        layout = QVBoxLayout(self)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.list_widget.itemDoubleClicked.connect(self._on_jump)
        self.list_widget.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.list_widget)

        # Note editing row
        note_layout = QHBoxLayout()
        note_layout.addWidget(QLabel("Note:"))
        self.note_edit = QLineEdit()
        self.note_edit.setPlaceholderText("Select a bookmark to edit its note")
        self.note_edit.setEnabled(False)
        self.note_edit.editingFinished.connect(self._on_note_changed)
        self.note_edit.returnPressed.connect(self._on_note_return)
        note_layout.addWidget(self.note_edit)
        layout.addLayout(note_layout)

        button_layout = QHBoxLayout()
        self.jump_button = QPushButton("Jump to")
        self.jump_button.setAutoDefault(False)
        self.jump_button.clicked.connect(self._on_jump)
        button_layout.addWidget(self.jump_button)

        self.delete_button = QPushButton("Delete Selected")
        self.delete_button.setAutoDefault(False)
        self.delete_button.clicked.connect(self._on_delete)
        button_layout.addWidget(self.delete_button)

        self.close_button = QPushButton("Close")
        self.close_button.setAutoDefault(False)
        self.close_button.clicked.connect(self.reject)
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)

        self._populate()

    def _populate(self):
        self.list_widget.clear()
        notes = get_bookmark_notes()
        for frame in get_bookmarks():
            note = notes.get(frame, "")
            label = f"Frame {frame}" + (f"  —  {note}" if note else "")
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, frame)
            self.list_widget.addItem(item)

    def _on_selection_changed(self):
        selected = self.list_widget.selectedItems()
        if len(selected) == 1:
            frame = selected[0].data(Qt.ItemDataRole.UserRole)
            notes = get_bookmark_notes()
            self.note_edit.setEnabled(True)
            self.note_edit.setText(notes.get(frame, ""))
        else:
            self.note_edit.setEnabled(False)
            self.note_edit.clear()

    def _on_note_return(self):
        """Commit the note and move focus back to the list on Enter."""
        self._on_note_changed()
        self.list_widget.setFocus()

    def _on_note_changed(self):
        selected = self.list_widget.selectedItems()
        if len(selected) != 1:
            return
        frame = selected[0].data(Qt.ItemDataRole.UserRole)
        set_bookmark_note(frame, self.note_edit.text().strip())
        # Refresh the item label to reflect the new note
        note = self.note_edit.text().strip()
        label = f"Frame {frame}" + (f"  —  {note}" if note else "")
        selected[0].setText(label)

    def _on_delete(self):
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            return
        frames_to_remove = {
            item.data(Qt.ItemDataRole.UserRole) for item in selected_items
        }
        notes = get_bookmark_notes()
        remaining = {f: n for f, n in notes.items() if f not in frames_to_remove}
        set_bookmarks(remaining)
        self._populate()

    def _on_jump(self):
        selected_items = self.list_widget.selectedItems()
        if len(selected_items) != 1:
            return
        self._selected_frame = selected_items[0].data(Qt.ItemDataRole.UserRole)
        self.accept()

    @property
    def selected_frame(self) -> int | None:
        """Frame index chosen via 'Jump to', or None if dialog was closed."""
        return self._selected_frame
