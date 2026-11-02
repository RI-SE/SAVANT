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

from edit.frontend.utils.project_config import persist_bookmarks
from edit.frontend.utils.settings_store import (
    get_bookmark_notes,
    get_bookmarks,
    set_bookmark_note,
    set_bookmarks,
)


class BookmarkManagerDialog(QDialog):
    """Dialog for viewing, jumping to, editing notes, and deleting bookmarked frames."""

    def __init__(self, parent=None, frame_count: int = 0):
        super().__init__(parent)
        self.setWindowTitle("Bookmarks")
        self.setModal(True)
        self.resize(400, 420)

        self._selected_frame: int | None = None
        self._frame_count = frame_count

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

    def _is_past_end(self, frame: int) -> bool:
        """Return whether a stored bookmark lies beyond the open video's last frame."""
        return frame >= self._frame_count

    def _item_label(self, frame: int, note: str) -> str:
        label = f"Frame {frame}" + (f"  —  {note}" if note else "")
        return label + ("  (past end of video)" if self._is_past_end(frame) else "")

    def _populate(self):
        self.list_widget.clear()
        notes = get_bookmark_notes()
        for frame in get_bookmarks():
            item = QListWidgetItem(self._item_label(frame, notes.get(frame, "")))
            item.setData(Qt.ItemDataRole.UserRole, frame)
            self.list_widget.addItem(item)

    def _on_selection_changed(self):
        selected = self.list_widget.selectedItems()
        if len(selected) == 1:
            frame = selected[0].data(Qt.ItemDataRole.UserRole)
            self.jump_button.setEnabled(not self._is_past_end(frame))
            notes = get_bookmark_notes()
            self.note_edit.setEnabled(True)
            self.note_edit.setText(notes.get(frame, ""))
        else:
            self.jump_button.setEnabled(True)
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
        note = self.note_edit.text().strip()
        # editingFinished also fires on focus loss, so skip the write when nothing changed.
        if note == get_bookmark_notes().get(frame):
            return
        set_bookmark_note(frame, note)
        selected[0].setText(self._item_label(frame, note))
        persist_bookmarks()

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
        persist_bookmarks()

    def _on_jump(self):
        selected_items = self.list_widget.selectedItems()
        if len(selected_items) != 1:
            return
        frame = selected_items[0].data(Qt.ItemDataRole.UserRole)
        if self._is_past_end(frame):
            return
        self._selected_frame = frame
        self.accept()

    @property
    def selected_frame(self) -> int | None:
        """Frame index chosen via 'Jump to', or None if dialog was closed."""
        return self._selected_frame
