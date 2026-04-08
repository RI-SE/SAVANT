from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

# (shortcut, description) — grouped by category with separator rows.
_SHORTCUTS: list[tuple[str, str]] = [
    # -- Playback & Navigation --
    ("", "Playback & Navigation"),
    ("Space", "Advance one frame"),
    ("Shift+Space", "Go back one frame"),
    ("Ctrl+G", "Go to frame number"),
    ("Ctrl+B", "Toggle bookmark on current frame"),
    ("Ctrl+Shift+N", "Jump to next bookmark"),
    ("Ctrl+Shift+P", "Jump to previous bookmark"),
    # -- Undo / Redo --
    ("", "Undo / Redo"),
    ("Ctrl+S", "Save project"),
    ("Ctrl+Z", "Undo last action"),
    ("Ctrl+Y", "Redo last action"),
    # -- Zoom --
    ("", "Zoom"),
    ("Ctrl++", "Zoom in"),
    ("Ctrl+-", "Zoom out"),
    ("Ctrl+0", "Reset view (fit to default zoom)"),
    ("Ctrl+Scroll", "Zoom in / out"),
    ("Z", "Toggle rectangle zoom mode"),
    # -- Annotation editing --
    ("", "Annotation Editing"),
    ("Arrow keys", "Move selected bounding box / pan view when zoomed in"),
    ("Shift+Left / Right", "Rotate selected bounding box"),
    ("Ctrl+Shift+Left / Right", "Rotate selected bounding box (fine, 1/8 step)"),
    ("Shift+Click", "Select and zoom to bounding box (sidebar list)"),
    ("Tab", "Cycle to next bounding box (zoomed)"),
    ("Shift+Tab", "Cycle to previous bounding box (zoomed)"),
    ("Delete", "Delete selected bounding box or frame tag"),
    ("R", "Repeat last adjustment (re-apply same position/size/rotation delta)"),
    ("Shift+R", "Cascade delta forward — apply last delta to all subsequent frames"),
    ("Ctrl+R", "Cascade delta backward — apply last delta to all previous frames"),
    # -- Seek bar --
    ("", "Seek Bar"),
    ("Escape", "Cancel frame number input"),
]


class ShortcutsDialog(QDialog):
    """Dialog listing all keyboard shortcuts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setModal(True)
        self.resize(460, 480)

        layout = QVBoxLayout(self)

        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Shortcut", "Action"])
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        row = 0
        for shortcut, description in _SHORTCUTS:
            if shortcut == "":
                # Category header row
                table.insertRow(row)
                header_item = QTableWidgetItem(description)
                header_item.setFlags(Qt.ItemFlag.NoItemFlags)
                font = header_item.font()
                font.setBold(True)
                header_item.setFont(font)
                table.setItem(row, 0, header_item)
                table.setSpan(row, 0, 1, 2)
                row += 1
            else:
                table.insertRow(row)
                key_item = QTableWidgetItem(shortcut)
                key_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                desc_item = QTableWidgetItem(description)
                desc_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                table.setItem(row, 0, key_item)
                table.setItem(row, 1, desc_item)
                row += 1

        layout.addWidget(table)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        close_button = QPushButton("Close")
        close_button.setFixedWidth(100)
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)
