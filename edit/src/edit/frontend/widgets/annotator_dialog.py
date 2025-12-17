from typing import Iterable, Sequence

from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


class AnnotatorDialog(QDialog):
    def __init__(
        self,
        parent=None,
        annotator_names: Sequence[str] | None = None,
    ):
        super().__init__(parent)
        self.annotator_name = None
        self._has_centered = False
        self._annotator_names = list(annotator_names or [])
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Annotator Information")
        self.setMinimumWidth(300)

        layout = QVBoxLayout()

        # Label
        label = QLabel("Who is currently annotating?")
        layout.addWidget(label)

        # Text input + dropdown
        self.name_input = QComboBox()
        self.name_input.setEditable(True)
        self.name_input.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.name_input.setMaxVisibleItems(10)
        line_edit = self.name_input.lineEdit()
        if line_edit is not None:
            line_edit.setPlaceholderText("Enter your name or choose from list")
            line_edit.returnPressed.connect(self.accept_input)
        self._populate_annotator_options(self._annotator_names)
        self.name_input.setCurrentIndex(-1)
        layout.addWidget(self.name_input)

        # OK button
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept_input)
        layout.addWidget(ok_button)

        # Cancel button
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        layout.addWidget(cancel_button)

        self.setLayout(layout)

        # Set focus to input field
        self.name_input.setFocus()

    def showEvent(self, event):
        super().showEvent(event)
        if not self._has_centered:
            self._center_on_screen()
            self._has_centered = True

    def _center_on_screen(self):
        cursor_pos = QCursor.pos()
        screen = (
            QApplication.screenAt(cursor_pos)
            or self.screen()
            or QApplication.primaryScreen()
        )
        if screen is None:
            return
        window_handle = self.windowHandle()
        if window_handle is not None:
            window_handle.setScreen(screen)
        self.adjustSize()
        frame_geometry = self.frameGeometry()
        frame_geometry.moveCenter(screen.availableGeometry().center())
        self.move(frame_geometry.topLeft())

    def _populate_annotator_options(self, annotator_names: Iterable[str]) -> None:
        self.name_input.clear()
        normalized_names: list[str] = []
        seen: set[str] = set()
        for entry in annotator_names or []:
            candidate = (entry or "").strip()
            lowered = candidate.lower()
            if not candidate or lowered in seen:
                continue
            seen.add(lowered)
            normalized_names.append(candidate)
        if normalized_names:
            self.name_input.addItems(normalized_names)

    def accept_input(self):
        name = self.name_input.currentText().strip()
        if name:
            self.annotator_name = name
            self.accept()
        else:
            QMessageBox.warning(self, "Warning", "Please enter a name.")

    def get_annotator_name(self):
        return self.annotator_name
