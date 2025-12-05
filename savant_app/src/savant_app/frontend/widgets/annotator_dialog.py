from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


class AnnotatorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.annotator_name = None
        self._has_centered = False
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Annotator Information")
        self.setMinimumWidth(300)

        layout = QVBoxLayout()

        # Label
        label = QLabel("Who is currently annotating?")
        layout.addWidget(label)

        # Text input
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter your name")
        layout.addWidget(self.name_input)

        # OK button
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept_input)
        layout.addWidget(ok_button)

        # Allow Enter key to submit
        self.name_input.returnPressed.connect(self.accept_input)

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

    def accept_input(self):
        name = self.name_input.text().strip()
        if name:
            self.annotator_name = name
            self.accept()
        else:
            QMessageBox.warning(self, "Warning", "Please enter a name.")

    def get_annotator_name(self):
        return self.annotator_name
