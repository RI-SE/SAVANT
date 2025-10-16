# savant_app/frontend/widgets/cascade_button.py
from PyQt6.QtWidgets import QWidget, QPushButton, QVBoxLayout
from PyQt6.QtCore import Qt, pyqtSignal


class CascadeButton(QWidget):
    """
    A simple button that appears after resizing an annotation.
    When clicked, it shows cascade options.
    """

    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Set up the UI
        self._setup_ui()

        # Initially hidden
        self.hide()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Cascade button
        self.cascade_btn = QPushButton("Cascade")
        self.cascade_btn.clicked.connect(self._on_clicked)
        self.cascade_btn.setFixedSize(60, 24)

        # Style the button
        self.cascade_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #0078d4;
                border: 1px solid #005a9e;
                border-radius: 4px;
                color: #ffffff;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1088e4;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
        """
        )

        layout.addWidget(self.cascade_btn)
        self.setLayout(layout)

    def show_at_position(self, x, y):
        """Show the button at the specified position."""
        # Position the widget
        self.move(int(x), int(y))

        # Show the widget
        self.show()
        self.raise_()

    def hide(self):
        """Hide the button."""
        super().hide()

    def _on_clicked(self):
        """Handle button click."""
        self.clicked.emit()
