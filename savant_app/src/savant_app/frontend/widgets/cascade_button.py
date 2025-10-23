# savant_app/frontend/widgets/cascade_button.py
from PyQt6.QtWidgets import QWidget, QPushButton, QVBoxLayout
from PyQt6.QtCore import Qt, pyqtSignal
from savant_app.frontend.theme.menu_styler import cascade_button_css 


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
            cascade_button_css()
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
