# savant_app/frontend/widgets/cascade_dropdown.py
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QSizePolicy
from PyQt6.QtCore import Qt, QTimer, pyqtSignal


class CascadeDropdown(QWidget):
    """
    A dropdown widget that appears near annotations to provide cascade options.
    """

    applyToAll = pyqtSignal()
    applyToFrameRange = pyqtSignal()
    cancelled = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

        # Set up the UI
        self._setup_ui()

        # Timer for auto-hiding
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide)

        # Default timeout (5 seconds)
        self.timeout = 5000

        # Initially hidden
        self.hide()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(1)

        # Apply size to all frames button
        #self.apply_all_btn = QPushButton("Apply Size to All Frames")
        #self.apply_all_btn.clicked.connect(self._on_apply_all)
        #layout.addWidget(self.apply_all_btn)

        self.apply_size_to_all_btn = QPushButton("Apply Size to All Frames")
        self.apply_size_to_all_btn.clicked.connect(self._on_apply_all)
        layout.addWidget(self.apply_size_to_all_btn)

        self.apply_rotation_to_all_btn = QPushButton("Apply Rotation to All Frames")
        self.apply_rotation_to_all_btn.clicked.connect(self._on_apply_all)
        layout.addWidget(self.apply_rotation_to_all_btn)

        # Apply size to next X frames button
        #self.apply_next_btn = QPushButton("Apply Size to Next X Frames")
        #self.apply_next_btn.clicked.connect(self._on_apply_next)
        #layout.addWidget(self.apply_next_btn)

        self.apply_size_to_next_btn = QPushButton("Apply size to Next X Frames")
        self.apply_size_to_next_btn.clicked.connect(self._on_apply_next)
        layout.addWidget(self.apply_size_to_next_btn)

        self.apply_rotation_to_next_btn = QPushButton("Apply Rotation to Next X Frames")
        self.apply_rotation_to_next_btn.clicked.connect(self._on_apply_next)
        layout.addWidget(self.apply_rotation_to_next_btn)
        

        # Cancel button
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)
        layout.addWidget(self.cancel_btn)

        # Style the widget
        self.setStyleSheet(
            """
            QWidget {
                background-color: #2d2d2d;
                border: 1px solid #555555;
                border-radius: 4px;
            }
            QPushButton {
                background-color: #3d3d3d;
                border: 1px solid #555555;
                border-radius: 3px;
                color: #ffffff;
                padding: 4px 8px;
                font-size: 12px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
            QPushButton:pressed {
                background-color: #1d1d1d;
            }
        """
        )

        self.setLayout(layout)

        # Set size policy
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)

    def show_at_position(self, x, y):
        """Show the dropdown at the specified position."""
        # Position the widget
        self.move(int(x), int(y))

        # Show the widget
        self.show()
        self.raise_()

        # Start the hide timer
        self._hide_timer.start(self.timeout)

    def hide(self):
        """Hide the dropdown and stop the timer."""
        self._hide_timer.stop()
        super().hide()

    def _on_apply_all(self):
        """Handle apply to all frames button click."""
        self.hide()
        self.applyToAll.emit()

    def _on_apply_next(self):
        """Handle apply to next X frames button click."""
        self.hide()
        self.applyToFrameRange.emit()

    def _on_cancel(self):
        """Handle cancel button click."""
        self.hide()
        self.cancelled.emit()

    def setTimeout(self, timeout_ms):
        """Set the timeout for auto-hiding in milliseconds."""
        self.timeout = timeout_ms
