# savant_app/frontend/widgets/cascade_dropdown.py
from enum import Enum

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QPushButton, QSizePolicy, QVBoxLayout, QWidget

from savant_app.frontend.theme.menu_styler import cascade_dropdown_css


class CascadeDirection(Enum):
    FORWARDS = "forwards"
    BACKWARDS = "backwards"


class CascadeDropdown(QWidget):
    """
    A dropdown widget that appears near annotations to provide cascade options.
    """

    applySizeToAll = pyqtSignal(object)
    applyRotationToAll = pyqtSignal(object)
    applySizeToFrameRange = pyqtSignal(object)
    applyRotationToFrameRange = pyqtSignal(object)
    cancelled = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

        # Initialize the direction state for cascading forwards/backwards
        self._current_direction = None

        # Set up the UI
        self._setup_ui()

        # Initially hidden
        self.hide()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(1)

        # Direction selection buttons
        self.forwards_btn = QPushButton("Forwards")
        self.forwards_btn.clicked.connect(
            lambda: self._select_direction(CascadeDirection.FORWARDS)
        )
        layout.addWidget(self.forwards_btn)

        self.backwards_btn = QPushButton("Backwards")
        self.backwards_btn.clicked.connect(
            lambda: self._select_direction(CascadeDirection.BACKWARDS)
        )
        layout.addWidget(self.backwards_btn)

        # Action-specific buttons
        self.apply_size_to_all_btn = QPushButton("Apply Size to All Frames")
        self.apply_size_to_all_btn.clicked.connect(self._on_apply_size_all)
        layout.addWidget(self.apply_size_to_all_btn)

        self.apply_rotation_to_all_btn = QPushButton("Apply Rotation to All Frames")
        self.apply_rotation_to_all_btn.clicked.connect(self._on_apply_rotation_all)
        layout.addWidget(self.apply_rotation_to_all_btn)

        self.apply_size_to_next_btn = QPushButton("Apply size to Next X Frames")
        self.apply_size_to_next_btn.clicked.connect(self._on_apply_size_next_frame)
        layout.addWidget(self.apply_size_to_next_btn)

        self.apply_rotation_to_next_btn = QPushButton("Apply Rotation to Next X Frames")
        self.apply_rotation_to_next_btn.clicked.connect(
            self._on_apply_rotation_next_frame
        )
        layout.addWidget(self.apply_rotation_to_next_btn)

        # Store action buttons for easy access
        self._action_buttons = [
            self.apply_size_to_all_btn,
            self.apply_rotation_to_all_btn,
            self.apply_size_to_next_btn,
            self.apply_rotation_to_next_btn,
        ]

        # Cancel button (always visible)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)
        layout.addWidget(self.cancel_btn)

        # Style the widget
        self.setStyleSheet(cascade_dropdown_css())

        self.setLayout(layout)

        # Set size policy
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)

        # Initially show only direction buttons and cancel
        self._show_direction_buttons()

    def show_at_position(self, x, y):
        """Show the dropdown at the specified position."""
        # Reset state to direction selection
        self._current_direction = None
        self._show_direction_buttons()
        # Position the widget
        self.move(int(x), int(y))

        # Show the widget
        self.show()
        self.raise_()

    def _on_apply_size_all(self):
        """Handle cascade size to all frames button click."""
        self.hide()
        self.applySizeToAll.emit(self._current_direction)

    def _on_apply_rotation_all(self):
        """Handle cascade rotation to all frames button click."""
        self.hide()
        self.applyRotationToAll.emit(self._current_direction)

    def _on_apply_size_next_frame(self):
        """Handle cascade size to next X frames button click."""
        self.hide()
        self.applySizeToFrameRange.emit(self._current_direction)

    def _on_apply_rotation_next_frame(self):
        """Handle cascade rotation to next X frames button click."""
        self.hide()
        self.applyRotationToFrameRange.emit(self._current_direction)

    def _on_cancel(self):
        """Handle cancel button click."""
        self.hide()
        self._current_direction = None
        self.cancelled.emit()

    def _show_direction_buttons(self):
        """Show only the direction selection buttons."""
        self.forwards_btn.setVisible(True)
        self.backwards_btn.setVisible(True)
        for btn in self._action_buttons:
            btn.setVisible(False)
        self.adjustSize()

    def _show_action_buttons(self):
        """Show only the action-specific buttons."""
        self.forwards_btn.setVisible(False)
        self.backwards_btn.setVisible(False)
        for btn in self._action_buttons:
            btn.setVisible(True)
            # Update button text to reflect selected direction
            original_text = (
                btn.text().replace(" (Forwards)", "").replace(" (Backwards)", "")
            )
            btn.setText(
                f"{original_text} ({self._current_direction.value.capitalize()})"
            )
        self.adjustSize()

    def _select_direction(self, direction: CascadeDirection):
        """Handle direction selection."""
        self._current_direction = direction
        print(f"Current direction in dropdown: {self._current_direction}")
        self._show_action_buttons()
