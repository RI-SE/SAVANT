from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QMenuBar
from PyQt6.QtGui import QAction
from ..controllers.example_controller import CounterController
from .widgets.counter_widget import CounterWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAVANT - Main Window")
        self.resize(1280, 720)

        self.example_controller = CounterController()  # store controller reference

        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)

        # Counter widget
        self.counter_widget = CounterWidget()
        self.main_layout.addWidget(self.counter_widget)

        # Connect widget signal to MainWindow slot
        self.counter_widget.increment_requested.connect(self.on_increment_requested)

        # Initialize menu bar
        self._create_menu_bar()

    def _create_menu_bar(self):
        menu_bar = QMenuBar(self)
        file_menu = menu_bar.addMenu("&File")

        exit_action = QAction("&Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        self.setMenuBar(menu_bar)

    # Slot to handle increment signal
    def on_increment_requested(self):
        # Ask controller to increment the counter
        new_value = self.example_controller.increment_counter()
        # Update the widget display
        self.counter_widget.update_display(new_value)
