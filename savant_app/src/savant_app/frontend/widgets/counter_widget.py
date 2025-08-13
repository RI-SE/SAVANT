from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import pyqtSignal

class CounterWidget(QWidget):
    increment_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.count = 0
        
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        self.display = QLabel("0")
        self.layout.addWidget(self.display)
        
        self.increment_btn = QPushButton("+")
        self.increment_btn.clicked.connect(self.increment_requested.emit)
        self.layout.addWidget(self.increment_btn)

    def update_display(self, value):
        self.display.setText(str(value))
