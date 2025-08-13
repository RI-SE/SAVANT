from PyQt6.QtWidgets import QLabel, QSizePolicy
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

class VideoDisplay(QLabel):
    def __init__(self):
        super().__init__()
        self.setPixmap(QPixmap(800, 600))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background-color: black; border: 1px solid #444;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setScaledContents(True)
