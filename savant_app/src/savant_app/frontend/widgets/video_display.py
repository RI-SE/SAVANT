from PyQt6.QtWidgets import QLabel, QSizePolicy
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt


class VideoDisplay(QLabel):
    def __init__(self):
        super().__init__()
        self.setPixmap(QPixmap())
        self.setMinimumSize(320, 240)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background-color: black; border: 1px solid #444;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setScaledContents(True)

    def show_frame(self, pixmap: QPixmap) -> None:
        if pixmap and not pixmap.isNull():
            self.setPixmap(pixmap)
