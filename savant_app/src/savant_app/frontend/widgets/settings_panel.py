from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

class SettingsPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(QLabel("Settings Panel"))
        # Add configuration controls here
