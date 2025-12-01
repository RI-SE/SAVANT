from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStyle,
    QVBoxLayout,
)

from savant_app.frontend.utils.assets import asset_path


class AboutDialog(QDialog):
    def __init__(self, theme: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About SAVANT")
        self.setModal(True)
        self.resize(700, 280)  # Removed fixed resize

        # --- Main Layout ---
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        self.setLayout(main_layout)

        # --- Content Container (Logo + Text) ---
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)  # Space between logo and text

        # 1. Logo (Left Side)
        logo_label = QLabel()
        if theme == "Dark":
            logo_pixmap = QPixmap(asset_path("Logo_darkmode.png"))
        else:
            logo_pixmap = QPixmap(asset_path("Logo_lightmode.png"))
        logo_label.setPixmap(
            logo_pixmap.scaledToHeight(250, Qt.TransformationMode.SmoothTransformation)
        )
        logo_label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter
        )

        content_layout.addWidget(logo_label)

        # 2. Text Info (Right Side)
        text_layout = QVBoxLayout()
        text_layout.setSpacing(5)

        title_label = QLabel("SAVANT")
        font_title = QFont()
        font_title.setBold(True)
        font_title.setPointSize(20)
        title_label.setFont(font_title)

        subtitle_label = QLabel("Video Annotation Tool")
        font_sub = QFont()
        font_sub.setBold(True)
        font_sub.setPointSize(11)
        subtitle_label.setFont(font_sub)

        version_label = QLabel("Version 1.3.2")
        version_label.setStyleSheet("color: #666666;")  # Greyish text

        description_label = QLabel(
            "A tool for creating and managing vehicle annotations."
        )
        description_label.setWordWrap(True)
        description_label.setStyleSheet("margin-top: 10px;")

        text_layout.addWidget(title_label)
        text_layout.addWidget(subtitle_label)
        text_layout.addWidget(version_label)
        text_layout.addWidget(description_label)
        text_layout.addStretch()  # Push text to top

        content_layout.addLayout(text_layout)

        main_layout.addLayout(content_layout)

        # --- Spacer between content and button ---
        main_layout.addStretch()

        button_layout = QHBoxLayout()
        button_layout.addStretch()  # Pushes button to the right

        self.ok_button = QPushButton("OK")
        self.ok_button.setFixedWidth(100)

        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
        self.ok_button.setIcon(icon)

        self.ok_button.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_button)

        main_layout.addLayout(button_layout)
