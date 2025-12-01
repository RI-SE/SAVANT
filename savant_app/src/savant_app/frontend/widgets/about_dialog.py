from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from savant_app.frontend.utils.assets import asset_path


class AboutDialog(QDialog):
    def __init__(self, theme: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About SAVANT")
        self.setModal(True)
        self.resize(720, 300)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(25, 25, 25, 25)
        self.setLayout(main_layout)

        # Content layout
        content_layout = QHBoxLayout()
        content_layout.setSpacing(40)

        # Logo
        logo_label = QLabel()
        pixmap = QPixmap(
            asset_path("Logo_darkmode.png" if theme == "Dark" else "Logo_lightmode.png")
        )
        logo_label.setPixmap(
            pixmap.scaledToHeight(250, Qt.TransformationMode.SmoothTransformation)
        )
        logo_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        content_layout.addWidget(logo_label)

        text_container = QWidget()
        text_layout = QVBoxLayout(text_container)
        text_layout.setSpacing(8)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Ensure ALL labels align to the same left edge
        def left_align(label: QLabel):
            label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            return label

        # Title
        title = left_align(QLabel("SAVANT"))
        ft = QFont()
        ft.setPointSize(26)
        ft.setBold(True)
        title.setFont(ft)

        # Subtitle
        subtitle = left_align(QLabel("Video Annotation Tool"))
        fs = QFont()
        fs.setPointSize(15)
        fs.setBold(True)
        subtitle.setFont(fs)

        # Version
        version = left_align(QLabel("Version 1.3.2"))
        version.setStyleSheet("color: #8c8c8c; font-size: 13px;")

        # Description
        desc = left_align(
            QLabel("A tool for creating and managing vehicle annotations.")
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 13px; line-height: 150%; padding-top: 10px")

        # Add all labels
        text_layout.addWidget(title)
        text_layout.addWidget(subtitle)
        text_layout.addWidget(version)
        text_layout.addWidget(desc)

        # Add fixed container to the content layout
        content_layout.addWidget(text_container)
        main_layout.addLayout(content_layout)

        # Spacer
        main_layout.addStretch()

        # OK button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        ok_button = QPushButton("OK")
        ok_button.setFixedWidth(100)
        ok_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
        )
        ok_button.clicked.connect(self.accept)
        button_layout.addWidget(ok_button)
        main_layout.addLayout(button_layout)
