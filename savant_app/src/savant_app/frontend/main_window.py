from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout
from savant_app.src.savant_app.frontend.widgets.video_display import VideoDisplay
from savant_app.src.savant_app.frontend.widgets.playback_controls import PlaybackControls
from savant_app.src.savant_app.frontend.widgets.sidebar import Sidebar
from PyQt6.QtGui import QIcon


class MainWindow(QMainWindow):
    def __init__(self, project_name):
        super().__init__()
        self.project_name = project_name
        self.update_title()
        self.resize(1600, 800)

        # Video + playback
        video_layout = QVBoxLayout()
        video_widget = VideoDisplay()
        video_layout.addWidget(video_widget, stretch=1)
        video_layout.addLayout(PlaybackControls())

        # Sidebar
        sidebar = Sidebar()

        # Main layout
        main_layout = QHBoxLayout()
        main_layout.addLayout(video_layout, stretch=1)
        main_layout.addWidget(sidebar)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def update_title(self):
        self.setWindowTitle(f"SAVANT {self.project_name}")