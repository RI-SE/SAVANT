# frontend/widgets/playback_controls.py
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton
from PyQt6.QtCore import QSize, pyqtSignal
from PyQt6.QtGui import QIcon
from frontend.utils.assets import icon


class PlaybackControls(QWidget):

    prev_frame_clicked = pyqtSignal()
    next_frame_clicked = pyqtSignal()
    play_clicked = pyqtSignal()
    skip_backward_clicked = pyqtSignal(int)
    skip_forward_clicked = pyqtSignal(int)

    def __init__(self):
        super().__init__()

        def make_btn(filename: str, tooltip: str) -> QPushButton:
            btn = QPushButton()
            btn.setIcon(icon(filename))
            btn.setIconSize(QSize(36, 36))
            btn.setToolTip(tooltip)
            btn.setFlat(True)
            return btn

        self.btn_prev_frame = make_btn("skip_backward.svg", "Skip -30")
        self.btn_skip_back = make_btn("seek_backward.svg", "Previous Frame")
        self.btn_play = make_btn("play.svg", "Play")
        self.btn_skip_forward = make_btn("seek_forward.svg", "Next Frame")
        self.btn_next_frame = make_btn("skip_forward.svg", "Skip +30")

        self.btn_skip_back.clicked.connect(self.prev_frame_clicked.emit)
        self.btn_prev_frame.clicked.connect(
            lambda: self.skip_backward_clicked.emit(30))
        self.btn_skip_forward.clicked.connect(self.next_frame_clicked.emit)
        self.btn_next_frame.clicked.connect(
            lambda: self.skip_forward_clicked.emit(30))

        self.btn_play.clicked.connect(self.play_clicked.emit)

        layout = QHBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.addStretch(1)
        layout.addWidget(self.btn_prev_frame)
        layout.addWidget(self.btn_skip_back)
        layout.addWidget(self.btn_play)
        layout.addWidget(self.btn_skip_forward)
        layout.addWidget(self.btn_next_frame)
        layout.addStretch(1)

    def set_icon_paths(
            self, *, prev: str, skip_back: str, play: str,
            skip_forward: str, next_: str) -> None:
        self.btn_prev_frame.setIcon(QIcon(prev))
        self.btn_skip_back.setIcon(QIcon(skip_back))
        self.btn_play.setIcon(QIcon(play))
        self.btn_skip_forward.setIcon(QIcon(skip_forward))
        self.btn_next_frame.setIcon(QIcon(next_))

    def set_playing(self, playing: bool) -> None:
        """
        Change the play button to reflect play/pause state.
        Called from MainWindow when playback starts/stops.
        """
        if playing:
            self.btn_play.setToolTip("Pause")
            self.btn_play.setIcon(icon("pause.svg"))
        else:
            self.btn_play.setToolTip("Play")
            self.btn_play.setIcon(icon("play.svg"))
