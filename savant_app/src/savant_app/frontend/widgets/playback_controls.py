from PyQt6.QtWidgets import QHBoxLayout, QPushButton
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import QSize


class PlaybackControls(QHBoxLayout):
    def __init__(self):
        super().__init__()
        self.setSpacing(10)
        self.setContentsMargins(0, 10, 0, 0)

        def make_btn(icon_name, tooltip):
            btn = QPushButton()
            btn.setIcon(QIcon.fromTheme(icon_name))
            btn.setIconSize(QSize(32, 32))
            btn.setToolTip(tooltip)
            btn.setFlat(True)
            return btn

        self.addStretch(
            1
        )  # TODO - Change icons for our own assets (System icons wont work across platforms)
        self.addWidget(make_btn("media-skip-backward", "Previous Frame"))
        self.addWidget(make_btn("media-seek-backward", "Rewind"))
        self.addWidget(make_btn("media-playback-start", "Play"))
        self.addWidget(make_btn("media-seek-forward", "Fast Forward"))
        self.addWidget(make_btn("media-skip-forward", "Next Frame"))
        self.addStretch(1)
