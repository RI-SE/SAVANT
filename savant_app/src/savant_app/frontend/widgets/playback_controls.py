from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from savant_app.frontend.types import BBoxDimensionData
from savant_app.frontend.utils.assets import icon


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
        self.btn_prev_frame.clicked.connect(lambda: self.skip_backward_clicked.emit(30))
        self.btn_skip_forward.clicked.connect(self.next_frame_clicked.emit)
        self.btn_next_frame.clicked.connect(lambda: self.skip_forward_clicked.emit(30))

        self.btn_play.clicked.connect(self.play_clicked.emit)

        # Main layout is now Vertical
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(0, 10, 0, 10)

        # This widget will hold the detailed info labels
        self.info_widget = QWidget()
        self.info_widget.setObjectName("AnnotationInfoBar")
        info_layout = QHBoxLayout(self.info_widget)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(20)  # Add some spacing between items

        # Create clear, distinct labels for each piece of data
        self.center_label = QLabel()
        self.center_label.setObjectName("InfoLabel")
        self.center_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.size_label = QLabel()
        self.size_label.setObjectName("InfoLabel")
        self.size_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.rotation_label = QLabel()
        self.rotation_label.setObjectName("InfoLabel")
        self.rotation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Add to the info layout with stretches to center them
        info_layout.addStretch(1)
        info_layout.addWidget(self.center_label)
        info_layout.addWidget(self.size_label)
        info_layout.addWidget(self.rotation_label)
        info_layout.addStretch(1)

        # --- This label is for the default "No selection" text ---
        self.default_info_label = QLabel("No annotation selected.")
        self.default_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.default_info_label.setObjectName("DefaultInfoLabel")
        # Make the default text a bit more subtle
        self.default_info_label.setStyleSheet("color: #888888; font-style: italic;")

        # Control buttons are in a nested Horizontal layout
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        controls_layout.addStretch(1)
        controls_layout.addWidget(self.btn_prev_frame)
        controls_layout.addWidget(self.btn_skip_back)
        controls_layout.addWidget(self.btn_play)
        controls_layout.addWidget(self.btn_skip_forward)
        controls_layout.addWidget(self.btn_next_frame)
        controls_layout.addStretch(1)

        # Add the widgets to the main vertical layout in the new order
        main_layout.addWidget(self.info_widget)
        main_layout.addWidget(self.default_info_label)
        main_layout.addLayout(controls_layout)

        # Set initial visibility state
        self.clear_annotation_info()

    def set_icon_paths(
        self, *, prev: str, skip_back: str, play: str, skip_forward: str, next_: str
    ) -> None:
        self.btn_prev_frame.setIcon(QIcon(prev))
        self.btn_skip_back.setIcon(QIcon(skip_back))
        self.btn_play.setIcon(QIcon(play))
        self.btn_skip_forward.setIcon(QIcon(skip_forward))
        self.btn_next_frame.setIcon(QIcon(next_))

    def set_playing(self, playing: bool) -> None:
        """Change the play button to reflect play/pause state.
        Called from MainWindow when playback starts/stops.
        """
        if playing:
            self.btn_play.setToolTip("Pause")
            self.btn_play.setIcon(icon("pause.svg"))
        else:
            self.btn_play.setToolTip("Play")
            self.btn_play.setIcon(icon("play.svg"))

    def display_annotation_info(self, annotation_details: BBoxDimensionData):
        """
        Displays the dimensions of the selected bounding box.
        Called from MainWindow when an annotation is selected.
        """
        if annotation_details is None:
            self.clear_annotation_info()
        else:
            # Format the string with 2 decimal places for positions/dimensions
            # and 1 for rotation, using rich text for clarity.
            center_str = f"""
                Center (x, y): <b>{annotation_details.x_center:.2f},
                {annotation_details.y_center:.2f}</b>
            """
            size_str = f"""
                Size (w, h): <b>{annotation_details.width:.2f},
                {annotation_details.height:.2f}</b>
            """
            rot_str = f"Rotation: <b>{annotation_details.rotation:.1f}°</b>"

            self.center_label.setText(center_str)
            self.size_label.setText(size_str)
            self.rotation_label.setText(rot_str)

            # Show the detailed info, hide the default text
            self.info_widget.show()
            self.default_info_label.hide()

    def clear_annotation_info(self):
        """Resets the annotation info label to its default text."""
        # Hide the detailed info, show the default text
        self.info_widget.hide()
        self.default_info_label.show()

        # Optional: Clear the text to be tidy
        self.center_label.setText("")
        self.size_label.setText("")
        self.rotation_label.setText("")
