from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from savant_app.frontend.types import BBoxDimensionData
from savant_app.frontend.utils.assets import icon
from savant_app.frontend.theme.constants import (
    PLAYBACK_BUTTON_ICON_SIZE,
    PLAYBACK_ISSUE_ARROW_COLOR,
)
from savant_app.frontend.theme.playback_styles import (
    apply_issue_navigation_button_style,
)


class PlaybackControls(QWidget):

    prev_frame_clicked = pyqtSignal()
    next_frame_clicked = pyqtSignal()
    play_clicked = pyqtSignal()
    skip_backward_clicked = pyqtSignal(int)
    skip_forward_clicked = pyqtSignal(int)
    prev_issue_clicked = pyqtSignal()
    next_issue_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()

        icon_size = QSize(PLAYBACK_BUTTON_ICON_SIZE, PLAYBACK_BUTTON_ICON_SIZE)

        def make_btn(
            filename: str, tooltip: str, *, tint=None
        ) -> QPushButton:
            btn = QPushButton()
            btn.setIcon(self._build_icon(filename, tint=tint))
            btn.setIconSize(icon_size)
            btn.setToolTip(tooltip)
            btn.setFlat(True)
            return btn

        self.btn_prev_frame = make_btn("skip_backward.svg", "Skip -30")
        self.btn_skip_back = make_btn("seek_backward.svg", "Previous Frame")
        self.btn_prev_issue = make_btn(
            "seek_backward.svg",
            "Previous warning/error frame",
            tint=PLAYBACK_ISSUE_ARROW_COLOR,
        )
        self.btn_play = make_btn("play.svg", "Play")
        self.btn_next_issue = make_btn(
            "seek_forward.svg",
            "Next warning/error frame",
            tint=PLAYBACK_ISSUE_ARROW_COLOR,
        )
        self.btn_skip_forward = make_btn("seek_forward.svg", "Next Frame")
        self.btn_next_frame = make_btn("skip_forward.svg", "Skip +30")

        self.btn_skip_back.clicked.connect(self.prev_frame_clicked.emit)
        self.btn_prev_frame.clicked.connect(lambda: self.skip_backward_clicked.emit(30))
        self.btn_prev_issue.clicked.connect(self.prev_issue_clicked.emit)
        self.btn_skip_forward.clicked.connect(self.next_frame_clicked.emit)
        self.btn_next_frame.clicked.connect(lambda: self.skip_forward_clicked.emit(30))
        self.btn_next_issue.clicked.connect(self.next_issue_clicked.emit)

        self.btn_play.clicked.connect(self.play_clicked.emit)

        apply_issue_navigation_button_style(self.btn_prev_issue)
        apply_issue_navigation_button_style(self.btn_next_issue)

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(0, 10, 0, 10)

        self.issue_info_widget = QWidget()
        issue_layout = QVBoxLayout(self.issue_info_widget)
        issue_layout.setContentsMargins(0, 0, 0, 0)
        issue_layout.setSpacing(4)
        self.issue_heading = QLabel("Frame Issues")
        self.issue_heading.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.issue_heading.setStyleSheet("font-weight: bold;")
        self.issue_heading.setSizePolicy(
            self.issue_heading.sizePolicy().horizontalPolicy(),
            self.issue_heading.sizePolicy().verticalPolicy(),
        )
        self.issue_details_label = QLabel()
        self.issue_details_label.setWordWrap(True)
        self.issue_details_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self.issue_details_label.setObjectName("IssueInfoLabel")
        self.issue_scroll = QScrollArea()
        self.issue_scroll.setWidgetResizable(True)
        self.issue_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.issue_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.issue_scroll.setWidget(self.issue_details_label)
        issue_layout.addWidget(self.issue_heading)
        issue_layout.addWidget(self.issue_scroll)

        # Annotation info widget (existing)
        self.info_widget = QWidget()
        self.info_widget.setObjectName("AnnotationInfoBar")
        info_layout = QVBoxLayout(self.info_widget)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(4)

        self.center_label = QLabel()
        self.center_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.center_label.setTextFormat(Qt.TextFormat.RichText)
        self.size_label = QLabel()
        self.size_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.size_label.setTextFormat(Qt.TextFormat.RichText)
        self.rotation_label = QLabel()
        self.rotation_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.rotation_label.setTextFormat(Qt.TextFormat.RichText)

        info_layout.addWidget(self.center_label)
        info_layout.addWidget(self.size_label)
        info_layout.addWidget(self.rotation_label)

        self.default_info_label = QLabel("No annotation selected.")
        self.default_info_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.default_info_label.setObjectName("DefaultInfoLabel")
        self.default_info_label.setStyleSheet("color: #888888; font-style: italic;")

        annotation_container = QWidget()
        annotation_layout = QVBoxLayout(annotation_container)
        annotation_layout.setContentsMargins(0, 0, 0, 0)
        annotation_layout.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop
        )
        self.annotation_heading = QLabel("Annotation Info")
        self.annotation_heading.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.annotation_heading.setStyleSheet("font-weight: bold;")
        annotation_layout.addWidget(self.annotation_heading)
        annotation_layout.addWidget(self.info_widget)
        annotation_layout.addWidget(self.default_info_label)

        # Control buttons are in a nested Horizontal layout
        controls_container = QWidget()
        controls_layout = QHBoxLayout(controls_container)
        controls_layout.setSpacing(10)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.btn_prev_issue)
        controls_layout.addWidget(self.btn_prev_frame)
        controls_layout.addWidget(self.btn_skip_back)
        controls_layout.addWidget(self.btn_play)
        controls_layout.addWidget(self.btn_skip_forward)
        controls_layout.addWidget(self.btn_next_frame)
        controls_layout.addWidget(self.btn_next_issue)
        controls_layout.addStretch(1)

        # Add the widgets to the main horizontal layout
        content_row = QHBoxLayout()
        content_row.setSpacing(20)
        content_row.addWidget(self.issue_info_widget, stretch=1)
        content_row.addStretch(1)
        content_row.addWidget(controls_container, stretch=0)
        content_row.addStretch(1)
        content_row.addWidget(annotation_container, stretch=1)
        content_row.addStretch(0)

        main_layout.addLayout(content_row)

        # Set initial visibility state
        self._issue_nav_visible = None
        self.clear_annotation_info()
        self.clear_issue_details()
        self.set_issue_navigation_visible(False)

    def set_icon_paths(
        self,
        *,
        prev: str,
        skip_back: str,
        play: str,
        skip_forward: str,
        next_: str,
        prev_issue: str | None = None,
        next_issue: str | None = None,
    ) -> None:
        self.btn_prev_frame.setIcon(QIcon(prev))
        self.btn_skip_back.setIcon(QIcon(skip_back))
        self.btn_play.setIcon(QIcon(play))
        self.btn_skip_forward.setIcon(QIcon(skip_forward))
        self.btn_next_frame.setIcon(QIcon(next_))
        if prev_issue:
            self.btn_prev_issue.setIcon(QIcon(prev_issue))
        if next_issue:
            self.btn_next_issue.setIcon(QIcon(next_issue))

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
            center_str = (
                f"<b>Center (x, y):</b> "
                f"{annotation_details.x_center:.2f}, {annotation_details.y_center:.2f}"
            )
            size_str = (
                f"<b>Size (w, h):</b> "
                f"{annotation_details.width:.2f}, {annotation_details.height:.2f}"
            )
            rot_str = f"<b>Rotation:</b> {annotation_details.rotation:.1f} radians"

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

    def display_issue_details(self, entries: list[dict]):
        """Render issue/tag details for the current frame."""
        if not entries:
            self.clear_issue_details()
            return
        self.issue_details_label.setStyleSheet("")
        parts = []
        for entry in entries:
            parts.append(
                "<b>Type:</b> {type}<br>"
                "<b>Object:</b> {object}<br>"
                "<b>Info:</b> {info}".format(
                    type=entry.get("type", "Unknown"),
                    object=entry.get("object", "Unknown"),
                    info=entry.get("info", ""),
                )
            )
        self.issue_details_label.setText("<br><br>".join(parts))

    def clear_issue_details(self):
        """Show placeholder when no tags/warnings are active."""
        self.issue_details_label.setStyleSheet("color: #888888; font-style: italic;")
        self.issue_details_label.setText("No tags, warnings, or errors on this frame.")

    def set_issue_navigation_visible(self, visible: bool) -> None:
        visible = bool(visible)
        if getattr(self, "_issue_nav_visible", None) == visible:
            return
        self._issue_nav_visible = visible
        self.btn_prev_issue.setVisible(visible)
        self.btn_next_issue.setVisible(visible)

    def _build_icon(self, filename: str, tint=None) -> QIcon:
        base_icon = icon(filename)
        if tint is None:
            return base_icon
        pixmap = base_icon.pixmap(
            PLAYBACK_BUTTON_ICON_SIZE, PLAYBACK_BUTTON_ICON_SIZE
        )
        if pixmap.isNull():
            return base_icon
        tinted = QPixmap(pixmap.size())
        tinted.fill(Qt.GlobalColor.transparent)
        painter = QPainter(tinted)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        painter.drawPixmap(0, 0, pixmap)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(tinted.rect(), tint)
        painter.end()
        return QIcon(tinted)
