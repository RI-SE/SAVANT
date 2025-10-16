# main_window.py
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QDialog,
)
from PyQt6.QtGui import QKeySequence, QShortcut

from savant_app.frontend.widgets.video_display import VideoDisplay
from savant_app.frontend.widgets.playback_controls import PlaybackControls
from savant_app.frontend.widgets.sidebar import Sidebar
from savant_app.frontend.widgets.seek_bar import SeekBar
from savant_app.frontend.widgets.overlay import Overlay
from savant_app.frontend.widgets.menu import AppMenu
from savant_app.frontend.widgets.settings import SettingsDialog
from savant_app.frontend.utils.settings_store import (
    get_ontology_path,
    get_action_interval_offset,
)
from savant_app.frontend.states.sidebar_state import SidebarState
from savant_app.frontend.states.frontend_state import FrontendState

from savant_app.frontend.utils import (
    project_io,
    playback,
    navigation,
    render,
    annotation_ops,
    zoom,
)


class MainWindow(QMainWindow):
    def __init__(
        self,
        project_name,
        video_controller,
        project_state_controller,
        annotation_controller,
    ):
        super().__init__()
        self.project_name = project_name
        self.update_title()
        self.resize(1600, 800)

        # Controllers
        self.video_controller = video_controller
        self.annotation_controller = annotation_controller
        self.project_state_controller = project_state_controller

        # state
        self.state = FrontendState(self)

        # Menu
        self.menu = AppMenu(
            self,
            on_new=self.noop,
            on_load=self.noop,
            on_save=self.noop,
            on_settings=self.open_settings,
        )

        # Video + overlay
        self.video_widget = VideoDisplay()
        self.overlay = Overlay(self.video_widget)
        self.overlay.set_interactive(True)
        self.video_widget.pan_changed.connect(self.overlay.set_pan)
        # Seek + controls
        self.seek_bar = SeekBar()
        self.playback_controls = PlaybackControls()

        # Layout
        video_container = QWidget()
        vc_layout = QVBoxLayout(video_container)
        vc_layout.setContentsMargins(0, 0, 0, 0)
        vc_layout.addWidget(self.video_widget)

        video_layout = QVBoxLayout()
        video_layout.addWidget(video_container, stretch=1)
        video_layout.addWidget(self.seek_bar)
        video_layout.addWidget(self.playback_controls)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(0)
        vc_layout.setSpacing(0)

        # Sidebar
        self.sidebar_state = SidebarState()
        actors = self.annotation_controller.allowed_bbox_types()
        self.sidebar = Sidebar(
            actors,
            self.annotation_controller,
            self.video_controller,
            self.project_state_controller,
            self.sidebar_state,
        )
        self.seek_bar.frame_changed.connect(self.sidebar.on_frame_changed)

        main_layout = QHBoxLayout()
        main_layout.addLayout(video_layout, stretch=1)
        main_layout.addWidget(self.sidebar)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self._undo_stack: list[dict] = []

        project_io.wire(self)
        playback.wire(self)
        navigation.wire(self)
        render.wire(self)
        annotation_ops.wire(self)
        zoom.wire(self, initial=1.15)

        QShortcut(
            QKeySequence.StandardKey.Undo,
            self,
            activated=lambda: annotation_ops.undo_delete(self),
        )

    def update_title(self):
        self.setWindowTitle(f"SAVANT {self.project_name}")

    def open_settings(self):
        dlg = SettingsDialog(
            theme="Dark",
            zoom_rate=1.2,
            ontology_path=get_ontology_path(),
            action_interval_offset=get_action_interval_offset(),
            parent=self,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            vals = dlg.values()
            self.sidebar_state.historic_obj_frame_count = vals["previous_frame_count"]

    def noop(*args, **kwargs):
        print("Not implemented yet")
