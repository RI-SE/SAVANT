# main_window.py
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from savant_app.frontend.exceptions import InvalidWarningErrorRange
from savant_app.frontend.states.frontend_state import FrontendState
from savant_app.frontend.states.sidebar_state import SidebarState
from savant_app.frontend.widgets.annotator_dialog import AnnotatorDialog
from savant_app.frontend.utils.settings_store import (
    get_action_interval_offset,
    # get_ontology_path,  # legacy manual ontology picker
    get_warning_range,
    get_error_range,
    get_show_warnings,
    get_show_errors,
    get_tag_options,
    set_threshold_ranges,
    set_show_warnings,
    set_show_errors,
    set_tag_option_states,
)
from savant_app.frontend.utils import (
    annotation_ops,
    confidence_ops,
    navigation,
    playback,
    project_io,
    render,
    zoom,
)

from savant_app.frontend.utils.undo import (
    ControllerAnnotationGateway,
    ControllerFrameTagGateway,
    GatewayHolder,
    UndoRedoManager,
)
from savant_app.frontend.widgets.about_dialog import AboutDialog
from savant_app.frontend.widgets.menu import AppMenu
from savant_app.frontend.widgets.overlay import Overlay
from savant_app.frontend.widgets.playback_controls import PlaybackControls
from savant_app.frontend.widgets.seek_bar import SeekBar
from savant_app.frontend.widgets.settings import SettingsDialog
from savant_app.frontend.widgets.sidebar import Sidebar
from savant_app.frontend.widgets.video_display import VideoDisplay


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

        self.undo_manager = UndoRedoManager()
        self.undo_context = GatewayHolder(
            annotation_gateway=ControllerAnnotationGateway(
                annotation_controller=self.annotation_controller,
                project_state_controller=self.project_state_controller,
            ),
            frame_tag_gateway=ControllerFrameTagGateway(self.annotation_controller),
        )

        # state
        self.state = FrontendState(self)
        self.state.register_annotator_prompt(self.prompt_for_annotator)
        self.state.confidenceIssuesChanged.connect(
            lambda _: confidence_ops.apply_confidence_markers(self)
        )

        # Menu
        self.menu = AppMenu(
            self,
            on_new=self.start_new_project_flow,
            on_load=self.load_project_flow,
            on_save=self.quick_save_project,
            on_settings=self.open_settings,
            on_new_bbox=self.create_new_bounding_box,
            on_new_frame_tag=self.create_new_frame_tag,
            on_interpolate=self.open_interpolation_dialog,
            on_create_relationship=self.open_relationship_dialog,
            on_about=self.open_about,
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
        actors: dict[str, list[str]] = {}
        self.sidebar = Sidebar(
            actors,
            self.annotation_controller,
            self.video_controller,
            self.project_state_controller,
            self.state,
            self.sidebar_state,
        )
        self.seek_bar.frame_changed.connect(self.sidebar.on_frame_changed)

        main_layout = QHBoxLayout()
        main_layout.addLayout(video_layout, stretch=1)
        main_layout.addWidget(self.sidebar)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        project_io.wire(self)
        playback.wire(self)
        navigation.wire(self)
        render.wire(self)

        annotation_ops.wire(self, self.state)
        zoom.wire(self, initial=1.0)

        QShortcut(
            QKeySequence.StandardKey.Undo,
            self,
            activated=lambda: annotation_ops.undo_last_action(self),
        )
        QShortcut(
            QKeySequence.StandardKey.Redo,
            self,
            activated=lambda: annotation_ops.redo_last_action(self),
        )
        QShortcut(
            QKeySequence(Qt.KeyboardModifier.ControlModifier | Qt.Key.Key_Y),
            self,
            activated=lambda: annotation_ops.redo_last_action(self),
        )

        self.refresh_confidence_issues()
        self.update_issue_info()

        initial_annotator = self.prompt_for_annotator()
        if initial_annotator:
            self.state.set_current_annotator(initial_annotator)

    def set_project_name(self, name: str) -> None:
        cleaned = (name or "").strip()
        if not cleaned:
            return
        self.project_name = cleaned
        self.update_title()

    def update_title(self):
        self.setWindowTitle(f"SAVANT {self.project_name}")

    def prompt_for_annotator(self) -> str | None:
        dialog = AnnotatorDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.get_annotator_name()
        return None

    def open_about(self):
        # Determine current theme based on the window's palette
        palette = self.palette()
        background_color = palette.color(self.backgroundRole())
        # if the average of RGB is less than half of max (255), it's dark
        if (
            background_color.red() + background_color.green() + background_color.blue()
        ) / 3 < 128:
            current_theme = "Dark"
        else:
            current_theme = "Light"

        about_dialog = AboutDialog(theme=current_theme, parent=self)
        about_dialog.exec()

    def open_settings(self):
        dlg = SettingsDialog(
            theme="Dark",
            zoom_rate=1.2,
            frame_count=self.sidebar_state.historic_obj_frame_count,
            # Manual ontology parameter:
            # ontology_path=get_ontology_path(),
            action_interval_offset=get_action_interval_offset(),
            tag_options=get_tag_options(),
            parent=self,
        )
        # Signal hook for manual ontology updates:
        # dlg.ontology_path_selected.connect(self.sidebar.reload_bbox_type_combo)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            vals = dlg.values()
            self.sidebar_state.historic_obj_frame_count = vals["previous_frame_count"]
            self.sidebar.refresh_confidence_issue_list()
            warning_vals = vals.get("warning_range", get_warning_range())
            error_vals = vals.get("error_range", get_error_range())
            warning_range = tuple(float(v) for v in warning_vals)
            error_range = tuple(float(v) for v in error_vals)
            show_warnings = vals.get("show_warnings", get_show_warnings())
            show_errors = vals.get("show_errors", get_show_errors())
            set_tag_option_states(vals.get("tag_options", {}))
            thresholds_valid = True
            if (
                show_warnings
                and show_errors
                and not (
                    warning_range[1] <= error_range[0]
                    or error_range[1] <= warning_range[0]
                )
            ):
                QMessageBox.critical(
                    self,
                    "Invalid Ranges",
                    "Warning and error ranges must not overlap when both markers are visible.",
                )
                thresholds_valid = False

            if thresholds_valid:
                try:
                    set_threshold_ranges(
                        warning_range=warning_range,
                        error_range=error_range,
                        show_warnings=show_warnings,
                        show_errors=show_errors,
                    )
                    set_show_warnings(show_warnings)
                    set_show_errors(show_errors)
                    confidence_ops.apply_confidence_markers(self)
                except InvalidWarningErrorRange as ex:
                    QMessageBox.critical(self, "Invalid Ranges", str(ex))
                    thresholds_valid = False
            if thresholds_valid:
                confidence_ops.refresh_confidence_issues(self)
            else:
                confidence_ops.apply_confidence_markers(self)
            self.update_issue_info()

    def noop(*args, **kwargs):
        print("Not implemented yet")

    def start_new_project_flow(self):
        """Proxy to the sidebar's staged new-project UI."""

        sidebar_flow = getattr(self.sidebar, "start_new_project_flow", None)
        if callable(sidebar_flow):
            sidebar_flow()

    def load_project_flow(self):
        """Trigger the sidebar's open-project folder dialog."""
        loader = getattr(self.sidebar, "open_project_folder_dialog", None)
        if callable(loader):
            loader()

    def quick_save_project(self):
        """Trigger quick-save through the sidebar wiring."""
        saver = getattr(self.sidebar, "quick_save_project", None)
        if callable(saver):
            saver()

    def create_new_bounding_box(self):
        """Open the sidebar's bounding box workflow."""
        creator = getattr(self.sidebar, "create_new_bbox", None)
        if callable(creator):
            creator()

    def create_new_frame_tag(self):
        """Open the sidebar's frame tag dialog."""
        tagger = getattr(self.sidebar, "create_new_frame_tag", None)
        if callable(tagger):
            tagger()

    def open_interpolation_dialog(self):
        """Open the sidebar's interpolation dialog."""
        interpolator = getattr(self.sidebar, "open_interpolation_dialog", None)
        if callable(interpolator):
            interpolator()

    def open_relationship_dialog(self):
        """Open the sidebar's relationship dialog."""
        rel = getattr(self.sidebar, "open_relationship_dialog", None)
        if callable(rel):
            rel()

    def refresh_confidence_issues(self):
        confidence_ops.refresh_confidence_issues(self)

    def apply_confidence_markers(self):
        confidence_ops.apply_confidence_markers(self)

    def update_issue_info(self):
        confidence_ops.update_issue_info(self)

    def execute_undoable_command(self, command):
        self.undo_manager.execute(command, self.undo_context)

    def undo_last_command(self):
        return self.undo_manager.undo(self.undo_context)

    def redo_last_command(self):
        return self.undo_manager.redo(self.undo_context)
