# main_window.py
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QMessageBox
from savant_app.frontend.widgets.video_display import VideoDisplay
from savant_app.frontend.widgets.playback_controls import PlaybackControls
from savant_app.frontend.widgets.sidebar import Sidebar
from savant_app.frontend.widgets.seek_bar import SeekBar
import os
from savant_app.controllers.project_state_controller import ProjectStateController
from savant_app.controllers.video_controller import VideoController
from savant_app.controllers.annotation_controller import AnnotationController


class MainWindow(QMainWindow):
    def __init__(
        self,
        project_name,
        video_controller: VideoController,
        project_state_controller: ProjectStateController,
        annotation_controller: AnnotationController
    ):
        super().__init__()
        self.project_name = project_name
        self.update_title()
        self.resize(1600, 800)

        # Controller
        self.video_controller = video_controller
        self.annotation_controller = annotation_controller
        self.project_state_controller = project_state_controller

        # Video + playback
        self.video_widget = VideoDisplay()
        self.seek_bar = SeekBar()
        self.playback_controls = PlaybackControls()

        video_layout = QVBoxLayout()
        video_layout.addWidget(self.video_widget, stretch=1)
        video_layout.addWidget(self.seek_bar)
        video_layout.addWidget(self.playback_controls)

        # Sidebar
        actors = self.project_state_controller.get_actor_types()
        self.sidebar = Sidebar(video_actors=actors)

        # Main layout
        main_layout = QHBoxLayout()
        main_layout.addLayout(video_layout, stretch=1)
        main_layout.addWidget(self.sidebar)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Playback timer/state
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._step_playback)
        self._is_playing = False

        # Sidebar signals
        self.sidebar.open_video.connect(self.on_open_video)
        self.sidebar.open_config.connect(self.open_openlabel_config)
        self.sidebar.start_bbox_drawing.connect(self.video_widget.start_drawing_mode)

        # Playback controls signals
        self.playback_controls.next_frame_clicked.connect(self.on_next)
        self.playback_controls.prev_frame_clicked.connect(self.on_prev)
        self.playback_controls.play_clicked.connect(self.on_play)
        
        # Connect annotation signals
        self.sidebar.add_new_object.connect(self.annotation_controller.add_new_object)
        self.video_widget.bbox_drawn.connect(self.handle_drawn_bbox)

        if hasattr(self.playback_controls, "skip_backward_clicked"):
            self.playback_controls.skip_backward_clicked.connect(self.on_skip_back)
        if hasattr(self.playback_controls, "skip_forward_clicked"):
            self.playback_controls.skip_forward_clicked.connect(self.on_skip_forward)

        # Seek bar → jump directly
        self.seek_bar.frame_changed.connect(self.on_seek)

    def update_title(self):
        self.setWindowTitle(f"SAVANT {self.project_name}")

    # seek (slider)
    def on_seek(self, index: int):
        """
        Handle seek bar interaction (user clicked or dragged the slider).

        Args:
        index (int): The target frame index selected on the seek bar.
        """
        pixmap, idx = self.controller.jump_to_frame(index)
        if pixmap:
            self.video_widget.show_frame(pixmap)
            self.seek_bar.set_position(idx)

    # File open
    def on_open_video(self, path: str):
        """
        Handle loading of a new video file.

        Args:
            path (str): Path to the video file selected by the user.

        Error Handling:
            - If the video cannot be loaded, shows a critical error message.
        """
        try:
            self.video_controller.load_video(path)
            pixmap, idx = self.video_controller.jump_to_frame(0)
            if pixmap:
                self.video_widget.show_frame(pixmap)
            total = self.video_controller.total_frames()
            self.seek_bar.update_range(total)
            self.seek_bar.set_position(idx if pixmap else 0)
            if hasattr(self.playback_controls, "set_fps"):
                self.playback_controls.set_fps(self.video_controller.fps())
            self._stop_playback()
            filename = os.path.basename(path)
            self.setWindowTitle(f"SAVANT {self.project_name} — {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Failed to open video", str(e))

    def open_openlabel_config(self, path: str):
        try:
            self.project_state_controller.load_openlabel_config(path)
            QMessageBox.information(
                self, "Config Loaded", "OpenLabel configuration loaded successfully."
            )
        except Exception as e:
            QMessageBox.critical(self, "Failed to load config", str(e))

    # Navigation
    def on_next(self):
        """Navigates to the next frame if possible."""
        try:
            pixmap, idx = self.video_controller.next_frame()
            if pixmap:
                self.video_widget.show_frame(pixmap)
                self.seek_bar.set_position(idx)
        except StopIteration:
            self._stop_playback()
            QMessageBox.information(self, "End of video", "No more frames.")
        except Exception as e:
            self._stop_playback()
            QMessageBox.critical(self, "Error", str(e))

    def on_prev(self):
        """Navigates to the previous frame if possible."""
        try:
            pm = self.video_controller.previous_frame()
            self.video_widget.show_frame(pm)
            pixmap, idx = self.video_controller.previous_frame()
            if pixmap:
                self.video_widget.show_frame(pixmap)
                self.seek_bar.set_position(idx)
        except IndexError:
            QMessageBox.information(self, "At start", "Already at the first frame.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # Play/Pause
    def on_play(self):
        """Toggle play/pause and handle edge cases (no video, end-of-video, fps=0)."""
        try:
            total = self.video_controller.total_frames()
        except Exception:
            QMessageBox.information(self, "No video", "Load a video first.")
            return

        if not self._is_playing:
            try:
                if self.video_controller.current_index() < 0:
                    pixmap, idx = self.controller.jump_to_frame(0)
                    if pixmap:
                        self.video_widget.show_frame(pixmap)
                        self.seek_bar.set_position(idx)
                if self.video_controller.current_index() >= total - 1:
                    pixmap, idx = self.video_controller.jump_to_frame(0)
                    if pixmap:
                        self.video_widget.show_frame(pixmap)
                        self.seek_bar.set_position(idx)
                fps = self.video_controller.fps()
                interval_ms = max(1, int(1000 / (fps if fps and fps > 0 else 30)))
                self._play_timer.start(interval_ms)
                self._is_playing = True
                if hasattr(self.playback_controls, "set_playing"):
                    self.playback_controls.set_playing(True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Cannot start playback: {e}")
        else:
            self._stop_playback()

    def _step_playback(self):
        """Advance one frame per tick while playing."""
        try:
            pixmap, idx = self.video_controller.next_frame()
            if pixmap:
                self.video_widget.show_frame(pixmap)
                self.seek_bar.set_position(idx)
            else:
                self._stop_playback()
        except StopIteration:
            self._stop_playback()
        except Exception as e:
            self._stop_playback()
            QMessageBox.critical(self, "Playback error", str(e))

    def _stop_playback(self):
        """Stop video playback."""
        if self._play_timer.isActive():
            self._play_timer.stop()
        if self._is_playing and hasattr(self.playback_controls, "set_playing"):
            self.playback_controls.set_playing(False)
        self._is_playing = False

    # Skip handlers
    def on_skip_back(self, n: int):
        """
        Skip backward by `n` frames.

        Args:
            n (int): Number of frames to move back.
        """
        try:
            pixmap, idx = self.video_controller.skip_frames(-n)
            if pixmap:
                self.video_widget.show_frame(pixmap)
                self.seek_bar.set_position(idx)
        except Exception as e:
            QMessageBox.critical(self, "Skip backward failed", str(e))

    def on_skip_forward(self, n: int):
        """
        Skip forward by `n` frames.

        Args:
            n (int): Number of frames to move forward.
        """
        try:
            pixmap, idx = self.video_controller.skip_frames(n)
            if pixmap:
                self.video_widget.show_frame(pixmap)
                self.seek_bar.set_position(idx)
        except Exception as e:
            QMessageBox.critical(self, "Skip forward failed", str(e))

    def handle_drawn_bbox(self, bbox_coords):
        """Handle newly drawn bounding box coordinates from video widget."""
        frame_idx = self.video_controller.current_index()
        self.annotation_controller.add_annotation(
            frame_number=frame_idx,
            coordinates=bbox_coords
        )
        # Update UI elements as needed
        self.sidebar.refresh_annotations_list()
