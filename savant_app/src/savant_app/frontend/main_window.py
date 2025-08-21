# main_window.py
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QMessageBox
)
from frontend.widgets.video_display import VideoDisplay
from frontend.widgets.playback_controls import PlaybackControls
from frontend.widgets.sidebar import Sidebar
from controllers.video_controller import VideoController
import os


class MainWindow(QMainWindow):
    def __init__(self, project_name):
        super().__init__()
        self.project_name = project_name
        self.update_title()
        self.resize(1600, 800)

        # Controller
        self.controller = VideoController()

        # Video + playback
        self.video_widget = VideoDisplay()
        self.playback_controls = PlaybackControls()

        video_layout = QVBoxLayout()
        video_layout.addWidget(self.video_widget, stretch=1)
        video_layout.addWidget(self.playback_controls)

        # Sidebar
        self.sidebar = Sidebar()

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

        # Sidebar signal
        self.sidebar.open_video.connect(self.on_open_video)

        # Playback controls signals
        self.playback_controls.next_frame_clicked.connect(self.on_next)
        self.playback_controls.prev_frame_clicked.connect(self.on_prev)
        self.playback_controls.play_clicked.connect(self.on_play)

        if hasattr(self.playback_controls, "skip_backward_clicked"):
            self.playback_controls.skip_backward_clicked.connect(
                self.on_skip_back)
        if hasattr(self.playback_controls, "skip_forward_clicked"):
            self.playback_controls.skip_forward_clicked.connect(
                self.on_skip_forward)

    def update_title(self):
        self.setWindowTitle(f"SAVANT {self.project_name}")

    # File open
    def on_open_video(self, path: str):
        try:
            self.controller.load_video(path)
            pm = self.controller.jump_to_frame(0)
            self.video_widget.show_frame(pm)
            self._stop_playback()
            filename = os.path.basename(path)
            self.setWindowTitle(f"SAVANT {self.project_name} — {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Failed to open video", str(e))

    # Navigation
    def on_next(self):
        try:
            pm = self.controller.next_frame()
            self.video_widget.show_frame(pm)
        except StopIteration:
            self._stop_playback()
            QMessageBox.information(self, "End of video", "No more frames.")
        except Exception as e:
            self._stop_playback()
            QMessageBox.critical(self, "Error", str(e))

    def on_prev(self):
        try:
            pm = self.controller.previous_frame()
            self.video_widget.show_frame(pm)
        except IndexError:
            QMessageBox.information(
                self, "At start", "Already at the first frame.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # Play/Pause
    def on_play(self):
        """Toggle play/pause and handle edge cases (no video, end-of-video, fps=0)."""
        try:
            _ = self.controller.total_frames()
        except Exception:
            QMessageBox.information(self, "No video", "Load a video first.")
            return

        if not self._is_playing:
            try:
                if self.controller.current_index() < 0:
                    pm = self.controller.jump_to_frame(0)
                    self.video_widget.show_frame(pm)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Cannot start playback: {e}")
                return

            try:
                if self.controller.current_index() >= self.controller.total_frames() - 1:
                    pm = self.controller.jump_to_frame(0)
                    self.video_widget.show_frame(pm)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Cannot start playback: {e}")
                return

            try:
                fps = self.controller.fps()
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
            pm = self.controller.next_frame()
            self.video_widget.show_frame(pm)
        except StopIteration:
            self._stop_playback()
        except Exception as e:
            self._stop_playback()
            QMessageBox.critical(self, "Playback error", str(e))

    def _stop_playback(self):
        if self._play_timer.isActive():
            self._play_timer.stop()
        if self._is_playing and hasattr(self.playback_controls, "set_playing"):
            self.playback_controls.set_playing(False)
        self._is_playing = False

    # Skip handlers
    def on_skip_back(self, n: int):
        try:
            pm = self.controller.skip_frames(-n)
            self.video_widget.show_frame(pm)
        except Exception as e:
            QMessageBox.critical(self, "Skip backward failed", str(e))

    def on_skip_forward(self, n: int):
        try:
            pm = self.controller.skip_frames(n)
            self.video_widget.show_frame(pm)
        except Exception as e:
            QMessageBox.critical(self, "Skip forward failed", str(e))
