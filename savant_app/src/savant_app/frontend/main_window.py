# main_window.py
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QMessageBox, QDialog
from savant_app.frontend.widgets.video_display import VideoDisplay
from savant_app.frontend.widgets.playback_controls import PlaybackControls
from savant_app.frontend.widgets.sidebar import Sidebar
from savant_app.frontend.widgets.seek_bar import SeekBar
from savant_app.frontend.widgets.overlay import Overlay
import os
from savant_app.controllers.project_state_controller import ProjectStateController
from savant_app.controllers.video_controller import VideoController
from savant_app.controllers.annotation_controller import AnnotationController
from savant_app.frontend.states.annotation_state import AnnotationMode, AnnotationState
from dataclasses import asdict
from pathlib import Path
from savant_app.frontend.widgets.menu import AppMenu

from savant_app.frontend.widgets.settings import SettingsDialog


class MainWindow(QMainWindow):
    def __init__(
        self,
        project_name,
        video_controller: VideoController,
        project_state_controller: ProjectStateController,
        annotation_controller: AnnotationController,
    ):
        super().__init__()
        self.project_name = project_name
        self.update_title()
        self.resize(1600, 800)
        self.menu = AppMenu(
            self,
            on_new=self.noop,
            on_load=self.noop,
            on_save=self.noop,
            on_settings=self.open_settings,
        )

        # Controllers
        self.video_controller = video_controller
        self.annotation_controller = annotation_controller
        self.project_state_controller = project_state_controller

        # Video display + overlay layered
        self.video_widget = VideoDisplay()
        self.overlay = Overlay(self.video_widget)
        self.overlay.set_interactive(True)
        self.overlay.boxMoved.connect(self._on_overlay_box_moved)
        self.overlay.boxResized.connect(self._on_overlay_box_resized)
        self.overlay.boxRotated.connect(self._on_overlay_box_rotated)
        self._overlay_ids: list[str] = []

        video_container = QWidget()
        video_container_layout = QVBoxLayout(video_container)
        video_container_layout.setContentsMargins(0, 0, 0, 0)
        video_container_layout.addWidget(self.video_widget)

        self.overlay.raise_()
        self.overlay.resize(self.video_widget.size())
        self.video_widget.resizeEvent = lambda e: self.overlay.resize(e.size())

        # Seek bar + playback
        self.seek_bar = SeekBar()
        self.playback_controls = PlaybackControls()

        video_layout = QVBoxLayout()
        video_layout.addWidget(video_container, stretch=1)
        video_layout.addWidget(self.seek_bar)
        video_layout.addWidget(self.playback_controls)

        video_container_layout.setContentsMargins(0, 0, 0, 0)
        video_container_layout.setSpacing(0)

        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(0)

        self._sync_overlay_geometry()
        self.video_widget.resizeEvent = self._on_video_resized

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
        self.sidebar.start_bbox_drawing.connect(self.on_new_object_bbox)
        self.sidebar.add_new_bbox_existing_obj.connect(self.on_existing_object_bbox)
        self.sidebar.open_project_dir.connect(self.on_open_project_dir)
        self.sidebar.quick_save.connect(self.quick_save)

        # Playback controls signals
        self.playback_controls.next_frame_clicked.connect(self.on_next)
        self.playback_controls.prev_frame_clicked.connect(self.on_prev)
        self.playback_controls.play_clicked.connect(self.on_play)

        # Connect annotation signals
        self.video_widget.bbox_drawn.connect(self.handle_drawn_bbox)

        if hasattr(self.playback_controls, "skip_backward_clicked"):
            self.playback_controls.skip_backward_clicked.connect(self.on_skip_back)
        if hasattr(self.playback_controls, "skip_forward_clicked"):
            self.playback_controls.skip_forward_clicked.connect(self.on_skip_forward)

        # Seek bar → jump directly
        self.seek_bar.frame_changed.connect(self.on_seek)
        # TODO add to settings menu
        self._zoom = 1.15
        self.video_widget.set_zoom(self._zoom)
        self.overlay.set_zoom(self._zoom)

        self.video_widget.pan_changed.connect(self.overlay.set_pan)

        from PyQt6.QtGui import QShortcut, QKeySequence

        QShortcut(
            QKeySequence(QKeySequence.StandardKey.ZoomIn), self, activated=self.zoom_in
        )
        QShortcut(
            QKeySequence(QKeySequence.StandardKey.ZoomOut),
            self,
            activated=self.zoom_out,
        )
        QShortcut(QKeySequence("Ctrl+0"), self, activated=self.zoom_fit)

        video_container.setMouseTracking(True)
        video_container.wheelEvent = self._wheel_zoom

    # TODO: Remove once functions implemented
    def noop(*args, **kwargs):
        print("Not implemented yet")

    def open_settings(self):
        # Create the dialog, pass in current values
        dlg = SettingsDialog(theme="Dark", zoom_rate=1.2, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            vals = dlg.values()
            print("New settings:", vals)

    def update_title(self):
        self.setWindowTitle(f"SAVANT {self.project_name}")

    def _show_frame(self, pixmap, frame_idx):
        """Render a frame + overlay; safely handles None at end-of-video."""
        if pixmap is not None:
            self.video_widget.show_frame(pixmap)
        else:
            self.overlay.set_rotated_boxes([])
            return

        if frame_idx is not None:
            self.seek_bar.set_position(int(frame_idx))

        try:
            if frame_idx is not None:

                self._update_overlay_from_model()

                w, h = self.video_controller.size()
                self.overlay.set_frame_size(w, h)
                # TODO: Refactor to put in annotation controller
                rot_boxes = self.project_state_controller.boxes_for_frame(
                    int(frame_idx)
                )
                self.overlay.set_rotated_boxes(rot_boxes)
                self.update_active_objects(frame_idx)

        except Exception:
            self.overlay.set_rotated_boxes([])

    def update_active_objects(self, frame_idx):
        active_objects = self.annotation_controller.get_active_objects(frame_idx)
        self.sidebar.refresh_active_objects(active_objects)

    def refresh_frame(self):
        idx = self.video_controller.current_index()

        # TODO Add check
        pixmap, _ = self.video_controller.jump_to_frame(idx)

        self._show_frame(pixmap, idx)

    def _update_overlay_from_model(self) -> None:
        """Update overlay boxes and the parallel object-id list for the current frame."""
        frame_idx = self.video_controller.current_index()
        try:
            pairs = self.project_state_controller.boxes_with_ids_for_frame(frame_idx)
            self._overlay_ids = [oid for (oid, _) in pairs]
            boxes = [geom for (_, geom) in pairs]

            # Ensure frame size is set before drawing
            w, h = self.video_controller.size()
            self.overlay.set_frame_size(w, h)
            self.overlay.set_rotated_boxes(boxes)
        except Exception:
            self._overlay_ids = []
            self.overlay.set_rotated_boxes([])

    # seek (slider)
    def on_seek(self, index: int):
        try:
            pixmap, idx = self.video_controller.jump_to_frame(index)
            self._show_frame(pixmap, idx)
        except Exception as e:
            QMessageBox.critical(self, "Seek failed", str(e))

    def on_open_project_dir(self, dir_path: str):
        """
        Given a directory, find 1 video and 1 OpenLabel JSON, then load both.
        """
        try:
            folder = Path(dir_path)
            if not folder.is_dir():
                raise ValueError(f"Not a directory: {dir_path}")

            video_exts = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v"}
            json_exts = {".json"}

            videos = sorted(
                [
                    p
                    for p in folder.iterdir()
                    if p.is_file() and p.suffix.lower() in video_exts
                ]
            )
            jsons = sorted(
                [
                    p
                    for p in folder.iterdir()
                    if p.is_file() and p.suffix.lower() in json_exts
                ]
            )

            # TODO: Allow users to open a video only and create a new config
            if not videos:
                raise FileNotFoundError("No video found in folder.")
            if not jsons:
                raise FileNotFoundError("No JSON (OpenLabel) found in folder.")

            preferred_jsons = [p for p in jsons if "openlabel" in p.stem.lower()]
            json_path = preferred_jsons[0] if preferred_jsons else jsons[0]
            video_path = videos[0]

            self.open_openlabel_config(str(json_path))

            if (
                getattr(self.project_state_controller, "project_state", None)
                and getattr(
                    self.project_state_controller.project_state,
                    "annotation_config",
                    None,
                )
                is None
            ):
                return

            self.on_open_video(str(video_path))

            self.setWindowTitle(
                f"SAVANT {self.project_name} — {video_path.name} ({folder.name})"
            )
            QMessageBox.information(
                self,
                "Project Loaded",
                "Project video and OpenLabel configuration loaded successfully.",
            )

        except Exception as e:
            QMessageBox.critical(self, "Open Folder Failed", str(e))

    # File open
    def on_open_video(self, path: str):
        try:
            self.video_controller.load_video(path)
            pixmap, idx = self.video_controller.jump_to_frame(0)
            self._show_frame(pixmap, idx)
            total = self.video_controller.total_frames()
            self.seek_bar.update_range(total)
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
        except Exception as e:
            QMessageBox.critical(self, "Failed to load config", str(e))

    def quick_save(self):
        try:
            self.project_state_controller.save_openlabel_config()
            QMessageBox.information(
                self, "Save Successful", "Project saved successfully."
            )
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", str(e))

    def _on_overlay_box_moved(self, overlay_idx: int, x_center: float, y_center: float) -> None:
        frame_key = self.video_controller.current_index()
        object_key = self._overlay_ids[overlay_idx]
        self.annotation_controller.move_resize_bbox(
            frame_key=frame_key,
            object_key=object_key,
            x_center=x_center,
            y_center=y_center,
        )
        self.refresh_frame()

    def _on_overlay_box_resized(self, overlay_idx: int, x_center: float, y_center: float,
                                width: float, height: float) -> None:
        frame_key = self.video_controller.current_index()
        object_key = self._overlay_ids[overlay_idx]
        self.annotation_controller.move_resize_bbox(
            frame_key=frame_key,
            object_key=object_key,
            x_center=x_center,
            y_center=y_center,
            width=width,
            height=height,
        )
        self.refresh_frame()

    def _on_overlay_box_rotated(self, overlay_idx: int, rotation: float) -> None:
        frame_key = self.video_controller.current_index()
        object_key = self.project_state_controller.object_id_for_frame_index(frame_key, overlay_idx)
        self.annotation_controller.move_resize_bbox(
            frame_key=frame_key,
            object_key=object_key,
            rotation=rotation,
        )
        self._update_overlay_from_model()

    # Navigation
    def on_next(self):
        try:
            pixmap, idx = self.video_controller.next_frame()
            self._show_frame(pixmap, idx)
        except StopIteration:
            self._stop_playback()
            QMessageBox.information(self, "End of video", "No more frames.")
        except Exception as e:
            self._stop_playback()
            QMessageBox.critical(self, "Error", str(e))

    def on_prev(self):
        try:
            pixmap, idx = self.video_controller.previous_frame()
            self._show_frame(pixmap, idx)
        except IndexError:
            QMessageBox.information(self, "At start", "Already at the first frame.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # Play/Pause
    def on_play(self):
        try:
            total = self.video_controller.total_frames()
        except Exception:
            QMessageBox.information(self, "No video", "Load a video first.")
            return

        if not self._is_playing:
            try:
                if self.video_controller.current_index() < 0:
                    pixmap, idx = self.video_controller.jump_to_frame(0)
                    self._show_frame(pixmap, idx)
                if self.video_controller.current_index() >= total - 1:
                    pixmap, idx = self.video_controller.jump_to_frame(0)
                    self._show_frame(pixmap, idx)
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
            if pixmap is None or idx is None:
                raise StopIteration
            self._show_frame(pixmap, idx)
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
            pixmap, idx = self.video_controller.skip_frames(-n)
            self._show_frame(pixmap, idx)
        except Exception as e:
            QMessageBox.critical(self, "Skip backward failed", str(e))

    def on_skip_forward(self, n: int):
        try:
            pixmap, idx = self.video_controller.skip_frames(n)
            self._show_frame(pixmap, idx)
        except Exception as e:
            QMessageBox.critical(self, "Skip forward failed", str(e))

    def on_new_object_bbox(self, object_type: str):
        """Set up state for drawing a new bounding box for a new object."""
        self.video_widget.start_drawing_mode(
            AnnotationState(mode=AnnotationMode.NEW, object_type=object_type)
        )

    def on_existing_object_bbox(self, object_type: str, object_id: str):
        """Set up state for drawing a new bounding box for an existing object."""
        self.video_widget.start_drawing_mode(
            AnnotationState(
                mode=AnnotationMode.EXISTING,
                object_type=object_type,
                object_id=object_id,
            )
        )

    def handle_drawn_bbox(self, annotation: AnnotationState):
        """Handle newly drawn bounding box coordinates from video widget."""
        frame_idx = self.video_controller.current_index()

        try:
            if annotation.mode == AnnotationMode.EXISTING:
                if not annotation.object_id:
                    QMessageBox.warning(
                        self, "No ID", "No object ID provided for existing object."
                    )
                    return
                self.annotation_controller.create_bbox_existing_object(
                    frame_number=frame_idx, bbox_info=asdict(annotation)
                )
            elif annotation.mode == AnnotationMode.NEW:
                self.annotation_controller.create_new_object_bbox(
                    frame_number=frame_idx, bbox_info=asdict(annotation)
                )
            else:
                QMessageBox.warning(
                    self, "Invalid State", "Annotation state is not set correctly."
                )
                return
            # Update UI elements as needed
            self.update_active_objects(frame_idx=frame_idx)

            self.refresh_frame()
        # TODO: Refacftor error handling.
        except Exception as e:
            QMessageBox.critical(self, "Error adding bbox", str(e))
            return

    def _sync_overlay_geometry(self):
        """Ensure overlay matches video widget area exactly (parented to video_widget)."""
        self.overlay.setGeometry(self.video_widget.rect())
        self.overlay.raise_()

    def _on_video_resized(self, e):
        """When video widget resizes, update overlay geometry too."""
        from PyQt6.QtWidgets import QLabel

        QLabel.resizeEvent(self.video_widget, e)
        self._sync_overlay_geometry()

    def _apply_zoom(self, z: float):
        """Apply zoom to both video & overlay and refresh overlay math."""
        self._zoom = max(0.05, min(z, 20.0))
        self.video_widget.set_zoom(self._zoom)
        self.overlay.set_zoom(self._zoom)

        try:
            idx = self.video_controller.current_index()
            if idx >= 0:
                self._update_overlay_from_model()
            else:
                self.overlay.update()
        except Exception:
            self.overlay.update()

    def zoom_fit(self):
        """Reset user zoom to 1.0 (i.e., pure 'fit to window')."""
        self._apply_zoom(1.0)

    def zoom_in(self):
        """Zoom in by 10%."""
        self._apply_zoom(self._zoom * 1.1)

    def zoom_out(self):
        """Zoom out by ~9% (inverse of 1.1)."""
        self._apply_zoom(self._zoom / 1.1)

    def _wheel_zoom(self, event):
        """Ctrl + mouse wheel to zoom in/out."""
        modifiers = event.modifiers()
        if modifiers & (
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier
        ):
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_in()
            elif delta < 0:
                self.zoom_out()
            event.accept()
        else:
            event.ignore()
