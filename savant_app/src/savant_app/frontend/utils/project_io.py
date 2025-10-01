from pathlib import Path
from PyQt6.QtWidgets import QMessageBox
from .render import show_frame
from .playback import _stop as stop


def wire(mw):
    mw.sidebar.open_video.connect(lambda p: on_open_video(mw, p))
    mw.sidebar.open_config.connect(lambda p: open_openlabel_config(mw, p))
    mw.sidebar.open_project_dir.connect(lambda p: on_open_project_dir(mw, p))
    mw.sidebar.quick_save.connect(lambda: quick_save(mw))


def on_open_video(mw, path: str):
    try:
        mw.video_controller.load_video(path)
        pixmap, idx = mw.video_controller.jump_to_frame(0)
        show_frame(mw, pixmap, idx)
        mw.seek_bar.update_range(mw.video_controller.total_frames())
        if hasattr(mw.playback_controls, "set_fps"):
            mw.playback_controls.set_fps(mw.video_controller.fps())
        stop(mw)
    except Exception as e:
        QMessageBox.critical(mw, "Failed to open video", str(e))


def open_openlabel_config(mw, path: str):
    try:
        mw.project_state_controller.load_openlabel_config(path)
    except Exception as e:
        QMessageBox.critical(mw, "Failed to load config", str(e))


def quick_save(mw):
    try:
        mw.project_state_controller.save_openlabel_config()
        QMessageBox.information(mw, "Save Successful", "Project saved successfully.")
    except Exception as e:
        QMessageBox.critical(mw, "Save Failed", str(e))


def on_open_project_dir(mw, dir_path: str):
    try:
        folder = Path(dir_path)
        if not folder.is_dir():
            raise ValueError(f"Not a directory: {dir_path}")
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v"}
        json_exts = {".json"}
        videos = sorted([p for p in folder.iterdir() if p.is_file() and
                         p.suffix.lower() in video_exts])
        jsons = sorted([p for p in folder.iterdir() if p.is_file() and
                        p.suffix.lower() in json_exts])
        if not videos:
            raise FileNotFoundError("No video found in folder.")
        if not jsons:
            raise FileNotFoundError("No JSON (OpenLabel) found in folder.")
        preferred_jsons = [p for p in jsons if "openlabel" in p.stem.lower()]
        json_path = preferred_jsons[0] if preferred_jsons else jsons[0]
        video_path = videos[0]
        open_openlabel_config(mw, str(json_path))
        if getattr(mw.project_state_controller, "project_state", None) and \
           getattr(mw.project_state_controller.project_state, "annotation_config", None) is None:
            return
        on_open_video(mw, str(video_path))
    except Exception as e:
        QMessageBox.critical(mw, "Open Folder Failed", str(e))
