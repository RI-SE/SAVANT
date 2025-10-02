from pathlib import Path
from PyQt6.QtWidgets import QMessageBox
from .render import show_frame
from .playback import _stop as stop


def wire(main_window):
    main_window.sidebar.open_video.connect(lambda p: on_open_video(main_window, p))
    main_window.sidebar.open_config.connect(lambda p: open_openlabel_config(main_window, p))
    main_window.sidebar.open_project_dir.connect(lambda p: on_open_project_dir(main_window, p))
    main_window.sidebar.quick_save.connect(lambda: quick_save(main_window))


def on_open_video(main_window, path: str):
    try:
        main_window.video_controller.load_video(path)
        pixmap, idx = main_window.video_controller.jump_to_frame(0)
        show_frame(main_window, pixmap, idx)
        main_window.seek_bar.update_range(main_window.video_controller.total_frames())
        if hasattr(main_window.playback_controls, "set_fps"):
            main_window.playback_controls.set_fps(main_window.video_controller.fps())
        stop(main_window)
    except Exception as e:
        QMessageBox.critical(main_window, "Failed to open video", str(e))


def open_openlabel_config(main_window, path: str):
    try:
        main_window.project_state_controller.load_openlabel_config(path)
    except Exception as e:
        QMessageBox.critical(main_window, "Failed to load config", str(e))


def quick_save(main_window):
    try:
        main_window.project_state_controller.save_openlabel_config()
        QMessageBox.information(main_window, "Save Successful", "Project saved successfully.")
    except Exception as e:
        QMessageBox.critical(main_window, "Save Failed", str(e))


def on_open_project_dir(main_window, dir_path: str):
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
        open_openlabel_config(main_window, str(json_path))
        if getattr(main_window.project_state_controller, "project_state", None) and \
           getattr(
               main_window.project_state_controller
               .project_state, "annotation_config", None) is None:
            return
        on_open_video(main_window, str(video_path))
    except Exception as e:
        QMessageBox.critical(main_window, "Open Folder Failed", str(e))
