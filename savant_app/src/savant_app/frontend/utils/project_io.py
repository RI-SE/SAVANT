from pathlib import Path
from PyQt6.QtWidgets import QMessageBox
from .render import show_frame
from .playback import _stop as stop
from ..exceptions import InvalidDirectoryError, MissingVideoError, MissingConfigError


def wire(main_window):
    main_window.sidebar.open_video.connect(lambda p: on_open_video(main_window, p))
    main_window.sidebar.open_config.connect(
        lambda p: open_openlabel_config(main_window, p)
    )
    main_window.sidebar.open_project_dir.connect(
        lambda p: on_open_project_dir(main_window, p)
    )
    main_window.sidebar.quick_save.connect(lambda: quick_save(main_window))


def on_open_video(main_window, path: str):
    main_window.video_controller.load_video(path)
    pixmap, idx = main_window.video_controller.jump_to_frame(0)
    show_frame(main_window, pixmap, idx)
    # Update seek bar range based on total frame count
    main_window.seek_bar.update_range(
        main_window.project_state_controller.get_frame_count()
    )
    if hasattr(main_window.playback_controls, "set_fps"):
        main_window.playback_controls.set_fps(
            main_window.project_state_controller.get_fps()
        )
    stop(main_window)


def open_openlabel_config(main_window, path: str):
    main_window.project_state_controller.load_openlabel_config(path)


def quick_save(main_window):
    main_window.project_state_controller.validate_before_save()
    main_window.project_state_controller.save_openlabel_config()
    QMessageBox.information(
        main_window, "Save Successful", "Project saved successfully."
    )


def on_open_project_dir(main_window, dir_path: str):
    folder = Path(dir_path)
    if not folder.is_dir():
        raise InvalidDirectoryError(f"Not a directory: {dir_path}")
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v"}
    json_exts = {".json"}
    videos = sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in video_exts]
    )
    jsons = sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in json_exts]
    )
    if not videos:
        raise MissingVideoError("No video found in folder.")
    if not jsons:
        raise MissingConfigError("No JSON (OpenLabel) found in folder.")
    preferred_jsons = [p for p in jsons if "openlabel" in p.stem.lower()]
    json_path = preferred_jsons[0] if preferred_jsons else jsons[0]
    video_path = videos[0]
    open_openlabel_config(main_window, str(json_path))
    # TODO: Refactor the getattr stuff?
    if (
        getattr(main_window.project_state_controller, "project_state", None)
        and getattr(
            main_window.project_state_controller.project_state,
            "annotation_config",
            None,
        )
        is None
    ):
        return
    on_open_video(main_window, str(video_path))
