from .render import show_frame
from PyQt6.QtWidgets import QMessageBox


def wire(main_window):
    if hasattr(main_window.seek_bar, "frame_changed"):
        main_window.seek_bar.frame_changed.connect(lambda idx: on_seek(main_window, idx))


def on_seek(main_window, index: int):
    pixmap, idx = main_window.video_controller.jump_to_frame(index)

    show_frame(main_window, pixmap, idx)