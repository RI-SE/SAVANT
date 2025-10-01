from .render import show_frame
from PyQt6.QtWidgets import QMessageBox


def wire(mw):
    if hasattr(mw.seek_bar, "frame_changed"):
        mw.seek_bar.frame_changed.connect(lambda idx: on_seek(mw, idx))


def on_seek(mw, index: int):
    try:
        pixmap, idx = mw.video_controller.jump_to_frame(index)

        show_frame(mw, pixmap, idx)
    except Exception as e:
        QMessageBox.critical(mw, "Seek failed", str(e))
