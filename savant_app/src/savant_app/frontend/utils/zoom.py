from PyQt6.QtGui import QShortcut, QKeySequence
from PyQt6.QtCore import Qt


def wire(mw, initial: float = 1.0):

    def _clamp(z: float) -> float:
        return max(0.05, min(z, 20.0))

    def _apply_zoom(z: float):
        mw._zoom = _clamp(z)
        if hasattr(mw.video_widget, "set_zoom"):
            mw.video_widget.set_zoom(mw._zoom)
        if hasattr(mw.overlay, "set_zoom"):
            mw.overlay.set_zoom(mw._zoom)
        if hasattr(mw.overlay, "update"):
            mw.overlay.update()

    def zoom_in():  _apply_zoom(mw._zoom * 1.1)
    def zoom_out(): _apply_zoom(mw._zoom / 1.1)
    def zoom_fit(): _apply_zoom(1.0)

    mw._zoom = initial
    _apply_zoom(mw._zoom)

    def _wheel_zoom(event):
        mods = event.modifiers()
        if mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier):
            delta = event.angleDelta().y()
            if delta > 0:
                zoom_in()
            elif delta < 0:
                zoom_out()
            event.accept()
        else:
            event.ignore()

    if hasattr(mw.video_widget, "setMouseTracking"):
        mw.video_widget.setMouseTracking(True)
    mw.video_widget.wheelEvent = _wheel_zoom

    QShortcut(QKeySequence(QKeySequence.StandardKey.ZoomIn), mw, activated=zoom_in)
    QShortcut(QKeySequence(QKeySequence.StandardKey.ZoomOut), mw, activated=zoom_out)
    QShortcut(QKeySequence("Ctrl+0"), mw, activated=zoom_fit)

    mw.zoom_in = zoom_in
    mw.zoom_out = zoom_out
    mw.zoom_fit = zoom_fit
