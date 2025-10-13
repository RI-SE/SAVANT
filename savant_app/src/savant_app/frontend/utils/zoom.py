from PyQt6.QtGui import QShortcut, QKeySequence
from PyQt6.QtCore import Qt


def wire(main_window, initial: float = 1.0):

    def _clamp(z: float) -> float:
        return max(0.05, min(z, 20.0))

    def _apply_zoom(z: float):
        main_window._zoom = _clamp(z)
        if hasattr(main_window.video_widget, "set_zoom"):
            main_window.video_widget.set_zoom(main_window._zoom)
        if hasattr(main_window.overlay, "set_zoom"):
            main_window.overlay.set_zoom(main_window._zoom)
        if hasattr(main_window.overlay, "update"):
            main_window.overlay.update()

    def zoom_in():
        _apply_zoom(main_window._zoom * 1.1)

    def zoom_out():
        _apply_zoom(main_window._zoom / 1.1)

    def zoom_fit():
        _apply_zoom(1.0)

    main_window._zoom = initial
    _apply_zoom(main_window._zoom)

    def _wheel_zoom(event):
        mods = event.modifiers()
        if mods & (
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier
        ):
            delta = event.angleDelta().y()
            if delta > 0:
                zoom_in()
            elif delta < 0:
                zoom_out()
            event.accept()
        else:
            event.ignore()

    if hasattr(main_window.video_widget, "setMouseTracking"):
        main_window.video_widget.setMouseTracking(True)
    main_window.video_widget.wheelEvent = _wheel_zoom

    QShortcut(
        QKeySequence(QKeySequence.StandardKey.ZoomIn), main_window, activated=zoom_in
    )
    QShortcut(
        QKeySequence(QKeySequence.StandardKey.ZoomOut), main_window, activated=zoom_out
    )
    QShortcut(QKeySequence("Ctrl+0"), main_window, activated=zoom_fit)

    main_window.zoom_in = zoom_in
    main_window.zoom_out = zoom_out
    main_window.zoom_fit = zoom_fit
