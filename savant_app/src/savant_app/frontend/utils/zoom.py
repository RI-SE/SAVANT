from PyQt6.QtGui import QShortcut, QKeySequence
from PyQt6.QtCore import Qt


def wire(main_window, initial: float = 1.0):

    def _clamp(zoom_value: float) -> float:
        return max(0.05, min(zoom_value, 20.0))

    def _apply_zoom(zoom_value: float, anchor_position=None):
        main_window._zoom = _clamp(zoom_value)
        if hasattr(main_window.video_widget, "set_zoom"):
            if anchor_position is not None:
                main_window.video_widget.set_zoom(main_window._zoom, anchor_position)
            else:
                main_window.video_widget.set_zoom(main_window._zoom)
        if hasattr(main_window.overlay, "set_zoom"):
            main_window.overlay.set_zoom(main_window._zoom)
        if hasattr(main_window.overlay, "update"):
            main_window.overlay.update()

    def zoom_in(anchor_position=None):
        _apply_zoom(main_window._zoom * 1.1, anchor_position)

    def zoom_out(anchor_position=None):
        _apply_zoom(main_window._zoom / 1.1, anchor_position)

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
            cursor_position = event.position()
            if delta > 0:
                zoom_in(cursor_position)
            elif delta < 0:
                zoom_out(cursor_position)
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
