from PyQt6.QtGui import QKeySequence, QShortcut


def setup_measure_shortcuts(main_window) -> None:
    """Wire the M key to toggle measure mode on the overlay."""

    def _toggle():
        main_window.overlay.toggle_measure_mode()

    QShortcut(QKeySequence("M"), main_window, activated=_toggle)
