from __future__ import annotations

from PyQt6.QtCore import QObject, QEvent
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QCheckBox


def style_checkbox(checkbox: QCheckBox) -> None:
    """
    Apply a theme-aware style to a checkbox indicator so it stays visible on both
    light and dark palettes.
    """
    if checkbox is None:
        return

    def apply_palette(_: QObject | None = None) -> None:
        palette: QPalette = checkbox.palette()
        text_color = palette.windowText().color().name()
        highlight_color = palette.highlight().color().name()

        checkbox.setStyleSheet(
            f"""
            QCheckBox::indicator {{
                width: 14px;
                height: 14px;
                border-radius: 4px;
                border: 1px solid {text_color};
            }}
            QCheckBox::indicator:checked {{
                background-color: {highlight_color};
                border: 1px solid {text_color};
            }}
            """
        )

    apply_palette()

    class _PaletteWatcher(QObject):
        def __init__(self, target: QCheckBox, callback):
            super().__init__(target)
            self._target = target
            self._callback = callback

        def eventFilter(self, obj: QObject, event: QEvent) -> bool:
            if obj is self._target and event.type() == QEvent.Type.PaletteChange:
                self._callback()
            return False

    watcher = _PaletteWatcher(checkbox, apply_palette)
    checkbox.installEventFilter(watcher)
    checkbox.destroyed.connect(lambda *_: watcher.deleteLater())
    setattr(checkbox, "_savant_palette_watcher", watcher)
