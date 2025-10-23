# savant_app/frontend/theme/menu_styler.py
from __future__ import annotations
from PyQt6.QtCore import QObject, QEvent, Qt
from PyQt6.QtWidgets import QApplication, QMenu


def _menu_css() -> str:
    palette = QApplication.palette()
    background_hex = palette.color(palette.ColorRole.Base).name()
    text_hex = palette.color(palette.ColorRole.WindowText).name()
    highlight_hex = palette.color(palette.ColorRole.Highlight).name()
    highlight_text_hex = palette.color(palette.ColorRole.HighlightedText).name()
    border_hex = palette.color(palette.ColorRole.Mid).name()
    return f"""
    QMenu {{
      background-color: {background_hex};
      color: {text_hex};
      border: 1px solid {border_hex};
    }}
    QMenu::item {{
      background-color: transparent;
      padding: 6px 12px;
    }}
    QMenu::item:selected {{
      background-color: {highlight_hex};
      color: {highlight_text_hex};
    }}
    QMenu::separator {{
      height: 1px;
      background-color: {border_hex};
      margin: 4px 8px;
    }}
    """


class _MenuStyler(QObject):
    def eventFilter(self, obj, event):
        if isinstance(obj, QMenu) and event.type() in (QEvent.Type.Polish, QEvent.Type.Show):
            obj.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            obj.setStyleSheet(_menu_css())
        return False


def install_menu_styler(app) -> None:
    app.setStyle("Fusion")
    app.installEventFilter(_MenuStyler(app))
