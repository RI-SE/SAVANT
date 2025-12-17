"""Helpers for styling sidebar-specific widgets."""

from __future__ import annotations

from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QPushButton

from .constants import (
    SIDEBAR_SORT_BUTTON_ACTIVE_FONT_WEIGHT,
    SIDEBAR_SORT_BUTTON_ACTIVE_PADDING_H,
    SIDEBAR_SORT_BUTTON_ACTIVE_PADDING_V,
    SIDEBAR_SORT_BUTTON_BORDER_RADIUS,
    SIDEBAR_SORT_BUTTON_FONT_SIZE,
    SIDEBAR_SORT_BUTTON_INACTIVE_FONT_WEIGHT,
    SIDEBAR_SORT_BUTTON_INACTIVE_PADDING_H,
    SIDEBAR_SORT_BUTTON_INACTIVE_PADDING_V,
)


def apply_issue_sort_button_style(
    button: QPushButton, *, active: bool, palette: QPalette | None = None
) -> None:
    """
    Apply the theme-aware stylesheet for the issue sort buttons.
    """
    if palette is None:
        palette = button.palette()

    highlight_hex = palette.color(QPalette.ColorRole.Highlight).name()
    highlight_text_hex = palette.color(QPalette.ColorRole.HighlightedText).name()
    button_text_hex = palette.color(QPalette.ColorRole.ButtonText).name()

    if active:
        padding_v = SIDEBAR_SORT_BUTTON_ACTIVE_PADDING_V
        padding_h = SIDEBAR_SORT_BUTTON_ACTIVE_PADDING_H
        font_weight = SIDEBAR_SORT_BUTTON_ACTIVE_FONT_WEIGHT
        background = highlight_hex
        text_color = highlight_text_hex
    else:
        padding_v = SIDEBAR_SORT_BUTTON_INACTIVE_PADDING_V
        padding_h = SIDEBAR_SORT_BUTTON_INACTIVE_PADDING_H
        font_weight = SIDEBAR_SORT_BUTTON_INACTIVE_FONT_WEIGHT
        background = "transparent"
        text_color = button_text_hex

    padding_css = f"{padding_v}px {padding_h}px"
    button.setStyleSheet(
        (
            "QPushButton {{ border: none; padding: {padding}; font-size: {font_size}px; "
            "font-weight: {font_weight}; border-radius: {radius}px; "
            "color: {text_color}; background-color: {background}; }}"
        ).format(
            padding=padding_css,
            font_size=SIDEBAR_SORT_BUTTON_FONT_SIZE,
            font_weight=font_weight,
            radius=SIDEBAR_SORT_BUTTON_BORDER_RADIUS,
            text_color=text_color,
            background=background,
        )
    )
