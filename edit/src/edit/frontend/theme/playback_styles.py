"""Playback control styling helpers."""

from __future__ import annotations

from PyQt6.QtWidgets import QPushButton

from .constants import (
    PLAYBACK_ISSUE_BUTTON_BACKGROUND,
    PLAYBACK_ISSUE_BUTTON_BACKGROUND_DISABLED,
    PLAYBACK_ISSUE_BUTTON_BACKGROUND_HOVER,
    PLAYBACK_ISSUE_BUTTON_BORDER_RADIUS,
    PLAYBACK_ISSUE_BUTTON_PADDING,
)


def apply_issue_navigation_button_style(button: QPushButton) -> None:
    """Apply the shared stylesheet for the issue navigation buttons."""
    if button is None:
        return

    base_hex = PLAYBACK_ISSUE_BUTTON_BACKGROUND.name()
    hover_hex = PLAYBACK_ISSUE_BUTTON_BACKGROUND_HOVER.name()
    disabled_hex = PLAYBACK_ISSUE_BUTTON_BACKGROUND_DISABLED.name()
    padding = f"{PLAYBACK_ISSUE_BUTTON_PADDING}px"
    radius = f"{PLAYBACK_ISSUE_BUTTON_BORDER_RADIUS}px"

    button.setStyleSheet(
        (
            "QPushButton {{ background-color: {base}; border: none; "
            "border-radius: {radius}; padding: {padding}; }}"
            "QPushButton:hover:!disabled {{ background-color: {hover}; }}"
            "QPushButton:pressed {{ background-color: {hover}; }}"
            "QPushButton:disabled {{ background-color: {disabled}; }}"
        ).format(
            base=base_hex,
            hover=hover_hex,
            disabled=disabled_hex,
            radius=radius,
            padding=padding,
        )
    )
