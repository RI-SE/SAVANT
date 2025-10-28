"""Shared UI constants for colors, spacing, and other theme values."""

from __future__ import annotations

from PyQt6.QtGui import QColor, QPixmap

from savant_app.frontend.utils.assets import asset_path

# Seek bar markers
SEEK_BAR_MARKER_THICKNESS = 2
SEEK_BAR_WARNING_MARKER_COLOR = QColor("#ffbb00")
SEEK_BAR_ERROR_MARKER_COLOR = QColor("#ff1500")

# Sidebar warning/error highlight colours
SIDEBAR_WARNING_HIGHLIGHT = QColor("#ebc139")
SIDEBAR_ERROR_HIGHLIGHT = QColor("#eb414f")
SIDEBAR_HIGHLIGHT_TEXT_COLOUR = QColor("#000000")

# Overlay warning/error icon
_WARNING_ICON: QPixmap | None = None
_ERROR_ICON: QPixmap | None = None
OVERLAY_CONFIDENCE_ICON_SIZE = 28
OVERLAY_ICON_SPACING = 6


def _load_icon(filename: str) -> QPixmap:
    """Load an icon pixmap from the bundled assets directory."""
    return QPixmap(asset_path(filename))


def get_warning_icon() -> QPixmap:
    """Return the warning icon pixmap, loading it lazily."""
    global _WARNING_ICON
    if _WARNING_ICON is None:
        _WARNING_ICON = _load_icon("warning_icon.png")
    return _WARNING_ICON


def get_error_icon() -> QPixmap:
    """Return the error icon pixmap, loading it lazily."""
    global _ERROR_ICON
    if _ERROR_ICON is None:
        _ERROR_ICON = _load_icon("error_icon.png")
    return _ERROR_ICON


def __getattr__(name: str) -> QPixmap:
    if name == "WARNING_ICON":
        return get_warning_icon()
    if name == "ERROR_ICON":
        return get_error_icon()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
