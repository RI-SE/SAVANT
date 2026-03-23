import sys
import traceback
import logging
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtCore import QTimer
from .services.exceptions import DomainException, InternalException
from .frontend.exceptions import FrontendException, FrontendDevException

# Create a proper logger instance
logger = logging.getLogger(__name__)


def show_error_box(message: str, title: str = "Error"):
    """Display a critical error dialog safely."""
    QMessageBox.critical(None, title, message)


def _release_any_mouse_grab():
    """Release mouse grab on any widget to prevent X11 desktop freeze."""
    from PyQt6.QtWidgets import QApplication
    widget = QApplication.instance() and QApplication.activeWindow()
    if widget and hasattr(widget, "video_widget"):
        vw = widget.video_widget
        if hasattr(vw, "_force_cancel_all_gestures"):
            vw._force_cancel_all_gestures()


def exception_hook(exc_type, exc_value, exc_tb):
    """Global Qt exception hook for error handling."""

    # Always release mouse grabs on any exception to prevent X11 freeze
    try:
        _release_any_mouse_grab()
    except Exception:
        pass

    # Recoverable domain-level errors
    if issubclass(exc_type, DomainException):
        QTimer.singleShot(0, lambda: show_error_box(str(exc_value), "Warning"))
        return

    # Internal / unrecoverable errors
    elif issubclass(exc_type, InternalException):
        err_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        QTimer.singleShot(0, lambda: show_error_box(str(exc_value), "Critical Error"))
        # Call default excepthook to still print traceback
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return

    # Recoverable frontend errors
    elif issubclass(exc_type, FrontendException):
        QTimer.singleShot(0, lambda: show_error_box(str(exc_value), "Warning"))
        return

    elif issubclass(exc_type, FrontendDevException):
        err_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logger.error("Frontend development error: %s", err_msg)
        QTimer.singleShot(
            0,
            lambda: show_error_box(
                """An unexpected error occurred.\nPlease contact support.\nDetails logged.
                """,
                "Unexpected Error",
            ),
        )
        return

    # Any other unhandled exceptions
    else:
        err_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logger.error("Unhandled exception: %s", err_msg)
        QTimer.singleShot(
            0,
            lambda: show_error_box(
                """An unexpected error occurred.\nPlease contact support.\nDetails logged.
                """,
                "Unexpected Error",
            ),
        )
        sys.__excepthook__(exc_type, exc_value, exc_tb)
