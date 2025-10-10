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


def exception_hook(exc_type, exc_value, exc_tb):
    """Global Qt exception hook for error handling."""

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
    
    elif issubclass(exc_type, )

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
