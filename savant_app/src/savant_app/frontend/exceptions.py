from typing import Optional


class FrontendException(Exception):
    """Base class for all frontend exceptions."""

    def __init__(self, message: str, log_message: Optional[str] = None):
        super().__init__(message)
        self.log_message = log_message  # In case devs want to include extra log info.


class InvalidDirectoryError(FrontendException):
    """Path is not a valid directory."""

    pass


class MissingVideoError(FrontendException):
    """No video files found in directory."""

    pass


class MissingConfigError(FrontendException):
    """No JSON config files found in directory."""

    pass
