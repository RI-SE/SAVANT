from typing import Optional


class FrontendException(Exception):
    """Base class for all frontend exceptions."""

    def __init__(self, message: str, log_message: Optional[str] = None):
        super().__init__(message)
        self.log_message = log_message  # In case devs want to include extra log info.


class FrontendDevException(Exception):
    """Base class for all frontend exceptions that are not user related"""

    def __init__(self, message: str):
        super().__init__(message)


class InvalidDirectoryError(FrontendException):
    """Path is not a valid directory."""

    pass


class MissingVideoError(FrontendException):
    """No video files found in directory."""

    pass


class MissingConfigError(FrontendException):
    """No JSON config files found in directory."""

    pass


class MissingObjectIDError(FrontendException):
    """No object ID for the selected bounding box."""

    pass


class InvalidObjectIDFormat(FrontendDevException):
    """Raised when an object ID does not conform to expected format."""

    pass


class InvalidFrameRangeInput(FrontendException):
    """Raised when the user inputs an invalid frame range for cascade operations."""

    pass


class OntologyNotFound(FrontendException):
    """Raised when an ontology file can not be found."""

    pass
