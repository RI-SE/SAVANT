from typing import Optional


class InternalException(Exception):
    """
    Raised when an internal error occurs.

    This is raised by default to the frontend when
    an unexpected error occurs in the backend.
    """

    def __init__(self, message: str = "An internal error occurred."):
        super().__init__(message)


class DomainException(Exception):
    """Base class for domain-specific exceptions."""

    def __init__(
        self, message: Optional[str] = None, log_message: Optional[str] = None
    ):
        super().__init__(message)
        self.log_message = log_message  # In case devs want to include extra log info.


class ObjectInFrameError(DomainException):
    """Raised when an object already has a bbox in the specified frame."""

    pass


class BBoxNotFoundError(DomainException):
    """Raised when bounding boxes are not found in the config"""

    pass


class ObjectNotFoundError(DomainException):
    """Raised when an object ID does not exist in the annotation config."""

    pass


class FrameNotFoundError(DomainException):
    """Raised when a frame number does not exist in the annotation config."""

    pass


class InvalidFrameRangeError(DomainException):
    """Raised when the provided frame range is invalid."""

    pass


class OpenLabelFileNotFoundError(DomainException):
    """Raised when open label file not found"""

    pass


class OpenLabelFileNotValid(DomainException):
    """Raised when the open label file is not following the spec"""

    pass


class VideoLoadError(DomainException):
    """Raised when video file can't be loaded"""

    pass


class VideoFrameIndexError(DomainException):
    """Raised when an invalid frame index is accessed"""

    pass


class VideoReadError(DomainException):
    """Raised when a video frame cannot be read"""

    pass


class OverlayIndexError(DomainException):
    """Raised when an invalid overlay index is accessed"""

    pass


class InvalidInputError(DomainException):
    """Raised when the input provided is invalid."""

    pass


class NoFrameLabelFoundError(DomainException):
    """Raised when no action label is found in the config"""

    pass


class UnsportedTagTypeError(DomainException):
    """Raised when an unsupported tag type is encountered"""

    pass


class OntologyNotFound(DomainException):
    """Raised when an ontology file can not be found"""

    pass


class ObjectLinkConflictError(DomainException):
    """Raised when linking object IDs would create an invalid state."""

    pass
