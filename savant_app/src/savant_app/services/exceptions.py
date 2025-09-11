class ObjectInFrameError(Exception):
    """Raised when an object already has a bbox in the specified frame."""
    pass

class ObjectNotFoundError(Exception):
    """Raised when an object ID does not exist in the annotation config."""
    pass