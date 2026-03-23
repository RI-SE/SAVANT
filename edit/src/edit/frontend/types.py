"""Module to hold data types used in the frontend."""

from dataclasses import dataclass
from typing import Literal, Optional

ConfidenceSeverity = Literal["warning", "error"]
ConfidenceFlagMap = dict[str, ConfidenceSeverity]


@dataclass
class BBoxData:
    object_id: str
    object_type: str
    center_x: float
    center_y: float
    width: float
    height: float
    theta: float  # in radians
    is_interpolated: bool = False


@dataclass
class BBoxDimensionData:
    """
    Dataclass representing bounding box
    dimension data.
    """

    x_center: float
    y_center: float
    width: float
    height: float
    rotation: float


@dataclass
class BBoxResizedEvent:
    """Payload for the boxResized signal."""

    object_id: str
    center_x: float
    center_y: float
    width: float
    height: float
    rotation: float


@dataclass
class BBoxRotatedEvent:
    """Payload for the boxRotated signal."""

    object_id: str
    width: float
    height: float
    rotation: float


@dataclass
class CascadeApplyEvent:
    """Payload for cascadeApplyAll and cascadeApplyFrameRange signals."""

    object_id: str
    center_x: Optional[float]
    center_y: Optional[float]
    width: Optional[float]
    height: Optional[float]
    rotation: Optional[float]
    direction: str


@dataclass
class Relationship:
    """
    Dataclass representing relationship
    metadata
    """

    subject: str
    relationship_type: str
    object: str
    id: str
