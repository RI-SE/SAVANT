"""Module to hold data types used in the frontend."""

from dataclasses import dataclass
from typing import Literal

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
class Relationship:
    """
    Dataclass representing relationship
    metadata
    """

    subject: str
    relationship_type: str
    object: str
    id: str
