""" Module to hold data types used in the frontend. """

from dataclasses import dataclass

@dataclass
class BBoxData:
    object_id: str
    object_type: str
    center_x: float
    center_y: float
    width: float
    height: float
    theta: float  # in radians