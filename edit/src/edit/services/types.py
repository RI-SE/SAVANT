"""Module for defining data types used in the edit services."""

from dataclasses import dataclass


@dataclass
class VideoMetadata:
    frame_count: int = 0
    width: int = 0
    height: int = 0
    fps: float = 0.0
