"""
lib - SAVANT Markit library

Internal package for markit.py video processing tool.
Contains detection engines, video processing, postprocessing, and OpenLabel handling.
"""

__version__ = "2.0.0"

from .config import (
    Constants,
    DetectionResult,
    OpticalFlowParams,
    ConflictResolutionConfig,
    MarkitConfig,
)
from .geometry import BBoxOverlapCalculator

__all__ = [
    "Constants",
    "DetectionResult",
    "OpticalFlowParams",
    "ConflictResolutionConfig",
    "MarkitConfig",
    "BBoxOverlapCalculator",
    "__version__",
]
