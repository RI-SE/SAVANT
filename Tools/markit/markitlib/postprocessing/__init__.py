"""
postprocessing - OpenLabel data postprocessing pipeline

Contains postprocessing passes for gap detection/filling, duplicate removal,
rotation adjustment, sudden event detection, and frame interval calculation.
"""

from .base import PostprocessingPass
from .passes import (
    GapDetectionPass,
    GapFillingPass,
    DuplicateRemovalPass,
    RotationAdjustmentPass,
    SuddenPass,
    FrameIntervalPass,
)
from .pipeline import PostprocessingPipeline

__all__ = [
    'PostprocessingPass',
    'GapDetectionPass',
    'GapFillingPass',
    'DuplicateRemovalPass',
    'RotationAdjustmentPass',
    'SuddenPass',
    'FrameIntervalPass',
    'PostprocessingPipeline',
]
