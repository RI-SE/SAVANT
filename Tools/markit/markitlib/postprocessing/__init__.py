"""
postprocessing - OpenLabel data postprocessing pipeline

Contains postprocessing passes for gap detection/filling, duplicate removal,
rotation adjustment, sudden event detection, frame interval calculation,
and static object removal.
"""

from .base import PostprocessingPass
from .passes import (
    GapDetectionPass,
    GapFillingPass,
    DuplicateRemovalPass,
    RotationAdjustmentPass,
    SuddenPass,
    FrameIntervalPass,
    StaticObjectRemovalPass,
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
    'StaticObjectRemovalPass',
    'PostprocessingPipeline',
]
