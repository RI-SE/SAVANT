"""
passes - Postprocessing pass implementations

Contains all postprocessing passes for gap detection/filling, duplicate removal,
rotation adjustment, sudden event detection, and frame interval calculation.
"""

from ._passes import *  # noqa: F401,F403
from ._passes import (  # noqa: F401
    GapDetectionPass,
    GapFillingPass,
    DuplicateRemovalPass,
    FirstDetectionRefinementPass,
    RotationAdjustmentPass,
    SuddenPass,
    FrameIntervalPass,
    StaticObjectRemovalPass,
    ShortDurationPass,
    BboxSmoothingPass,
    SizeOutlierFilterPass,
    SizeStepDetectionPass,
    Rotation90JumpFixPass,
    RotationTemporalSmoothingPass,
    AngleNormalizationPass,
)
