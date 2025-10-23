"""
processing - Video processing and object detection

Contains video processor, detection engines, conflict resolution, and frame annotation.
All components work together in the detection/tracking pipeline.
"""

from .engines import (
    BaseDetectionEngine,
    YOLOEngine,
    OpticalFlowEngine,
    SimpleTracker,
)
from .conflict_resolution import DetectionConflictResolver
from .video_processor import VideoProcessor, FrameAnnotator

__all__ = [
    'BaseDetectionEngine',
    'YOLOEngine',
    'OpticalFlowEngine',
    'SimpleTracker',
    'DetectionConflictResolver',
    'VideoProcessor',
    'FrameAnnotator',
]
