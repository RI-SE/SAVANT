"""
config - Configuration classes and constants for markit

Contains all configuration dataclasses, constants, and application configuration.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

__version__ = '2.0.0'


class Constants:
    """Constants used throughout the application."""
    MP4V_FOURCC = "mp4v"
    SCHEMA_VERSION = "0.1"
    ANNOTATOR_NAME = f"SAVANT Markit {__version__}"
    ONTOLOGY_URL = "https://savant.ri.se/savant_ontology_1.0.0.ttl"
    # FIXME: To be replaced with uid defined in our ontology
    DEFAULT_CLASS_MAP = {
        0: "vehicle",
        1: "car",
        2: "truck",
        3: "bus"
    }


@dataclass
class DetectionResult:
    """Standardized detection result from any detection engine."""
    object_id: Optional[int]
    class_id: int
    confidence: float
    oriented_bbox: np.ndarray  # 4 corner points for OBB
    center: Tuple[float, float]
    angle: float
    source_engine: str  # 'yolo' or 'optical_flow'


@dataclass
class OpticalFlowParams:
    """Parameters for optical flow detection."""
    motion_threshold: float = 0.5
    min_area: int = 200
    morph_kernel_size: int = 9


@dataclass
class ConflictResolutionConfig:
    """Configuration for detection conflict resolution using IoU."""
    iou_threshold: float = 0.3  # IoU threshold for conflict detection (0.0-1.0)
    yolo_precedence: bool = True  # YOLO takes precedence over optical flow
    enable_logging: bool = False  # Log conflicts for debugging


class MarkitConfig:
    """Configuration class for markit application."""

    def __init__(self, args: argparse.Namespace):
        """Initialize configuration from command line arguments.

        Args:
            args: Parsed command line arguments
        """
        self.weights_path = args.weights
        self.video_path = args.input
        self.output_json_path = args.output_json
        self.schema_path = args.schema
        self.output_video_path = args.output_video
        self.class_map = Constants.DEFAULT_CLASS_MAP.copy()

        # Detection method configuration
        self.use_yolo = args.detection_method in ['yolo', 'both']
        self.use_optical_flow = args.detection_method in ['optical_flow', 'both']

        # Optical flow parameters
        self.optical_flow_params = OpticalFlowParams(
            motion_threshold=args.motion_threshold,
            min_area=args.min_object_area
        )

        # IoU-based conflict resolution configuration
        self.iou_threshold = args.iou_threshold
        self.verbose_conflicts = args.verbose_conflicts
        self.enable_conflict_resolution = not args.disable_conflict_resolution

        # Postprocessing configuration
        self.enable_housekeeping = args.housekeeping
        self.duplicate_avg_iou = args.duplicate_avg_iou
        self.duplicate_min_iou = args.duplicate_min_iou
        self.rotation_threshold = args.rotation_threshold
        self.edge_distance = args.edge_distance

        self.validate_config()

    def validate_config(self) -> None:
        """Validate configuration parameters."""
        # Check required files
        required_files = [self.video_path, self.schema_path]
        if self.use_yolo:
            required_files.append(self.weights_path)

        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")

        if not any([self.use_yolo, self.use_optical_flow]):
            raise ValueError("At least one detection method must be enabled")

        # Validate IoU threshold
        if not (0.0 <= self.iou_threshold <= 1.0):
            raise ValueError(f"IoU threshold must be between 0.0 and 1.0, got: {self.iou_threshold}")
