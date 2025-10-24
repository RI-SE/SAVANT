"""
config - Configuration classes and constants for markit

Contains all configuration dataclasses, constants, and application configuration.
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# Add parent directory to path to import common package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common.ontology import create_class_map

__version__ = '2.0.1'


class Constants:
    """Constants used throughout the application."""
    MP4V_FOURCC = "mp4v"
    SCHEMA_VERSION = "0.1"
    ANNOTATOR_NAME = f"SAVANT Markit {__version__}"
    ONTOLOGY_URL = "https://savant.ri.se/savant_ontology_1.2.0.ttl"
    # Fallback class map used when ontology file cannot be loaded
    # In normal operation, class_map is loaded dynamically from the ontology
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
        self.ontology_path = args.ontology

        # ArUco detection configuration
        self.aruco_csv_path = args.aruco_csv if hasattr(args, 'aruco_csv') else None
        self.use_aruco = self.aruco_csv_path is not None

        # Load class map from ontology if provided, otherwise use default
        self.class_map = self._load_class_map()

        # Get ArUco class ID from ontology (if ArUco detection enabled)
        self.aruco_class_id = None
        if self.use_aruco:
            self.aruco_class_id = self._get_aruco_class_id()

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
        self.min_movement_pixels = args.min_movement_pixels
        self.temporal_smoothing = args.temporal_smoothing
        self.edge_distance = args.edge_distance
        self.static_threshold = args.static_threshold
        self.static_mark = args.static_mark

        self.validate_config()

    def validate_config(self) -> None:
        """Validate configuration parameters."""
        # Check required files
        required_files = [self.video_path, self.schema_path]
        if self.use_yolo:
            required_files.append(self.weights_path)
        if self.use_aruco:
            required_files.append(self.aruco_csv_path)

        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")

        if not any([self.use_yolo, self.use_optical_flow, self.use_aruco]):
            raise ValueError("At least one detection method must be enabled")

        # Validate IoU threshold
        if not (0.0 <= self.iou_threshold <= 1.0):
            raise ValueError(f"IoU threshold must be between 0.0 and 1.0, got: {self.iou_threshold}")

    def _load_class_map(self) -> Dict[int, str]:
        """Load class map from ontology file or use default.

        Returns:
            Dictionary mapping class_id → label

        Raises:
            FileNotFoundError: If ontology file doesn't exist
            Exception: If ontology parsing fails
        """
        logger = logging.getLogger(__name__)

        # If no ontology path specified, use default map
        if not self.ontology_path:
            logger.info("Class map source: DEFAULT_CLASS_MAP (no ontology specified)")
            return Constants.DEFAULT_CLASS_MAP.copy()

        # Check if ontology file exists
        if not os.path.exists(self.ontology_path):
            logger.warning(f"Ontology file not found: {self.ontology_path}")
            logger.info("Class map source: DEFAULT_CLASS_MAP (fallback)")
            return Constants.DEFAULT_CLASS_MAP.copy()

        try:
            # Load all classes with UIDs from ontology
            class_map = create_class_map(self.ontology_path)

            if not class_map:
                logger.warning(f"No classes with UIDs found in ontology: {self.ontology_path}")
                logger.info("Class map source: DEFAULT_CLASS_MAP (fallback)")
                return Constants.DEFAULT_CLASS_MAP.copy()

            logger.info(f"Class map source: {self.ontology_path} ({len(class_map)} classes)")
            return class_map

        except Exception as e:
            logger.warning(f"Failed to load ontology: {e}")
            logger.info("Class map source: DEFAULT_CLASS_MAP (fallback)")
            return Constants.DEFAULT_CLASS_MAP.copy()

    def _get_aruco_class_id(self) -> int:
        """Get class ID for MarkerAruco from ontology.

        Returns:
            Class ID for ArUco markers

        Raises:
            ValueError: If MarkerAruco class not found in ontology
        """
        logger = logging.getLogger(__name__)

        # Search for MarkerAruco in class_map
        for class_id, class_name in self.class_map.items():
            if class_name == "Aruco" or class_name == "MarkerAruco":
                logger.info(f"Found ArUco class in ontology: ID={class_id}, Name={class_name}")
                return class_id

        # MarkerAruco not found - log error and raise exception
        logger.error("MarkerAruco class not found in ontology")
        logger.error("Please ensure the ontology contains a 'MarkerAruco' class with label 'Aruco'")
        raise ValueError("MarkerAruco class not found in ontology - ArUco detection cannot proceed")
