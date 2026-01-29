"""
config - Configuration classes and constants for markit

Contains all configuration dataclasses, constants, and application configuration.
"""

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from savant_common.ontology import create_class_map, extract_namespace_uri

from . import __version__


class Constants:
    """Constants used throughout the application."""

    MP4V_FOURCC = "mp4v"  # MPEG-4 Part 2 (widely supported; H.264 tried first if available)
    SCHEMA_VERSION = "1.1"
    ANNOTATOR_NAME = f"SAVANT markit v{__version__}"
    # Fallback class map used when ontology file cannot be loaded
    # In normal operation, class_map is loaded dynamically from the ontology
    DEFAULT_CLASS_MAP = {0: "vehicle", 1: "car", 2: "truck", 3: "bus"}


@dataclass
class DetectionResult:
    """Standardized detection result with semantic representation.

    Convention:
    - width: Long axis (semantic, never swaps)
    - height: Short axis (semantic, never swaps)
    - angle: Continuous rotation in (-∞, +∞), rebased at ±2π
    - positive x-axis is 0 radians, rotation increases counterclockwise
      (or clockwise in image coordinates where y-axis points down)
    """

    object_id: Optional[int]
    class_id: int
    confidence: float
    oriented_bbox: np.ndarray  # 4 corner points for OBB
    center: Tuple[float, float]
    angle: float  # Continuous semantic angle
    source_engine: str  # 'yolo', 'optical_flow', or 'aruco'
    width: Optional[float] = None  # Semantic long axis
    height: Optional[float] = None  # Semantic short axis


@dataclass
class OpticalFlowParams:
    """Parameters for optical flow detection.

    Attributes:
        motion_threshold: Threshold for motion magnitude detection.
        min_area: Minimum contour area in pixels at full resolution (scaled with processing_scale²).
        max_area: Maximum contour area in pixels at full resolution (scaled with processing_scale², 0 to disable).
        algorithm: Optical flow algorithm - "dis", "farneback", or "lucas_kanade".
        temporal_smoothing: Weight for exponential moving average smoothing (0=no smoothing, 1=full).
        pyramid_levels: Number of pyramid levels for Farneback algorithm.
        window_size: Averaging window size for Farneback algorithm.
        iterations: Iterations per pyramid level for Farneback algorithm.
        median_filter_size: Median filter kernel size (0 to disable).
        debug_visualization: Enable caching of intermediate data for visualization.
        processing_scale: Scale factor for frame downscaling before flow computation (0.25-1.0).
        track_max_age: Maximum frames a track can be unmatched before expiring.
        track_min_iou: Minimum IoU overlap for track association (prevents merging nearby vehicles).
        mask_mode: How to combine bg subtraction and optical flow masks ("or", "and", "flow_only", "bg_only").
        dilate_kernel_size: Dilation kernel size for motion mask (0 to disable).
        morph_close_size: MORPH_CLOSE kernel size (0 to disable).
        morph_open_size: MORPH_OPEN kernel size (0 to disable).
        exclusion_mask: Path to mask image for excluding regions from detection (black areas = excluded).
    """

    motion_threshold: float = 1.0  # Increased from 0.5 for drone footage
    track_max_age: int = 10  # Frames before track expires (prevents ID reuse)
    track_min_iou: float = 0.1  # Minimum IoU for track association (prevents track merging)
    min_area: int = 2000  # For full resolution, scaled down with processing_scale²
    max_area: int = 30000  # For full resolution, scaled down with processing_scale² (0 to disable)
    algorithm: str = "dis"
    temporal_smoothing: float = 0.3
    pyramid_levels: int = 7
    window_size: int = 25
    iterations: int = 5
    median_filter_size: int = 5
    debug_visualization: bool = False
    processing_scale: float = 0.5  # Downscale factor for flow computation (0.25 = 1/4 res)
    # Mask combination and morphological tuning parameters
    mask_mode: str = "flow_only"  # "or", "and", "flow_only", "bg_only"
    dilate_kernel_size: int = 0  # Dilation for motion mask (0 to disable)
    morph_close_size: int = 3  # MORPH_CLOSE kernel size (0 to disable)
    morph_open_size: int = 5  # MORPH_OPEN kernel size (0 to disable)
    exclusion_mask: Optional[str] = None  # Path to mask image for excluding regions from detection


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

        # Ontology URI for OpenLabel output (explicit or extracted from file)
        self.ontology_uri = self._resolve_ontology_uri(args)

        # ArUco detection configuration
        self.aruco_csv_path = args.aruco_csv if hasattr(args, "aruco_csv") else None
        self.aruco_dict = (
            args.aruco_dict if hasattr(args, "aruco_dict") else "DICT_4X4_50"
        )
        self.use_aruco = self.aruco_csv_path is not None

        # Visual markers configuration
        self.visual_markers_csv_path = (
            args.visual_markers if hasattr(args, "visual_markers") else None
        )

        # Load class map from ontology if provided, otherwise use default
        # Pass verbose flag for debug logging
        verbose = args.verbose if hasattr(args, "verbose") else False
        self.class_map = self._load_class_map(verbose)

        # Get ArUco class ID from ontology (if ArUco detection enabled)
        self.aruco_class_id = None
        if self.use_aruco:
            self.aruco_class_id = self._get_aruco_class_id()

        # Detection method configuration
        self.use_yolo = args.detection_method in ["yolo", "both"]
        self.use_optical_flow = args.detection_method in ["optical_flow", "both"]

        # Optical flow parameters
        self.optical_flow_params = OpticalFlowParams(
            motion_threshold=args.motion_threshold,
            min_area=args.min_object_area,
            max_area=getattr(args, "max_object_area", 30000),
            algorithm=getattr(args, "flow_algorithm", "dis"),
            temporal_smoothing=getattr(args, "flow_temporal_smoothing", 0.3),
            pyramid_levels=getattr(args, "flow_pyramid_levels", 7),
            window_size=getattr(args, "flow_window_size", 25),
            iterations=getattr(args, "flow_iterations", 5),
            median_filter_size=getattr(args, "flow_median_filter", 5),
            debug_visualization=getattr(args, "debug_flow", False),
            processing_scale=getattr(args, "flow_scale", 0.5),
            track_max_age=getattr(args, "track_max_age", 10),
            mask_mode=getattr(args, "flow_mask_mode", "or"),
            dilate_kernel_size=getattr(args, "flow_dilate_size", 5),
            morph_close_size=getattr(args, "flow_morph_close", 5),
            morph_open_size=getattr(args, "flow_morph_open", 5),
            exclusion_mask=getattr(args, "exclusion_mask", None),
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
        self.min_total_movement = args.min_total_movement
        self.max_rotation_change = args.max_rotation_change
        self.edge_distance = args.edge_distance
        self.static_threshold = args.static_threshold
        self.static_mark = args.static_mark

        # Logging configuration
        self.verbose = args.verbose if hasattr(args, "verbose") else False

        self.validate_config()

    def validate_config(self) -> None:
        """Validate configuration parameters."""
        # Check required files
        required_files = [self.video_path, self.schema_path]
        if self.use_yolo:
            required_files.append(self.weights_path)
        if self.use_aruco:
            required_files.append(self.aruco_csv_path)
        if self.visual_markers_csv_path:
            required_files.append(self.visual_markers_csv_path)

        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")

        if not any([self.use_yolo, self.use_optical_flow, self.use_aruco]):
            raise ValueError("At least one detection method must be enabled")

        # Validate IoU threshold
        if not (0.0 <= self.iou_threshold <= 1.0):
            raise ValueError(
                f"IoU threshold must be between 0.0 and 1.0, got: {self.iou_threshold}"
            )

    def _resolve_ontology_uri(self, args: argparse.Namespace) -> Optional[str]:
        """Resolve ontology URI from explicit argument or by extracting from file.

        Args:
            args: Parsed command line arguments

        Returns:
            Ontology URI string, or None if not available.
        """
        logger = logging.getLogger(__name__)

        # Use explicit URI if provided
        if hasattr(args, "ontology_uri") and args.ontology_uri:
            logger.info(f"Ontology URI: {args.ontology_uri} (from --ontology-uri)")
            return args.ontology_uri

        # Otherwise extract from ontology file
        if self.ontology_path and os.path.exists(self.ontology_path):
            uri = extract_namespace_uri(self.ontology_path)
            if uri:
                logger.info(
                    f"Ontology URI: {uri} (extracted from {self.ontology_path})"
                )
                return uri
            else:
                logger.warning(
                    f"Could not extract namespace URI from {self.ontology_path}"
                )

        return None

    def _load_class_map(self, verbose: bool = False) -> Dict[int, str]:
        """Load class map from ontology file or use default.

        Args:
            verbose: Enable verbose logging of class mappings

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
                logger.warning(
                    f"No classes with UIDs found in ontology: {self.ontology_path}"
                )
                logger.info("Class map source: DEFAULT_CLASS_MAP (fallback)")
                return Constants.DEFAULT_CLASS_MAP.copy()

            logger.info(
                f"Class map source: {self.ontology_path} ({len(class_map)} classes)"
            )

            # If verbose mode, log first 25 class mappings for debugging
            if verbose:
                logger.info("Class map (first 25 YOLO class IDs):")
                for class_id in sorted(class_map.keys())[:25]:
                    logger.info(f"  YOLO class {class_id:3d} → '{class_map[class_id]}'")

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
                logger.info(
                    f"Found ArUco class in ontology: ID={class_id}, Name={class_name}"
                )
                return class_id

        # MarkerAruco not found - log error and raise exception
        logger.error("MarkerAruco class not found in ontology")
        logger.error(
            "Please ensure the ontology contains a 'MarkerAruco' class with label 'Aruco'"
        )
        raise ValueError(
            "MarkerAruco class not found in ontology - ArUco detection cannot proceed"
        )
