#!/usr/bin/env python3
"""
run_markit

Advanced command-line tool for running multi-engine object detection (YOLO + Optical Flow)
with IoU-based conflict resolution and optional VLM scene analysis. Exports results in
OpenLabel JSON format with SAVANT ontology integration and optionally as annotated video.

Usage:
    run_markit --input INPUT_VIDEO --output_json OUTPUT_JSON [OPTIONS]

Required Arguments:
    --input              Path to input video file
    --output_json        Path to output OpenLabel JSON file

Optional Arguments:
    --weights            Path to YOLO weights file (.pt) - required if using YOLO detection (default: markit_yolo.pt)
    --schema             Path to OpenLabel JSON schema file (default: ../schema/savant_openlabel_subset.schema.json)
    --ontology           Path to SAVANT ontology file for class mapping (default: ../ontology/savant.ttl)
    --ontology-uri       Ontology URI for OpenLabel output (default: extracted from ontology file)
    --output_video       Path to output annotated video file (optional)
    --aruco-csv          Path to CSV file with ArUco marker GPS positions (enables ArUco detection)
    --provenance         Path to provenance chain file for W3C PROV-JSON tracking (created if not exists)

Detection Configuration:
    --detection-method   Detection method: yolo, optical_flow, or both (default: yolo)
    --motion-threshold   Optical flow motion threshold (default: 1.0)
    --min-object-area    Minimum object area at full resolution (default: 2000)
    --max-object-area    Maximum object area at full resolution (default: 30000, 0 to disable)
    --flow-scale         Scale factor for optical flow processing (default: 0.5)
    --flow-algorithm     Optical flow algorithm: dis, farneback, lucas_kanade (default: dis)
    --flow-temporal-smoothing  Temporal smoothing factor (0-1, default: 0.3)
    --flow-pyramid-levels      Pyramid levels for Farneback (default: 7)
    --flow-window-size         Window size for Farneback (default: 25)
    --flow-iterations          Iterations for Farneback (default: 5)
    --flow-median-filter       Median filter size for noise reduction (default: 5, 0 to disable)
    --debug-flow         Enable optical flow visualization in output video (magnitude heatmap)
    --aruco-dict         ArUco dictionary type (default: DICT_4X4_50)

Conflict Resolution:
    --iou-threshold      IoU threshold for conflict resolution when using both engines (default: 0.3)
    --verbose-conflicts  Enable verbose conflict resolution logging
    --disable-conflict-resolution  Disable conflict resolution (keep all detections)

Postprocessing (Housekeeping):
    --housekeeping       Enable postprocessing passes (gap detection, filling, duplicate removal, etc.)
    --duplicate-avg-iou  Average IOU threshold for duplicate detection (default: 0.7)
    --duplicate-min-iou  Minimum IOU threshold for duplicate detection (default: 0.3)
    --rotation-threshold Rotation angle threshold in radians for adjustment (default: 0.1)
    --min-movement-pixels Minimum movement in pixels for rotation calculation (default: 5.0)
    --temporal-smoothing Temporal smoothing factor for rotation, 0-1 (default: 0.3)
    --min-total-movement Minimum cumulative movement to trust direction (default: 30.0)
    --max-rotation-change Maximum rotation change per frame in radians (default: 0.524 ≈ 30°)
    --edge-distance      Distance in pixels from frame edge for sudden appear/disappear detection (default: 200)
    --static-threshold   Movement threshold in pixels for static object removal (default: 20, negative disables)
    --static-mark        Mark static objects instead of removing them (adds "staticdynamic" annotation)

VLM Scene Analysis:
    --vlm                Enable VLM-based scene analysis for scenario tagging
    --vlm-model          VLM model name on the vLLM server (default: llama-3.2-11b-vision-instruct)
    --vlm-url            vLLM API base URL (default: http://localhost:8000)
    --vlm-api-key        API key for vLLM server (if required)
    --vlm-sampling       Frame sampling strategy: uniform, scene_change, keyframes (default: uniform)
    --vlm-interval       Frame interval for uniform sampling (default: 30)
    --vlm-max-frames     Maximum frames to analyze with VLM (default: 20)
    --vlm-timeout        VLM request timeout in seconds (default: 120)
    --vlm-prompts        Path to custom prompts JSON file
    --vlm-max-resolution Max frame height in pixels (e.g., 1080). Reduces VRAM usage.
    --vlm-delay          Delay between VLM requests in seconds (default: 0)
    --vlm-rationale      Request rationale explanations for weather fields (increases token usage)

Logging and Debug:
    --verbose            Enable verbose output with detailed angle and detection logging

Features:
    - YOLO OBB (Oriented Bounding Box) detection with tracking
    - Background subtraction + optical flow detection
    - IoU-based conflict resolution with YOLO precedence
    - VLM scene analysis for automatic scenario tagging (BSI PAS-1883 ODD taxonomy)
    - OpenLabel JSON export with SAVANT ontology integration
    - Dynamic class mapping from ontology (41 classes)
    - Configurable postprocessing pipeline for data quality improvement
    - Optional W3C PROV-JSON provenance tracking via dataprov
"""

import argparse
import logging
import sys

import cv2
import numpy as np
from ultralytics import __version__ as ultralytics_version

# Import from markitlib package
from markit.markitlib import MarkitConfig, __version__
from savant_common.resources import get_ontology_path, get_schema_path, get_weights_path
from markit.markitlib.processing import VideoProcessor
from markit.markitlib.openlabel import OpenLabelHandler
from markit.markitlib.outputvideo import render_output_video
from markit.markitlib.postprocessing import (
    PostprocessingPipeline,
    GapDetectionPass,
    GapFillingPass,
    DuplicateRemovalPass,
    FirstDetectionRefinementPass,
    BboxSmoothingPass,
    SizeOutlierFilterPass,
    SizeStepDetectionPass,
    Rotation90JumpFixPass,
    RotationAdjustmentPass,
    SuddenPass,
    FrameIntervalPass,
    StaticObjectRemovalPass,
    AngleNormalizationPass,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Advanced markit tool with multi-engine detection and IoU-based conflict resolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # YOLO only (uses default weights markit_yolo.pt, schema, and ontology)
  python markit.py --input video.mp4 --output_json output.json

  # YOLO with custom weights file
  python markit.py --weights model.pt --input video.mp4 --output_json output.json

  # With custom schema and ontology files
  python markit.py --weights model.pt --input video.mp4 --output_json output.json --schema custom.schema.json --ontology custom_ontology.ttl

  # Optical flow only
  python markit.py --detection-method optical_flow --input video.mp4 --output_json output.json

  # Both engines with default IoU threshold (0.3)
  python markit.py --detection-method both --weights model.pt --input video.mp4 --output_json output.json

  # Both engines with custom IoU threshold
  python markit.py --detection-method both --weights model.pt --input video.mp4 --output_json output.json --iou-threshold 0.5

  # Both engines without conflict resolution
  python markit.py --detection-method both --weights model.pt --input video.mp4 --output_json output.json --disable-conflict-resolution
        """,
    )

    parser.add_argument(
        "--version", action="version", version=f"SAVANT markit v{__version__}"
    )

    # Required arguments
    required = parser.add_argument_group("Required Arguments")
    required.add_argument("--input", required=True, help="Path to input video file")
    required.add_argument(
        "--output_json", required=True, help="Path to output OpenLabel JSON file"
    )

    # Optional arguments (paths/files)
    optional = parser.add_argument_group("Optional Arguments")
    optional.add_argument(
        "--weights",
        default=None,
        help="Path to YOLO weights file (.pt) (auto-downloads if not found)",
    )
    optional.add_argument(
        "--schema",
        default=None,
        help="Path to OpenLabel JSON schema file (uses package default if not specified)",
    )
    optional.add_argument(
        "--ontology",
        default=None,
        help="Path to SAVANT ontology file for class mapping (uses package default if not specified)",
    )
    optional.add_argument(
        "--ontology-uri",
        dest="ontology_uri",
        help="Ontology URI for OpenLabel output (default: extracted from ontology file)",
    )
    optional.add_argument("--output_video", help="Path to output annotated video file")
    optional.add_argument(
        "--aruco-csv",
        dest="aruco_csv",
        help="Path to CSV file with ArUco marker GPS positions (enables ArUco detection)",
    )
    optional.add_argument(
        "--visual-markers",
        dest="visual_markers",
        help="Path to CSV file with visual marker GPS positions (same format as ArUco CSV)",
    )
    optional.add_argument(
        "--provenance",
        help="Path to provenance chain file (will be created if not exists)",
    )

    # Detection configuration
    detection = parser.add_argument_group("Detection Configuration")
    detection.add_argument(
        "--detection-method",
        choices=["yolo", "optical_flow", "both"],
        default="yolo",
        help="Detection method(s) to use (default: yolo)",
    )
    detection.add_argument(
        "--motion-threshold",
        type=float,
        default=1.0,
        help="Optical flow motion threshold (default: 1.0)",
    )
    detection.add_argument(
        "--min-object-area",
        type=int,
        default=2000,
        help="Minimum object area in pixels at full resolution, scaled with flow-scale² (default: 2000)",
    )
    detection.add_argument(
        "--max-object-area",
        type=int,
        default=30000,
        help="Maximum object area in pixels at full resolution, scaled with flow-scale² (0 to disable, default: 30000)",
    )
    detection.add_argument(
        "--track-max-age",
        type=int,
        default=10,
        help="Maximum frames a track can be unmatched before expiring (default: 10)",
    )
    detection.add_argument(
        "--flow-algorithm",
        choices=["dis", "farneback", "lucas_kanade"],
        default="dis",
        help="Optical flow algorithm: dis (faster, recommended), farneback, lucas_kanade (default: dis)",
    )
    detection.add_argument(
        "--flow-temporal-smoothing",
        type=float,
        default=0.3,
        help="Temporal smoothing for flow (0=no smoothing, 1=full smoothing, default: 0.3)",
    )
    detection.add_argument(
        "--flow-pyramid-levels",
        type=int,
        default=7,
        help="Pyramid levels for Farneback algorithm (default: 7)",
    )
    detection.add_argument(
        "--flow-window-size",
        type=int,
        default=25,
        help="Window size for Farneback algorithm (default: 25)",
    )
    detection.add_argument(
        "--flow-iterations",
        type=int,
        default=5,
        help="Iterations per pyramid level for Farneback (default: 5)",
    )
    detection.add_argument(
        "--flow-median-filter",
        type=int,
        default=5,
        help="Median filter size for flow noise reduction (0 to disable, default: 5)",
    )
    detection.add_argument(
        "--debug-flow",
        action="store_true",
        help="Enable optical flow visualization in output video (magnitude heatmap overlay)",
    )
    detection.add_argument(
        "--flow-scale",
        type=float,
        default=0.5,
        help="Scale factor for optical flow processing (0.25-1.0, default: 0.5). Lower = faster but less precise.",
    )
    detection.add_argument(
        "--aruco-dict",
        dest="aruco_dict",
        default="DICT_4X4_50",
        choices=[
            "DICT_4X4_50",
            "DICT_4X4_100",
            "DICT_4X4_250",
            "DICT_4X4_1000",
            "DICT_5X5_50",
            "DICT_5X5_100",
            "DICT_5X5_250",
            "DICT_5X5_1000",
            "DICT_6X6_50",
            "DICT_6X6_100",
            "DICT_6X6_250",
            "DICT_6X6_1000",
            "DICT_7X7_50",
            "DICT_7X7_100",
            "DICT_7X7_250",
            "DICT_7X7_1000",
            "DICT_ARUCO_ORIGINAL",
        ],
        help="ArUco dictionary type (default: DICT_4X4_50)",
    )
    detection.add_argument(
        "--flow-mask-mode",
        dest="flow_mask_mode",
        choices=["or", "and", "flow_only", "bg_only"],
        default="flow_only",
        help="Mask combination mode: 'or' (union), 'and' (intersection), 'flow_only', 'bg_only' (default: flow_only)",
    )
    detection.add_argument(
        "--flow-dilate-size",
        dest="flow_dilate_size",
        type=int,
        default=0,
        help="Dilation kernel size for motion mask, 0 to disable (default: 0)",
    )
    detection.add_argument(
        "--flow-morph-close",
        dest="flow_morph_close",
        type=int,
        default=3,
        help="MORPH_CLOSE kernel size, 0 to disable (default: 3)",
    )
    detection.add_argument(
        "--flow-morph-open",
        dest="flow_morph_open",
        type=int,
        default=5,
        help="MORPH_OPEN kernel size, 0 to disable (default: 5)",
    )

    # Conflict resolution
    conflict = parser.add_argument_group("Conflict Resolution")
    conflict.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="IoU threshold for conflict resolution (default: 0.3)",
    )
    conflict.add_argument(
        "--verbose-conflicts",
        action="store_true",
        help="Enable verbose conflict resolution logging",
    )
    conflict.add_argument(
        "--disable-conflict-resolution",
        action="store_true",
        help="Disable conflict resolution (keep all detections)",
    )

    # Postprocessing (Housekeeping)
    postproc = parser.add_argument_group("Postprocessing (Housekeeping)")
    postproc.add_argument(
        "--housekeeping",
        action="store_true",
        help="Enable postprocessing passes (gap detection and filling)",
    )
    postproc.add_argument(
        "--duplicate-avg-iou",
        type=float,
        default=0.5,
        help="Average IOU threshold for duplicate detection (default: 0.5)",
    )
    postproc.add_argument(
        "--duplicate-min-iou",
        type=float,
        default=0.3,
        help="Minimum IOU threshold for duplicate detection (default: 0.3)",
    )
    postproc.add_argument(
        "--rotation-threshold",
        type=float,
        default=0.1,
        help="Rotation angle threshold in radians for adjustment (default: 0.1)",
    )
    postproc.add_argument(
        "--min-movement-pixels",
        type=float,
        default=5.0,
        help="Minimum movement in pixels for rotation calculation (default: 5.0)",
    )
    postproc.add_argument(
        "--temporal-smoothing",
        type=float,
        default=0.3,
        help="Temporal smoothing factor for rotation (0-1, higher = more smoothing, default: 0.3)",
    )
    postproc.add_argument(
        "--min-total-movement",
        type=float,
        default=30.0,
        help="Minimum cumulative movement in pixels to trust direction (default: 30.0)",
    )
    postproc.add_argument(
        "--max-rotation-change",
        type=float,
        default=0.524,
        help="Maximum rotation change per frame in radians (default: 0.524 ≈ 30°)",
    )
    postproc.add_argument(
        "--edge-distance",
        type=int,
        default=200,
        help="Distance in pixels from frame edge for sudden appear/disappear detection (default: 200)",
    )
    postproc.add_argument(
        "--static-threshold",
        type=int,
        default=20,
        help="Movement threshold in pixels for static object removal (default: 20, negative value disables)",
    )
    postproc.add_argument(
        "--static-mark",
        action="store_true",
        help='Mark static objects instead of removing them (adds "staticdynamic" annotation)',
    )

    # Logging and debug
    logging_group = parser.add_argument_group("Logging and Debug")
    logging_group.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with detailed angle and detection logging",
    )

    # VLM Analysis Configuration
    vlm_group = parser.add_argument_group('VLM Scene Analysis')
    vlm_group.add_argument('--vlm', action='store_true',
                           help='Enable VLM-based scene analysis for scenario tagging')
    vlm_group.add_argument('--vlm-model', default='llama-3.2-11b-vision-instruct',
                           help='VLM model name on the vLLM server (default: llama-3.2-11b-vision-instruct)')
    vlm_group.add_argument('--vlm-url', default='http://localhost:8000',
                           help='vLLM API base URL (default: http://localhost:8000)')
    vlm_group.add_argument('--vlm-api-key',
                           help='API key for vLLM server (if required)')
    vlm_group.add_argument('--vlm-sampling', choices=['uniform', 'scene_change', 'keyframes'],
                           default='uniform',
                           help='Frame sampling strategy for VLM analysis (default: uniform)')
    vlm_group.add_argument('--vlm-interval', type=int, default=30,
                           help='Frame interval for uniform sampling (default: 30)')
    vlm_group.add_argument('--vlm-max-frames', type=int, default=20,
                           help='Maximum frames to analyze with VLM (default: 20)')
    vlm_group.add_argument('--vlm-timeout', type=int, default=120,
                           help='VLM request timeout in seconds (default: 120)')
    vlm_group.add_argument('--vlm-prompts',
                           help='Path to custom prompts JSON file (optional)')
    vlm_group.add_argument('--vlm-max-resolution', type=int, default=None,
                           help='Max frame height in pixels for VLM (e.g., 1080). Reduces VRAM usage.')
    vlm_group.add_argument('--vlm-delay', type=float, default=0.0,
                           help='Delay between VLM requests in seconds (default: 0)')
    vlm_group.add_argument('--vlm-rationale', action='store_true',
                           help='Request rationale explanations for weather fields (increases token usage)')

    return parser.parse_args()


def build_arguments_string(args: argparse.Namespace) -> str:
    """Build a string representation of relevant CLI arguments for provenance.

    Args:
        args: Parsed command line arguments

    Returns:
        Space-separated string of CLI arguments used
    """
    parts = [
        f"--input {args.input}",
        f"--output_json {args.output_json}",
        f"--detection-method {args.detection_method}",
        f"--schema {args.schema}",
        f"--ontology {args.ontology}",
    ]
    if args.detection_method in ["yolo", "both"]:
        parts.append(f"--weights {args.weights}")
    if args.housekeeping:
        parts.append("--housekeeping")
        parts.append(f"--duplicate-avg-iou {args.duplicate_avg_iou}")
        parts.append(f"--duplicate-min-iou {args.duplicate_min_iou}")
        parts.append(f"--rotation-threshold {args.rotation_threshold}")
        parts.append(f"--min-movement-pixels {args.min_movement_pixels}")
        parts.append(f"--temporal-smoothing {args.temporal_smoothing}")
        parts.append(f"--min-total-movement {args.min_total_movement}")
        parts.append(f"--max-rotation-change {args.max_rotation_change}")
        parts.append(f"--edge-distance {args.edge_distance}")
        parts.append(f"--static-threshold {args.static_threshold}")
        if args.static_mark:
            parts.append("--static-mark")
    if args.output_video:
        parts.append(f"--output_video {args.output_video}")
    if args.aruco_csv:
        parts.append(f"--aruco-csv {args.aruco_csv}")
        parts.append(f"--aruco-dict {args.aruco_dict}")
    if args.detection_method in ["optical_flow", "both"]:
        parts.append(f"--motion-threshold {args.motion_threshold}")
        parts.append(f"--min-object-area {args.min_object_area}")
        parts.append(f"--max-object-area {args.max_object_area}")
        parts.append(f"--flow-algorithm {args.flow_algorithm}")
        parts.append(f"--flow-temporal-smoothing {args.flow_temporal_smoothing}")
        parts.append(f"--flow-scale {args.flow_scale}")
        parts.append(f"--flow-mask-mode {args.flow_mask_mode}")
        parts.append(f"--flow-dilate-size {args.flow_dilate_size}")
        parts.append(f"--flow-morph-close {args.flow_morph_close}")
        parts.append(f"--flow-morph-open {args.flow_morph_open}")
        if args.flow_algorithm == "farneback":
            parts.append(f"--flow-pyramid-levels {args.flow_pyramid_levels}")
            parts.append(f"--flow-window-size {args.flow_window_size}")
            parts.append(f"--flow-iterations {args.flow_iterations}")
        if args.flow_median_filter > 0:
            parts.append(f"--flow-median-filter {args.flow_median_filter}")
    if args.detection_method == "both" and not args.disable_conflict_resolution:
        parts.append(f"--iou-threshold {args.iou_threshold}")
    return " ".join(parts)


def process_video(
    video_processor: VideoProcessor,
    openlabel_handler: OpenLabelHandler,
    config: MarkitConfig,
) -> None:
    """Main video processing loop with multi-engine support.

    Args:
        video_processor: Video processor instance
        openlabel_handler: OpenLabel handler instance
        config: Application configuration
    """
    frame_idx = 0
    total_frames = 0

    logger.info("Starting multi-engine video processing...")

    try:
        while True:
            success, frame = video_processor.read_frame()
            if not success:
                break

            # Process frame with all configured engines
            detection_results = video_processor.process_frame(frame, frame_idx=frame_idx)

            # Add to OpenLabel structure
            openlabel_handler.add_frame_objects(
                frame_idx, detection_results, config.class_map
            )

            frame_idx += 1
            total_frames += 1

            # Log progress periodically
            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx} frames...")

    except Exception as e:
        logger.error(f"Error during video processing: {e}")
        raise

    # Log final statistics
    stats = video_processor.get_detection_statistics()
    logger.info(f"Video processing completed. Total frames processed: {total_frames}")
    logger.info(f"Detection statistics: {stats}")


def cleanup(
    video_processor: VideoProcessor,
    openlabel_handler: OpenLabelHandler,
    config: MarkitConfig,
) -> None:
    """Cleanup and finalization.

    Args:
        video_processor: Video processor instance
        openlabel_handler: OpenLabel handler instance
        config: Application configuration
    """
    try:
        # Save OpenLabel data
        openlabel_handler.save_to_file(config.output_json_path)

        # Clean up video resources
        video_processor.cleanup()

        logger.info("Cleanup completed successfully")

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise


def main():
    """Main function to orchestrate the multi-engine video processing workflow."""
    try:
        # Parse arguments
        args = parse_arguments()

        # Capture start time for provenance tracking
        start_time = None
        if args.provenance:
            from datetime import datetime, timezone

            start_time = datetime.now(timezone.utc)

        # Determine which engines will be used (before config to log first)
        engines = []
        if args.detection_method in ["yolo", "both"]:
            engines.append("YOLO")
        if args.detection_method in ["optical_flow", "both"]:
            engines.append("OpticalFlow")
        if args.aruco_csv:
            engines.append("ArUco")

        # Log startup message first
        logger.info(
            f"SAVANT markit v{__version__} starting with engines: {', '.join(engines)}"
        )
        logger.info(
            f"Library versions: OpenCV {cv2.__version__}, NumPy {np.__version__}, Ultralytics {ultralytics_version}"
        )

        # Resolve resource paths (package data or fallbacks)
        args.ontology = args.ontology or get_ontology_path()
        args.schema = args.schema or get_schema_path()
        if args.detection_method in ["yolo", "both"]:
            args.weights = args.weights or get_weights_path()

        # Create configuration
        config = MarkitConfig(args)

        if config.enable_conflict_resolution and len(engines) > 1:
            logger.info(
                f"Conflict resolution enabled with IoU threshold: {config.iou_threshold:.2f}"
            )

        # Initialize components
        video_processor = VideoProcessor(config)
        openlabel_handler = OpenLabelHandler(config.schema_path, verbose=config.verbose)

        # Initialize video processing
        video_processor.initialize()
        openlabel_handler.add_metadata(config.video_path)
        openlabel_handler.set_ontology(config.ontology_uri)

        # Pre-populate ArUco markers from GPS data (if ArUco detection enabled)
        aruco_gps = video_processor.get_aruco_gps_data()
        if aruco_gps:
            gps_data, csv_name = aruco_gps
            id_mapping = video_processor.get_aruco_id_mapping()
            openlabel_handler.add_aruco_objects(gps_data, csv_name, id_mapping)

        # Pre-populate visual markers from GPS data (if provided)
        visual_marker_result = video_processor.get_visual_marker_data()
        if visual_marker_result:
            visual_marker_data, vm_id_mapping = visual_marker_result
            openlabel_handler.add_visual_marker_objects(
                visual_marker_data.gps_data,
                visual_marker_data.marker_names,
                vm_id_mapping,
                visual_marker_data.csv_name,
            )

        # Process video
        process_video(video_processor, openlabel_handler, config)

        # Postprocessing pipeline (only if housekeeping enabled)
        if config.enable_housekeeping:
            logger.info("Starting postprocessing...")
            postprocessing_pipeline = PostprocessingPipeline()
            postprocessing_pipeline.set_video_properties(
                video_processor.frame_width,
                video_processor.frame_height,
                video_processor.fps,
            )
            postprocessing_pipeline.set_ontology_path(config.ontology_path)

            # Pipeline order:
            # 1. Gap detection and filling
            postprocessing_pipeline.add_pass(GapDetectionPass())
            postprocessing_pipeline.add_pass(GapFillingPass())

            # 2. Duplicate removal
            postprocessing_pipeline.add_pass(
                DuplicateRemovalPass(
                    avg_iou_threshold=config.duplicate_avg_iou,
                    min_iou_threshold=config.duplicate_min_iou,
                )
            )

            # 3. Static object removal (if enabled)
            if config.static_threshold >= 0:
                postprocessing_pipeline.add_pass(
                    StaticObjectRemovalPass(
                        static_threshold=config.static_threshold,
                        mark_only=config.static_mark,
                    )
                )

            # 4. Refine initial detection angles using lookahead
            postprocessing_pipeline.add_pass(
                FirstDetectionRefinementPass(
                    lookahead_frames=5, min_movement_pixels=5.0
                )
            )

            # 5. Filter size outliers (motion streaks, sudden elongation)
            # Run BEFORE smoothing to detect raw spikes before EMA blends them
            postprocessing_pipeline.add_pass(SizeOutlierFilterPass())

            # 6. Size smoothing (position NOT smoothed - raw is acceptable)
            # Run after outlier filter so smoothing works on clean data
            postprocessing_pipeline.add_pass(BboxSmoothingPass())

            # 7. Fix 90° and 180° rotation jumps from minAreaRect ambiguity
            # Run BEFORE RotationAdjustmentPass so jumps are fixed before temporal smoothing
            postprocessing_pipeline.add_pass(Rotation90JumpFixPass())

            # 8. Adjust rotation based on movement direction
            postprocessing_pipeline.add_pass(
                RotationAdjustmentPass(
                    rotation_threshold=config.rotation_threshold,
                    min_movement_pixels=config.min_movement_pixels,
                    min_total_movement=config.min_total_movement,
                    temporal_smoothing=config.temporal_smoothing,
                    max_rotation_change=config.max_rotation_change,
                )
            )

            # 9. Detect sudden appear/disappear events
            postprocessing_pipeline.add_pass(
                SuddenPass(edge_distance=config.edge_distance)
            )

            # 10. Detect persistent size changes (step changes) for manual review
            postprocessing_pipeline.add_pass(SizeStepDetectionPass())

            # 11. Add frame intervals
            postprocessing_pipeline.add_pass(FrameIntervalPass())

            # 12. Normalize all angles to [0, 2π) for OpenLabel output
            postprocessing_pipeline.add_pass(AngleNormalizationPass())

            # Final pipeline order:
            # 1. GapDetection → 2. GapFilling → 3. DuplicateRemoval →
            # 4. StaticObjectRemoval → 5. FirstDetectionRefinement →
            # 6. SizeOutlierFilter → 7. BboxSmoothing → 8. Rotation90JumpFix →
            # 9. RotationAdjustment → 10. Sudden → 11. SizeStepDetection →
            # 12. FrameInterval → 13. AngleNormalization

            openlabel_handler.openlabel_data = postprocessing_pipeline.execute(
                openlabel_handler.openlabel_data
            )
        else:
            logger.info("Housekeeping disabled, skipping postprocessing")

        # VLM Scene Analysis (if enabled)
        if args.vlm:
            logger.info("Starting VLM scene analysis...")
            from markit.markitlib.vlm import VLMConfig, VLMProvider, SamplingStrategy, VLMAnalysisPass

            vlm_config = VLMConfig(
                enabled=True,
                provider=VLMProvider.VLLM,
                model_name=args.vlm_model,
                base_url=args.vlm_url,
                api_key=args.vlm_api_key,
                timeout=args.vlm_timeout,
                sampling_strategy=SamplingStrategy(args.vlm_sampling),
                sample_interval=args.vlm_interval,
                max_samples=args.vlm_max_frames,
                max_resolution=args.vlm_max_resolution,
                request_delay=args.vlm_delay,
                prompts_file=args.vlm_prompts,
                rationale_enabled=args.vlm_rationale,
            )

            vlm_pass = VLMAnalysisPass(vlm_config)
            vlm_pass.set_video_path(config.video_path)
            vlm_pass.set_video_properties(
                video_processor.frame_width,
                video_processor.frame_height,
                video_processor.fps
            )

            openlabel_handler.openlabel_data = vlm_pass.process(
                openlabel_handler.openlabel_data
            )

            vlm_stats = vlm_pass.get_statistics()
            logger.info(f"VLM analysis statistics: {vlm_stats}")

        # Render output video from postprocessed data (if requested)
        if config.output_video_path:
            render_output_video(
                config, openlabel_handler.openlabel_data, openlabel_handler.debug_data
            )

        # Cleanup and save results
        cleanup(video_processor, openlabel_handler, config)

        # Record provenance if enabled
        if args.provenance:
            from datetime import datetime, timezone
            from dataprov import ProvenanceChain

            end_time = datetime.now(timezone.utc)

            chain = ProvenanceChain.load_or_create(
                args.provenance,
                entity_id="savant_markit_output",
                initial_source=args.input,
                description="SAVANT markit video processing",
            )

            # Build arguments string
            arguments = build_arguments_string(args)

            # Collect all input files used
            inputs = [args.input]
            input_formats = ["MP4"]

            # Add schema and ontology (always used)
            inputs.append(args.schema)
            input_formats.append("JSON")
            inputs.append(args.ontology)
            input_formats.append("TTL")

            # Add weights if YOLO detection used
            if args.detection_method in ["yolo", "both"]:
                inputs.append(args.weights)
                input_formats.append("PT")

            # Add ArUco CSV if provided
            if args.aruco_csv:
                inputs.append(args.aruco_csv)
                input_formats.append("CSV")

            # Collect outputs (JSON always, video if specified)
            outputs = [args.output_json]
            output_formats = ["JSON"]
            if args.output_video:
                outputs.append(args.output_video)
                output_formats.append("MP4")

            chain.add(
                started_at=start_time.isoformat().replace("+00:00", "Z"),
                ended_at=end_time.isoformat().replace("+00:00", "Z"),
                tool_name="run_markit",
                tool_version=__version__,
                operation="object detection and tracking",
                inputs=inputs,
                input_formats=input_formats,
                outputs=outputs,
                output_formats=output_formats,
                arguments=arguments,
                capture_agent=True,
                agent_type="automated",
                capture_environment=True,
            )

            chain.save(args.provenance)
            logger.info(f"Provenance recorded to {args.provenance}")

        logger.info("Multi-engine video processing completed successfully")

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
