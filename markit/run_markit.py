#!/usr/bin/env python3
"""
run_markit

Advanced command-line tool for running multi-engine object detection (YOLO + Optical Flow)
with IoU-based conflict resolution. Exports results in OpenLabel JSON format with SAVANT
ontology integration and optionally as annotated video.

Usage:
    run_markit --input INPUT_VIDEO --output_json OUTPUT_JSON [OPTIONS]

Required Arguments:
    --input              Path to input video file
    --output_json        Path to output OpenLabel JSON file

Optional Arguments:
    --weights            Path to YOLO weights file (.pt) - required if using YOLO detection (default: markit_yolo.pt)
    --schema             Path to OpenLabel JSON schema file (default: ../schema/savant_openlabel_subset.schema.json)
    --ontology           Path to SAVANT ontology file for class mapping (default: ../ontology/savant_ontology_1.3.1.ttl)
    --ontology-uri       Ontology URI for OpenLabel output (default: extracted from ontology file)
    --output_video       Path to output annotated video file (optional)
    --aruco-csv          Path to CSV file with ArUco marker GPS positions (enables ArUco detection)
    --provenance         Path to provenance chain file for W3C PROV-JSON tracking (created if not exists)

Detection Configuration:
    --detection-method   Detection method: yolo, optical_flow, or both (default: yolo)
    --motion-threshold   Optical flow motion threshold (default: 0.5)
    --min-object-area    Minimum object area for optical flow detection (default: 200)
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
    --edge-distance      Distance in pixels from frame edge for sudden appear/disappear detection (default: 200)
    --static-threshold   Movement threshold in pixels for static object removal (default: 20, negative disables)
    --static-mark        Mark static objects instead of removing them (adds "staticdynamic" annotation)

Logging and Debug:
    --verbose            Enable verbose output with detailed angle and detection logging

Features:
    - YOLO OBB (Oriented Bounding Box) detection with tracking
    - Background subtraction + optical flow detection
    - IoU-based conflict resolution with YOLO precedence
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
from markit.markitlib.processing import VideoProcessor
from markit.markitlib.openlabel import OpenLabelHandler
from markit.markitlib.outputvideo import render_output_video
from markit.markitlib.postprocessing import (
    PostprocessingPipeline,
    GapDetectionPass,
    GapFillingPass,
    DuplicateRemovalPass,
    FirstDetectionRefinementPass,
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
        default="markit_yolo.pt",
        help="Path to YOLO weights file (.pt) (default: markit_yolo.pt)",
    )
    optional.add_argument(
        "--schema",
        default="../schema/savant_openlabel_subset.schema.json",
        help="Path to OpenLabel JSON schema file (default: ../schema/savant_openlabel_subset.schema.json)",
    )
    optional.add_argument(
        "--ontology",
        default="../ontology/savant.ttl",
        help="Path to SAVANT ontology file for class mapping (default: ../ontology/savant.ttl)",
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
        default=0.5,
        help="Optical flow motion threshold (default: 0.5)",
    )
    detection.add_argument(
        "--min-object-area",
        type=int,
        default=200,
        help="Minimum object area for optical flow detection (default: 200)",
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
        default=0.7,
        help="Average IOU threshold for duplicate detection (default: 0.7)",
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
            detection_results = video_processor.process_frame(frame)

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
            postprocessing_pipeline.add_pass(GapDetectionPass())
            postprocessing_pipeline.add_pass(GapFillingPass())
            postprocessing_pipeline.add_pass(
                DuplicateRemovalPass(
                    avg_iou_threshold=config.duplicate_avg_iou,
                    min_iou_threshold=config.duplicate_min_iou,
                )
            )
            if config.static_threshold >= 0:
                postprocessing_pipeline.add_pass(
                    StaticObjectRemovalPass(
                        static_threshold=config.static_threshold,
                        mark_only=config.static_mark,
                    )
                )
            # MANDATORY: Refine initial detection angles using lookahead
            postprocessing_pipeline.add_pass(
                FirstDetectionRefinementPass(
                    lookahead_frames=5, min_movement_pixels=5.0
                )
            )
            # OPTIONAL: Further refine rotation using movement direction
            postprocessing_pipeline.add_pass(
                RotationAdjustmentPass(
                    rotation_threshold=config.rotation_threshold,
                    min_movement_pixels=config.min_movement_pixels,
                    temporal_smoothing=config.temporal_smoothing,
                )
            )
            postprocessing_pipeline.add_pass(
                SuddenPass(edge_distance=config.edge_distance)
            )
            postprocessing_pipeline.add_pass(FrameIntervalPass())
            # MANDATORY FINAL PASS: Normalize all angles to [0, 2π) for OpenLabel output
            postprocessing_pipeline.add_pass(AngleNormalizationPass())

            openlabel_handler.openlabel_data = postprocessing_pipeline.execute(
                openlabel_handler.openlabel_data
            )
        else:
            logger.info("Housekeeping disabled, skipping postprocessing")

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
