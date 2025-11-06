#!/usr/bin/env python3
"""
markit.py

Advanced command-line tool for running multi-engine object detection (YOLO + Optical Flow)
with IoU-based conflict resolution. Exports results in OpenLabel JSON format with SAVANT
ontology integration and optionally as annotated video.

Usage:
    python markit.py --input INPUT_VIDEO --output_json OUTPUT_JSON [OPTIONS]

Required Arguments:
    --input              Path to input video file
    --output_json        Path to output OpenLabel JSON file

Optional Arguments:
    --weights            Path to YOLO weights file (.pt) - required if using YOLO detection
    --schema             Path to OpenLabel JSON schema file (default: savant_openlabel_subset.schema.json)
    --ontology           Path to SAVANT ontology file for class mapping (default: savant_ontology_1.2.0.ttl)
    --output_video       Path to output annotated video file (optional)
    --aruco-csv          Path to CSV file with ArUco marker GPS positions (enables ArUco detection)

Detection Configuration:
    --detection-method   Detection method: yolo, optical_flow, or both (default: yolo)
    --motion-threshold   Optical flow motion threshold (default: 0.5)
    --min-object-area    Minimum object area for optical flow detection (default: 200)

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
    --static-threshold   Movement threshold in pixels for static object removal (default: 5, negative disables)
    --static-mark        Mark static objects instead of removing them (adds "staticdynamic" annotation)

Features:
    - YOLO OBB (Oriented Bounding Box) detection with tracking
    - Background subtraction + optical flow detection
    - IoU-based conflict resolution with YOLO precedence
    - OpenLabel JSON export with SAVANT ontology integration
    - Dynamic class mapping from ontology (41 classes)
    - Configurable postprocessing pipeline for data quality improvement
"""

import argparse
import logging
import sys

import cv2
import numpy as np
from ultralytics import __version__ as ultralytics_version

# Import from markitlib package
from markitlib import MarkitConfig, __version__
from markitlib.processing import VideoProcessor, FrameAnnotator
from markitlib.openlabel import OpenLabelHandler
from markitlib.outputvideo import render_output_video
from markitlib.postprocessing import (
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='Advanced markit tool with multi-engine detection and IoU-based conflict resolution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # YOLO only (uses default schema and ontology)
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
        """
    )

    # Required arguments
    parser.add_argument('--weights', help='Path to YOLO weights file (.pt)')
    parser.add_argument('--input', required=True, help='Path to input video file')
    parser.add_argument('--output_json', required=True, help='Path to output OpenLabel JSON file')
    parser.add_argument('--schema', default='savant_openlabel_subset.schema.json',
                       help='Path to OpenLabel JSON schema file (default: savant_openlabel_subset.schema.json)')
    parser.add_argument('--output_video', help='Path to output annotated video file (optional)')
    parser.add_argument('--ontology', default='savant_ontology_1.2.0.ttl',
                       help='Path to SAVANT ontology file for class mapping (default: savant_ontology_1.2.0.ttl)')
    parser.add_argument('--aruco-csv', dest='aruco_csv', help='Path to CSV file with ArUco marker GPS positions (enables ArUco detection)')

    # Detection method selection
    parser.add_argument('--detection-method',
                       choices=['yolo', 'optical_flow', 'both'],
                       default='yolo',
                       help='Detection method(s) to use (default: yolo)')

    # Optical flow parameters
    parser.add_argument('--motion-threshold', type=float, default=0.5,
                       help='Optical flow motion threshold (default: 0.5)')
    parser.add_argument('--min-object-area', type=int, default=200,
                       help='Minimum object area for optical flow detection (default: 200)')

    # IoU-based conflict resolution
    parser.add_argument('--iou-threshold', type=float, default=0.3,
                       help='IoU threshold for conflict resolution (default: 0.3)')
    parser.add_argument('--verbose-conflicts', action='store_true',
                       help='Enable verbose conflict resolution logging')
    parser.add_argument('--disable-conflict-resolution', action='store_true',
                       help='Disable conflict resolution (keep all detections)')

    # Postprocessing
    parser.add_argument('--housekeeping', action='store_true',
                       help='Enable postprocessing passes (gap detection and filling)')
    parser.add_argument('--duplicate-avg-iou', type=float, default=0.7,
                       help='Average IOU threshold for duplicate detection (default: 0.7)')
    parser.add_argument('--duplicate-min-iou', type=float, default=0.3,
                       help='Minimum IOU threshold for duplicate detection (default: 0.3)')
    parser.add_argument('--rotation-threshold', type=float, default=0.1,
                       help='Rotation angle threshold in radians for adjustment (default: 0.1)')
    parser.add_argument('--min-movement-pixels', type=float, default=5.0,
                       help='Minimum movement in pixels for rotation calculation (default: 5.0)')
    parser.add_argument('--temporal-smoothing', type=float, default=0.3,
                       help='Temporal smoothing factor for rotation (0-1, higher = more smoothing, default: 0.3)')
    parser.add_argument('--edge-distance', type=int, default=200,
                       help='Distance in pixels from frame edge for sudden appear/disappear detection (default: 200)')
    parser.add_argument('--static-threshold', type=int, default=5,
                       help='Movement threshold in pixels for static object removal (default: 5, negative value disables this pass)')
    parser.add_argument('--static-mark', action='store_true',
                       help='Mark static objects instead of removing them (adds "staticdynamic" annotation)')

    # Logging and debug
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output with detailed angle and detection logging')

    return parser.parse_args()


def process_video(video_processor: VideoProcessor, openlabel_handler: OpenLabelHandler,
                 config: MarkitConfig) -> None:
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
            openlabel_handler.add_frame_objects(frame_idx, detection_results, config.class_map)

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


def cleanup(video_processor: VideoProcessor, openlabel_handler: OpenLabelHandler,
           config: MarkitConfig) -> None:
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

        # Create configuration
        config = MarkitConfig(args)

        # Log configuration
        engines = []
        if config.use_yolo:
            engines.append("YOLO")
        if config.use_optical_flow:
            engines.append("OpticalFlow")
        if config.use_aruco:
            engines.append("ArUco")

        logger.info(f"Markit v{__version__} starting with engines: {', '.join(engines)}")

        # Log library versions
        logger.info(f"Library versions: OpenCV {cv2.__version__}, NumPy {np.__version__}, Ultralytics {ultralytics_version}")

        if config.enable_conflict_resolution and len(engines) > 1:
            logger.info(f"Conflict resolution enabled with IoU threshold: {config.iou_threshold:.2f}")

        # Initialize components
        video_processor = VideoProcessor(config)
        openlabel_handler = OpenLabelHandler(config.schema_path, verbose=config.verbose)

        # Initialize video processing
        video_processor.initialize()
        openlabel_handler.add_metadata(config.video_path)

        # Process video
        process_video(video_processor, openlabel_handler, config)

        # Postprocessing pipeline (only if housekeeping enabled)
        if config.enable_housekeeping:
            logger.info("Starting postprocessing...")
            postprocessing_pipeline = PostprocessingPipeline()
            postprocessing_pipeline.set_video_properties(
                video_processor.frame_width,
                video_processor.frame_height,
                video_processor.fps
            )
            postprocessing_pipeline.set_ontology_path(config.ontology_path)
            postprocessing_pipeline.add_pass(GapDetectionPass())
            postprocessing_pipeline.add_pass(GapFillingPass())
            postprocessing_pipeline.add_pass(
                DuplicateRemovalPass(
                    avg_iou_threshold=config.duplicate_avg_iou,
                    min_iou_threshold=config.duplicate_min_iou
                )
            )
            if config.static_threshold >= 0:
                postprocessing_pipeline.add_pass(
                    StaticObjectRemovalPass(
                        static_threshold=config.static_threshold,
                        mark_only=config.static_mark
                    )
                )
            # MANDATORY: Refine initial detection angles using lookahead
            postprocessing_pipeline.add_pass(
                FirstDetectionRefinementPass(
                    lookahead_frames=5,
                    min_movement_pixels=5.0
                )
            )
            # OPTIONAL: Further refine rotation using movement direction
            postprocessing_pipeline.add_pass(
                RotationAdjustmentPass(
                    rotation_threshold=config.rotation_threshold,
                    min_movement_pixels=config.min_movement_pixels,
                    temporal_smoothing=config.temporal_smoothing
                )
            )
            postprocessing_pipeline.add_pass(SuddenPass(edge_distance=config.edge_distance))
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
            render_output_video(config, openlabel_handler.openlabel_data, openlabel_handler.debug_data)

        # Cleanup and save results
        cleanup(video_processor, openlabel_handler, config)

        logger.info("Multi-engine video processing completed successfully")

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
