"""
markit.py

Refactored command-line tool for running YOLO OBB tracking on a video and exporting results in OpenLabel JSON format and optionally as an annotated video.

Usage:
    python markit.py --weights WEIGHTS_PATH --input INPUT_VIDEO --output_json OUTPUT_JSON --schema SCHEMA_JSON [--output_video OUTPUT_VIDEO]

Arguments:
    --weights      Path to YOLO weights file (.pt)
    --input        Path to input video file
    --output_json  Path to output OpenLabel JSON file
    --schema       Path to OpenLabel JSON schema file
    --output_video Path to output annotated video file (optional)

This script uses the Ultralytics YOLO model for oriented bounding box (OBB) tracking, annotates the video frames if requested, and saves results in OpenLabel format.
"""

__version__ = '1.0.0'

import argparse
import logging
import sys
import os
import json
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


class MarkitConfig:
    """Configuration class for Markit application."""
    
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
        
        self.validate_config()
    
    def validate_config(self) -> None:
        """Validate configuration parameters."""
        required_files = [self.weights_path, self.video_path, self.schema_path]
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")


class VideoProcessor:
    """Handles video input/output operations and YOLO model processing."""
    
    def __init__(self, config: MarkitConfig):
        """Initialize video processor.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.model = None
        self.cap = None
        self.out = None
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0.0
        
    def initialize(self) -> None:
        """Initialize YOLO model and video capture."""
        try:
            logger.info(f"Loading YOLO model from {self.config.weights_path}")
            self.model = YOLO(self.config.weights_path)
            
            logger.info(f"Opening video file: {self.config.video_path}")
            self.cap = cv2.VideoCapture(self.config.video_path)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video file: {self.config.video_path}")
                
            self._get_video_properties()
            
            if self.config.output_video_path:
                self._setup_video_writer()
                
        except Exception as e:
            logger.error(f"Failed to initialize video processor: {e}")
            raise
    
    def _get_video_properties(self) -> None:
        """Extract video properties from capture."""
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Video properties - Width: {self.frame_width}, Height: {self.frame_height}, FPS: {self.fps}")
    
    def _setup_video_writer(self) -> None:
        """Setup video writer for output."""
        try:
            fourcc = cv2.VideoWriter_fourcc(*Constants.MP4V_FOURCC)
            self.out = cv2.VideoWriter(
                self.config.output_video_path, 
                fourcc, 
                self.fps, 
                (self.frame_width, self.frame_height)
            )
            logger.info(f"Video writer initialized for output: {self.config.output_video_path}")
        except Exception as e:
            logger.error(f"Failed to setup video writer: {e}")
            raise
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame from video.
        
        Returns:
            Tuple of (success, frame)
        """
        if self.cap is None:
            return False, None
        return self.cap.read()
    
    def process_frame(self, frame: np.ndarray) -> List[Any]:
        """Process frame with YOLO model.
        
        Args:
            frame: Input frame
            
        Returns:
            YOLO tracking results
        """
        try:
            return self.model.track(frame, persist=True)
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return []
    
    def write_frame(self, frame: np.ndarray) -> None:
        """Write frame to output video.
        
        Args:
            frame: Frame to write
        """
        if self.out is not None:
            self.out.write(frame)
    
    def cleanup(self) -> None:
        """Clean up video resources."""
        if self.cap is not None:
            self.cap.release()
            logger.info("Video capture released")
            
        if self.out is not None:
            self.out.release()
            logger.info("Video writer released")
            
        cv2.destroyAllWindows()


class FrameAnnotator:
    """Handles frame annotation with bounding boxes and labels."""
    
    @staticmethod
    def annotate_frame(frame: np.ndarray, results: List[Any]) -> np.ndarray:
        """Annotate frame with bounding boxes and labels.
        
        Args:
            frame: Input frame
            results: YOLO tracking results
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for result in results:
            if result.obb.id is not None:
                obbs_xyxyxyxy = result.obb.xyxyxyxy.cpu()
                labels = [f"ID {id} Class {cls}" for id, cls in
                         zip(result.obb.id.int().cpu(), result.obb.cls.int().cpu())]
                
                for box, label in zip(obbs_xyxyxyxy, labels):
                    points = np.array(box, dtype=np.int32).reshape((4, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)
                    cv2.putText(annotated_frame, label, (points[0][0], points[0][1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return annotated_frame


class OpenLabelHandler:
    """Handles OpenLabel format operations."""
    
    def __init__(self, schema_path: str):
        """Initialize OpenLabel handler.
        
        Args:
            schema_path: Path to OpenLabel schema file
        """
        self.schema_path = schema_path
        self.sol = None
    
    def initialize_from_schema(self) -> None:
        """Initialize OpenLabel structure from schema."""
        try:
            self.sol = self._create_empty_from_schema(self.schema_path)
            logger.info(f"OpenLabel schema loaded from {self.schema_path}")
        except Exception as e:
            logger.error(f"Failed to load schema: {e}")
            raise
    
    def _create_empty_from_schema(self, schema_path: str) -> Dict[str, Any]:
        """Create empty OpenLabel structure from schema.
        
        Args:
            schema_path: Path to schema file
            
        Returns:
            Empty OpenLabel structure
        """
        def build_empty(schema: Dict[str, Any]) -> Any:
            schema_type = schema.get("type")
            if schema_type == "object":
                props = schema.get("properties", {})
                obj = {}
                for k, v in props.items():
                    obj[k] = build_empty(v)
                return obj
            elif schema_type == "array":
                return []
            elif schema_type in ("string", "number", "integer", "boolean"):
                return None
            else:
                return None
        
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        return build_empty(schema)
    
    def add_metadata(self, video_path: str) -> None:
        """Add metadata to OpenLabel structure.
        
        Args:
            video_path: Path to input video
        """
        if self.sol is None:
            raise RuntimeError("OpenLabel structure not initialized")
            
        self.sol["openlabel"]["metadata"]["schema_version"] = Constants.SCHEMA_VERSION
        self.sol["openlabel"]["metadata"]["tagged_file"] = video_path
        self.sol["openlabel"]["metadata"]["annotator"] = Constants.ANNOTATOR_NAME
        self.sol["openlabel"]["ontologies"]["0"] = Constants.ONTOLOGY_URL
        
        logger.info("OpenLabel metadata initialized")
    
    def add_frame_objects(self, frame_idx: int, results: List[Any], class_map: Dict[int, str]) -> None:
        """Add frame objects to OpenLabel structure.
        
        Args:
            frame_idx: Current frame index
            results: YOLO tracking results
            class_map: Mapping of class IDs to names
        """
        if self.sol is None:
            raise RuntimeError("OpenLabel structure not initialized")
        
        frame_objects = {}
        seen_object = False
        
        for result in results:
            if result.obb.id is not None:
                ids = result.obb.id.int().cpu().tolist()
                classes = result.obb.cls.int().cpu().tolist()
                obbs_xywhr = result.obb.xywhr.cpu().tolist()
                obbs_conf = result.obb.conf.cpu().tolist()
                
                for obj_id, cls, xywhr, conf in zip(ids, classes, obbs_xywhr, obbs_conf):
                    obj_id_str = str(obj_id)
                    seen_object = True
                    xywhr_formatted = [
                        int(xywhr[0]), int(xywhr[1]), int(xywhr[2]), 
                        int(xywhr[3]), round(float(xywhr[4]), 4)
                    ]
                    
                    # Add new object if not seen before
                    if obj_id_str not in self.sol["openlabel"]["objects"]:
                        self.sol["openlabel"]["objects"][obj_id_str] = {
                            "name": f"Object-{obj_id_str}",
                            "type": class_map.get(cls, str(cls)),
                            "ontology_uid": "0"
                        }
                    
                    # Add object data for this frame
                    frame_objects[obj_id_str] = {
                        "object_data": {
                            "rbbox": [{
                                "name": "shape",
                                "val": xywhr_formatted
                            }],
                            "vec": [{
                                "name": "confidence",
                                "val": [float(conf)]
                            }]
                        }
                    }
        
        # Only add frame if there are objects
        if seen_object:
            self.sol["openlabel"]["frames"][str(frame_idx)] = {
                "objects": frame_objects
            }
    
    def sort_objects(self) -> None:
        """Sort objects dictionary numerically by key."""
        if self.sol is None:
            return
            
        objects = self.sol["openlabel"]["objects"]
        sorted_objects = dict(sorted(objects.items(), key=lambda item: int(item[0])))
        self.sol["openlabel"]["objects"] = sorted_objects
    
    def save_to_file(self, output_path: str) -> None:
        """Save OpenLabel data to JSON file.
        
        Args:
            output_path: Output file path
        """
        if self.sol is None:
            raise RuntimeError("OpenLabel structure not initialized")
        
        try:
            self.sort_objects()
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.sol, f, indent=2, ensure_ascii=False)
            logger.info(f"OpenLabel data written to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save OpenLabel data: {e}")
            raise


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run YOLO OBB tracking and output OpenLabel JSON and optionally annotated video."
    )
    parser.add_argument("--weights", required=True, help="Path to YOLO weights file (.pt)")
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output_json", required=True, help="Path to output OpenLabel JSON file")
    parser.add_argument("--schema", required=True, help="Path to OpenLabel JSON schema file")
    parser.add_argument("--output_video", required=False, help="Path to output annotated video file (optional)")
    
    return parser.parse_args()


def process_video(video_processor: VideoProcessor, openlabel_handler: OpenLabelHandler, 
                 config: MarkitConfig) -> None:
    """Main video processing loop.
    
    Args:
        video_processor: Video processor instance
        openlabel_handler: OpenLabel handler instance
        config: Application configuration
    """
    frame_idx = 0
    total_frames = 0
    
    logger.info("Starting video processing...")
    
    try:
        while True:
            success, frame = video_processor.read_frame()
            if not success:
                break
            
            # Process frame with YOLO
            results = video_processor.process_frame(frame)
            
            # Add to OpenLabel structure
            openlabel_handler.add_frame_objects(frame_idx, results, config.class_map)
            
            # Annotate and write frame if output video requested
            if config.output_video_path:
                annotated_frame = FrameAnnotator.annotate_frame(frame, results)
                video_processor.write_frame(annotated_frame)
            
            frame_idx += 1
            total_frames += 1
            
            # Log progress periodically
            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx} frames...")
    
    except Exception as e:
        logger.error(f"Error during video processing: {e}")
        raise
    
    logger.info(f"Video processing completed. Total frames processed: {total_frames}")


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
    """Main function to orchestrate the video processing workflow."""
    try:
        # Parse arguments and create configuration
        args = parse_arguments()
        config = MarkitConfig(args)
        
        # Initialize components
        video_processor = VideoProcessor(config)
        openlabel_handler = OpenLabelHandler(config.schema_path)
        
        # Setup
        video_processor.initialize()
        openlabel_handler.initialize_from_schema()
        openlabel_handler.add_metadata(config.video_path)
        
        # Main processing loop
        process_video(video_processor, openlabel_handler, config)
        
        # Cleanup and save results
        cleanup(video_processor, openlabel_handler, config)
        
        logger.info("Application completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()