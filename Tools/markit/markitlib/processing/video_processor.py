"""
processing - Video processing and frame annotation

Contains video processor with multi-engine support and frame annotation.
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..config import Constants, ConflictResolutionConfig, DetectionResult, MarkitConfig
from .engines import YOLOEngine, OpticalFlowEngine, ArUcoEngine
from .conflict_resolution import DetectionConflictResolver

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Enhanced video processor with multi-engine support and conflict resolution."""

    def __init__(self, config: MarkitConfig):
        """Initialize video processor.

        Args:
            config: Application configuration
        """
        self.config = config
        self.engines = []
        self.conflict_resolver = None
        self.cap = None
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0.0

        # Initialize detection engines based on configuration
        if config.use_yolo:
            self.engines.append(YOLOEngine(config.weights_path))
        if config.use_optical_flow:
            self.engines.append(OpticalFlowEngine(config.optical_flow_params))
        if config.use_aruco:
            self.engines.append(ArUcoEngine(config.aruco_csv_path, config.aruco_class_id))

        # Initialize conflict resolver if multiple engines and conflict resolution enabled
        if len(self.engines) > 1 and config.enable_conflict_resolution:
            conflict_config = ConflictResolutionConfig(
                iou_threshold=config.iou_threshold,
                yolo_precedence=True,
                enable_logging=config.verbose_conflicts
            )
            self.conflict_resolver = DetectionConflictResolver(conflict_config)

    def initialize(self) -> None:
        """Initialize video capture."""
        try:
            logger.info(f"Opening video file: {self.config.video_path}")
            self.cap = cv2.VideoCapture(self.config.video_path)

            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video file: {self.config.video_path}")

            self._get_video_properties()

        except Exception as e:
            logger.error(f"Failed to initialize video processor: {e}")
            raise

    def _get_video_properties(self) -> None:
        """Extract video properties from capture."""
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        logger.info(f"Video properties - Width: {self.frame_width}, Height: {self.frame_height}, FPS: {self.fps}")

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame from video.

        Returns:
            Tuple of (success, frame)
        """
        if self.cap is None:
            return False, None
        return self.cap.read()

    def process_frame(self, frame: np.ndarray) -> List[DetectionResult]:
        """Process frame with all engines and resolve conflicts.

        Args:
            frame: Input frame

        Returns:
            Merged detection results with conflicts resolved
        """
        all_results = []

        # Get results from all engines
        for engine in self.engines:
            try:
                engine_results = engine.process_frame(frame)
                all_results.extend(engine_results)
            except Exception as e:
                logger.error(f"Error in {engine.__class__.__name__}: {e}")

        # Resolve conflicts if multiple engines and conflict resolution enabled
        if self.conflict_resolver and len(all_results) > 1:
            resolved_results = self.conflict_resolver.resolve_conflicts(all_results)
            return resolved_results

        return all_results

    def get_detection_statistics(self) -> dict:
        """Get detection and conflict resolution statistics.

        Returns:
            Dictionary with processing statistics
        """
        stats = {
            'active_engines': [engine.__class__.__name__ for engine in self.engines],
            'engine_count': len(self.engines)
        }

        if self.conflict_resolver:
            stats.update(self.conflict_resolver.get_conflict_statistics())

        return stats

    def cleanup(self) -> None:
        """Clean up video resources."""
        # Clean up engines
        for engine in self.engines:
            try:
                engine.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up engine: {e}")

        # Clean up video resources
        if self.cap is not None:
            self.cap.release()
            logger.info("Video capture released")

        cv2.destroyAllWindows()


class FrameAnnotator:
    """Handles frame annotation with bounding boxes and labels."""

    @staticmethod
    def annotate_frame(frame: np.ndarray, detection_results: List[DetectionResult]) -> np.ndarray:
        """Annotate frame with oriented bounding boxes and labels.

        Args:
            frame: Input frame
            detection_results: Detection results from engines

        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()

        # Color map for different engines
        color_map = {
            'yolo': (0, 255, 0),  # Green
            'optical_flow': (255, 0, 0),  # Blue
            'aruco': (0, 165, 255),  # Orange
            'postprocessed_gap': (255, 0, 255),  # Magenta (gap filling - higher priority)
            'postprocessed_rot': (255, 255, 0),  # Cyan (rotation adjustment)
        }

        for detection in detection_results:
            try:
                # Get color for engine
                color = color_map.get(detection.source_engine, (0, 255, 255))  # Yellow default

                # Draw oriented bounding box
                bbox_points = detection.oriented_bbox.astype(np.int32)
                cv2.drawContours(annotated_frame, [bbox_points], 0, color, 2)

                # Draw center point
                center = (int(detection.center[0]), int(detection.center[1]))
                cv2.circle(annotated_frame, center, 3, color, -1)

                # Prepare label text
                label_parts = []
                if detection.object_id is not None:
                    label_parts.append(f"ID:{detection.object_id}")

                class_name = Constants.DEFAULT_CLASS_MAP.get(detection.class_id, f"cls_{detection.class_id}")
                label_parts.append(f"{class_name}")
                label_parts.append(f"{detection.confidence:.2f}")
                label_parts.append(f"[{detection.source_engine}]")

                label = " ".join(label_parts)

                # Draw label background
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_frame,
                            (center[0] - 5, center[1] - label_h - 10),
                            (center[0] + label_w + 5, center[1] - 5),
                            color, -1)

                # Draw label text
                cv2.putText(annotated_frame, label,
                          (center[0], center[1] - 8),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            except Exception as e:
                logger.error(f"Error annotating detection: {e}")

        return annotated_frame
