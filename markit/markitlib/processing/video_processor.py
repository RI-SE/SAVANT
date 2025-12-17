"""
processing - Video processing and frame annotation

Contains video processor with multi-engine support and frame annotation.
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..config import Constants, ConflictResolutionConfig, DetectionResult, MarkitConfig
from .engines import (
    YOLOEngine,
    OpticalFlowEngine,
    ArUcoEngine,
    ArUcoGPSData,
    VisualMarkerGPSData,
)
from .conflict_resolution import DetectionConflictResolver
from .id_manager import ObjectIDManager

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

        # Create ID manager for unified sequential IDs
        self.id_manager = ObjectIDManager()

        # Initialize ArUco mapping first (if ArUco enabled) to set dynamic ID start point
        aruco_marker_ids = []
        if config.use_aruco:
            # Load ArUco GPS data to get marker IDs
            aruco_gps = ArUcoGPSData(config.aruco_csv_path)
            aruco_marker_ids = aruco_gps.get_marker_ids()

        # Initialize ArUco ID mapping (ArUcos get 0 to N-1)
        self.id_manager.initialize_aruco_mapping(aruco_marker_ids)

        # Initialize visual marker mapping (if provided) - IDs continue after ArUco
        self.visual_marker_data: Optional[VisualMarkerGPSData] = None
        visual_marker_ids = []
        if config.visual_markers_csv_path:
            self.visual_marker_data = VisualMarkerGPSData(
                config.visual_markers_csv_path
            )
            visual_marker_ids = self.visual_marker_data.get_marker_ids()

        # Initialize visual marker ID mapping (visual markers get N to M-1, dynamics start from M)
        self.id_manager.initialize_visual_marker_mapping(visual_marker_ids)

        # Initialize detection engines based on configuration
        if config.use_yolo:
            self.engines.append(
                YOLOEngine(
                    config.weights_path,
                    config.class_map,
                    config.verbose,
                    id_manager=self.id_manager,
                )
            )
        if config.use_optical_flow:
            self.engines.append(
                OpticalFlowEngine(
                    config.optical_flow_params, id_manager=self.id_manager
                )
            )
        if config.use_aruco:
            self.engines.append(
                ArUcoEngine(
                    config.aruco_csv_path,
                    config.aruco_class_id,
                    config.aruco_dict,
                    id_manager=self.id_manager,
                )
            )

        # Initialize conflict resolver if multiple engines and conflict resolution enabled
        if len(self.engines) > 1 and config.enable_conflict_resolution:
            conflict_config = ConflictResolutionConfig(
                iou_threshold=config.iou_threshold,
                yolo_precedence=True,
                enable_logging=config.verbose_conflicts,
            )
            self.conflict_resolver = DetectionConflictResolver(conflict_config)

    def get_aruco_gps_data(self) -> Optional[Tuple[Dict, str]]:
        """Get ArUco GPS data from the ArUco engine if available.

        Returns:
            Tuple of (gps_data dict, csv_name) or None if no ArUco engine
        """
        for engine in self.engines:
            if isinstance(engine, ArUcoEngine):
                return engine.gps_data.gps_data, engine.gps_data.csv_name
        return None

    def get_aruco_id_mapping(self) -> Optional[Dict[int, int]]:
        """Get ArUco physical-to-sequential ID mapping.

        Returns:
            Dict mapping physical ArUco ID to sequential object ID,
            or None if no ArUco markers configured
        """
        if self.id_manager.aruco_count > 0:
            return self.id_manager.aruco_mapping
        return None

    def get_visual_marker_data(
        self,
    ) -> Optional[Tuple[VisualMarkerGPSData, Dict[int, int]]]:
        """Get visual marker GPS data and ID mapping if available.

        Returns:
            Tuple of (VisualMarkerGPSData, id_mapping) or None if no visual markers
        """
        if self.visual_marker_data and self.id_manager.visual_marker_count > 0:
            return self.visual_marker_data, self.id_manager.visual_marker_mapping
        return None

    def initialize(self) -> None:
        """Initialize video capture."""
        try:
            logger.info(f"Opening video file: {self.config.video_path}")
            self.cap = cv2.VideoCapture(self.config.video_path)

            if not self.cap.isOpened():
                raise RuntimeError(
                    f"Failed to open video file: {self.config.video_path}"
                )

            self._get_video_properties()

        except Exception as e:
            logger.error(f"Failed to initialize video processor: {e}")
            raise

    def _get_video_properties(self) -> None:
        """Extract video properties from capture."""
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        logger.info(
            f"Video properties - Width: {self.frame_width}, Height: {self.frame_height}, FPS: {self.fps}"
        )

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
            "active_engines": [engine.__class__.__name__ for engine in self.engines],
            "engine_count": len(self.engines),
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
    def annotate_frame(
        frame: np.ndarray,
        detection_results: List[DetectionResult],
        class_map: Dict[int, str] = None,
    ) -> np.ndarray:
        """Annotate frame with oriented bounding boxes and labels.

        Args:
            frame: Input frame
            detection_results: Detection results from engines
            class_map: Optional class ID to name mapping (uses DEFAULT_CLASS_MAP if None)

        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()

        # Use provided class_map or fall back to default
        if class_map is None:
            class_map = Constants.DEFAULT_CLASS_MAP

        # Color map for different engines
        color_map = {
            "yolo": (0, 255, 0),  # Green
            "optical_flow": (255, 0, 0),  # Blue
            "aruco": (0, 165, 255),  # Orange
            "postprocessed_gap": (
                255,
                0,
                255,
            ),  # Magenta (gap filling - higher priority)
            "postprocessed_rot": (255, 255, 0),  # Cyan (rotation adjustment)
        }

        for detection in detection_results:
            try:
                # Get color for engine
                color = color_map.get(
                    detection.source_engine, (0, 255, 255)
                )  # Yellow default

                # Draw oriented bounding box
                bbox_points = detection.oriented_bbox.astype(np.int32)
                cv2.drawContours(annotated_frame, [bbox_points], 0, color, 2)

                # Draw center point
                center = (int(detection.center[0]), int(detection.center[1]))
                cv2.circle(annotated_frame, center, 3, color, -1)

                # Draw rotation direction indicator (arrow from center in angle direction)
                if detection.angle is not None:
                    arrow_length = 30  # pixels
                    angle_rad = detection.angle
                    end_x = int(detection.center[0] + arrow_length * np.cos(angle_rad))
                    end_y = int(detection.center[1] + arrow_length * np.sin(angle_rad))
                    cv2.arrowedLine(
                        annotated_frame, center, (end_x, end_y), color, 2, tipLength=0.3
                    )

                # Prepare label text
                label_parts = []
                if detection.object_id is not None:
                    label_parts.append(f"ID:{detection.object_id}")

                class_name = class_map.get(
                    detection.class_id, f"cls_{detection.class_id}"
                )
                label_parts.append(f"{class_name}")
                label_parts.append(f"{detection.confidence:.2f}")
                label_parts.append(f"[{detection.source_engine}]")

                label = " ".join(label_parts)

                # Draw label background
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    annotated_frame,
                    (center[0] - 5, center[1] - label_h - 10),
                    (center[0] + label_w + 5, center[1] - 5),
                    color,
                    -1,
                )

                # Draw label text
                cv2.putText(
                    annotated_frame,
                    label,
                    (center[0], center[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

            except Exception as e:
                logger.error(f"Error annotating detection: {e}")

        return annotated_frame
