"""
markit.py

Advanced command-line tool for running multi-engine object detection (YOLO + Optical Flow)
with IoU-based conflict resolution. Exports results in OpenLabel JSON format and optionally
as annotated video.

Usage:
    python markit.py --weights WEIGHTS_PATH --input INPUT_VIDEO --output_json OUTPUT_JSON --schema SCHEMA_JSON [--output_video OUTPUT_VIDEO]

Arguments:
    --weights            Path to YOLO weights file (.pt)
    --input              Path to input video file
    --output_json        Path to output OpenLabel JSON file
    --schema             Path to OpenLabel JSON schema file
    --output_video       Path to output annotated video file (optional)
    --detection-method   Detection method: yolo, optical_flow, or both (default: yolo)
    --iou-threshold      IoU threshold for conflict resolution (default: 0.3)
    --motion-threshold   Optical flow motion threshold (default: 1.5)
    --min-object-area    Minimum object area for optical flow (default: 500)

Features:
    - YOLO OBB (Oriented Bounding Box) detection with tracking
    - Background subtraction + optical flow detection
    - IoU-based conflict resolution with YOLO precedence
    - OpenLabel JSON export with SAVANT ontology integration
    - Configurable detection engines and conflict resolution
"""

__version__ = '2.0.0'

import argparse
import logging
import sys
import os
import json
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict
from dataclasses import dataclass
from abc import ABC, abstractmethod

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


class BBoxOverlapCalculator:
    """Optimized utility class for calculating IoU between oriented bounding boxes."""

    @staticmethod
    def calculate_intersection_over_union(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate IoU between two oriented bounding boxes using OpenCV.

        Args:
            bbox1: First OBB as 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            bbox2: Second OBB as 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

        Returns:
            IoU value between 0 and 1
        """
        try:
            # Ensure correct format
            pts1 = np.array(bbox1, dtype=np.float32).reshape(-1, 2)
            pts2 = np.array(bbox2, dtype=np.float32).reshape(-1, 2)

            # Calculate areas using the shoelace formula
            area1 = BBoxOverlapCalculator._polygon_area(pts1)
            area2 = BBoxOverlapCalculator._polygon_area(pts2)

            if area1 <= 0 or area2 <= 0:
                return 0.0

            # Find intersection using Sutherland-Hodgman clipping
            intersection_points = BBoxOverlapCalculator._sutherland_hodgman_clip(pts1, pts2)

            if len(intersection_points) < 3:
                return 0.0  # No meaningful intersection

            intersection_area = BBoxOverlapCalculator._polygon_area(intersection_points)

            if intersection_area <= 0:
                return 0.0

            # Calculate IoU
            union_area = area1 + area2 - intersection_area

            if union_area <= 0:
                return 0.0

            return intersection_area / union_area

        except Exception as e:
            logger.debug(f"IoU calculation failed: {e}")
            return 0.0

    @staticmethod
    def _polygon_area(points: np.ndarray) -> float:
        """Calculate polygon area using the shoelace formula.

        Args:
            points: Array of polygon vertices [[x1,y1], [x2,y2], ...]

        Returns:
            Polygon area
        """
        if len(points) < 3:
            return 0.0

        x = points[:, 0]
        y = points[:, 1]

        # Shoelace formula
        area = 0.5 * abs(sum(x[i] * y[(i + 1) % len(x)] - x[(i + 1) % len(x)] * y[i]
                            for i in range(len(x))))
        return area

    @staticmethod
    def _sutherland_hodgman_clip(subject_polygon: np.ndarray, clip_polygon: np.ndarray) -> np.ndarray:
        """Clip subject polygon against clip polygon using Sutherland-Hodgman algorithm.

        Args:
            subject_polygon: Polygon to be clipped
            clip_polygon: Clipping polygon

        Returns:
            Intersection polygon vertices
        """
        def _is_inside(point: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray) -> bool:
            """Check if point is inside the edge (left side of directed edge)."""
            edge_vec = edge_end - edge_start
            point_vec = point - edge_start
            return (edge_vec[0] * point_vec[1] - edge_vec[1] * point_vec[0]) >= 0

        def _intersection_point(p1: np.ndarray, p2: np.ndarray,
                              edge_start: np.ndarray, edge_end: np.ndarray) -> np.ndarray:
            """Find intersection point between line segment p1-p2 and edge."""
            d1 = edge_end - edge_start
            d2 = p2 - p1

            denominator = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(denominator) < 1e-10:
                return p1  # Lines are parallel

            p1_vec = p1 - edge_start
            t = (p1_vec[0] * d2[1] - p1_vec[1] * d2[0]) / denominator
            return edge_start + t * d1

        output_list = subject_polygon.tolist()

        # Process each edge of the clipping polygon
        for i in range(len(clip_polygon)):
            if not output_list:
                break

            edge_start = clip_polygon[i]
            edge_end = clip_polygon[(i + 1) % len(clip_polygon)]

            input_list = output_list
            output_list = []

            if not input_list:
                continue

            if len(input_list) > 0:
                s = np.array(input_list[-1])

                for vertex in input_list:
                    e = np.array(vertex)

                    if _is_inside(e, edge_start, edge_end):
                        if not _is_inside(s, edge_start, edge_end):
                            # Entering the clipping area
                            intersection = _intersection_point(s, e, edge_start, edge_end)
                            output_list.append(intersection.tolist())
                        output_list.append(e.tolist())
                    elif _is_inside(s, edge_start, edge_end):
                        # Leaving the clipping area
                        intersection = _intersection_point(s, e, edge_start, edge_end)
                        output_list.append(intersection.tolist())

                    s = e

        return np.array(output_list) if output_list else np.array([])


class DetectionConflictResolver:
    """Handles conflicts between different detection engines using IoU."""

    def __init__(self, config: ConflictResolutionConfig):
        self.config = config
        self.overlap_calc = BBoxOverlapCalculator()
        self.conflicts_resolved = 0
        self.total_conflicts = 0

        # Validate IoU threshold
        if not (0.0 <= self.config.iou_threshold <= 1.0):
            raise ValueError(f"IoU threshold must be between 0.0 and 1.0, got: {self.config.iou_threshold}")

    def resolve_conflicts(self, detection_results: List[DetectionResult]) -> List[DetectionResult]:
        """Resolve conflicts between detection results using IoU with YOLO precedence.

        Args:
            detection_results: List of detection results from all engines

        Returns:
            Filtered list with conflicts resolved
        """
        if len(detection_results) <= 1:
            return detection_results

        # Separate results by engine
        yolo_results = [r for r in detection_results if r.source_engine == 'yolo']
        optical_flow_results = [r for r in detection_results if r.source_engine == 'optical_flow']
        other_results = [r for r in detection_results if r.source_engine not in ['yolo', 'optical_flow']]

        # Keep all YOLO results (highest precedence)
        final_results = yolo_results.copy()

        # Filter optical flow results that conflict with YOLO
        filtered_optical_flow = self._filter_conflicting_detections_iou(
            primary_detections=yolo_results,
            secondary_detections=optical_flow_results
        )

        final_results.extend(filtered_optical_flow)
        final_results.extend(other_results)

        if self.config.enable_logging:
            conflicts = len(optical_flow_results) - len(filtered_optical_flow)
            if conflicts > 0:
                logger.info(f"Resolved {conflicts} conflicts using IoU threshold {self.config.iou_threshold:.2f}")

        return final_results

    def _filter_conflicting_detections_iou(self, primary_detections: List[DetectionResult],
                                         secondary_detections: List[DetectionResult]) -> List[DetectionResult]:
        """Filter secondary detections that conflict with primary detections using IoU.

        Args:
            primary_detections: High-priority detections (e.g., YOLO)
            secondary_detections: Lower-priority detections (e.g., optical flow)

        Returns:
            Filtered secondary detections with IoU conflicts removed
        """
        if not primary_detections:
            return secondary_detections

        filtered_results = []

        for secondary_det in secondary_detections:
            has_conflict = False
            max_iou = 0.0
            conflicting_primary = None

            for primary_det in primary_detections:
                iou = self.overlap_calc.calculate_intersection_over_union(
                    secondary_det.oriented_bbox,
                    primary_det.oriented_bbox
                )

                if iou >= self.config.iou_threshold:
                    has_conflict = True
                    if iou > max_iou:
                        max_iou = iou
                        conflicting_primary = primary_det
                    self.total_conflicts += 1

            if not has_conflict:
                filtered_results.append(secondary_det)
            else:
                self.conflicts_resolved += 1
                if self.config.enable_logging:
                    logger.debug(
                        f"Conflict resolved: {secondary_det.source_engine} "
                        f"(IoU={max_iou:.3f}) dropped in favor of {conflicting_primary.source_engine}"
                    )

        return filtered_results

    def get_conflict_statistics(self) -> dict:
        """Get statistics about resolved conflicts.

        Returns:
            Dictionary with conflict resolution statistics
        """
        return {
            'total_conflicts': self.total_conflicts,
            'conflicts_resolved': self.conflicts_resolved,
            'resolution_rate': (self.conflicts_resolved / max(1, self.total_conflicts)) * 100,
            'iou_threshold': self.config.iou_threshold
        }


class BaseDetectionEngine(ABC):
    """Abstract base class for detection engines."""

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> List[DetectionResult]:
        """Process frame and return detection results."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up engine resources."""
        pass


class YOLOEngine(BaseDetectionEngine):
    """YOLO-based detection engine."""

    def __init__(self, weights_path: str):
        """Initialize YOLO engine.

        Args:
            weights_path: Path to YOLO weights file
        """
        self.weights_path = weights_path
        self.model = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize YOLO model."""
        try:
            logger.info(f"Loading YOLO model from {self.weights_path}")
            self.model = YOLO(self.weights_path)
            self.model.verbose = False
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {e}")
            raise

    def process_frame(self, frame: np.ndarray) -> List[DetectionResult]:
        """Process frame with YOLO model.

        Args:
            frame: Input frame

        Returns:
            List of YOLO detection results
        """
        try:
            results = self.model.track(frame, persist=True, verbose=False)
            detection_results = []

            for result in results:
                if result.obb is not None and len(result.obb) > 0:
                    # Extract oriented bounding box data
                    ids = result.obb.id.cpu().numpy() if result.obb.id is not None else [None] * len(result.obb)
                    classes = result.obb.cls.cpu().numpy()
                    obbs_xywhr = result.obb.xywhr.cpu().numpy()
                    obbs_conf = result.obb.conf.cpu().numpy()

                    for obj_id, cls, xywhr, conf in zip(ids, classes, obbs_xywhr, obbs_conf):
                        # Convert YOLO xywhr to 4 corner points
                        center_x, center_y, width, height, rotation = xywhr

                        # Create oriented bounding box points
                        cos_r = np.cos(rotation)
                        sin_r = np.sin(rotation)

                        # Half dimensions
                        hw = width / 2
                        hh = height / 2

                        # Corner points relative to center
                        corners = np.array([
                            [-hw, -hh],
                            [hw, -hh],
                            [hw, hh],
                            [-hw, hh]
                        ])

                        # Rotate corners
                        rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
                        rotated_corners = corners @ rotation_matrix.T

                        # Translate to center position
                        oriented_bbox = rotated_corners + np.array([center_x, center_y])

                        detection_results.append(DetectionResult(
                            object_id=int(obj_id) if obj_id is not None else None,
                            class_id=int(cls),
                            confidence=float(conf),
                            oriented_bbox=oriented_bbox,
                            center=(float(center_x), float(center_y)),
                            angle=float(rotation),
                            source_engine='yolo'
                        ))

            return detection_results

        except Exception as e:
            logger.error(f"Error processing frame with YOLO: {e}")
            return []

    def cleanup(self) -> None:
        """Clean up YOLO engine resources."""
        self.model = None


class SimpleTracker:
    """Simple object tracker for optical flow detections."""

    def __init__(self, max_distance: float = 50.0, id_offset: int = 0):
        self.max_distance = max_distance
        self.tracks = {}
        self.next_id = 1 + id_offset
        self.id_offset = id_offset

    def get_id(self, center: Tuple[float, float]) -> int:
        """Get object ID for center position.

        Args:
            center: Object center position (x, y)

        Returns:
            Object ID
        """
        min_distance = float('inf')
        best_track_id = None

        # Find closest existing track
        for track_id, track_center in self.tracks.items():
            distance = np.sqrt((center[0] - track_center[0])**2 + (center[1] - track_center[1])**2)
            if distance < min_distance and distance < self.max_distance:
                min_distance = distance
                best_track_id = track_id

        if best_track_id is not None:
            # Update existing track
            self.tracks[best_track_id] = center
            return best_track_id
        else:
            # Create new track
            new_id = self.next_id
            self.tracks[new_id] = center
            self.next_id += 1
            return new_id


class OpticalFlowEngine(BaseDetectionEngine):
    """Optical flow + background subtraction detection engine."""

    def __init__(self, params: OpticalFlowParams):
        """Initialize optical flow engine.

        Args:
            params: Optical flow parameters
        """
        self.params = params
        self.back_sub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.prev_gray = None
        self.object_tracker = SimpleTracker(id_offset=1000000)
        self.optical_flow_method = None  # Will be set by availability check
        self.optical_flow_available = self._check_optical_flow_availability()

        if self.optical_flow_available:
            logger.info("Optical flow engine initialized with motion detection")
        else:
            logger.warning("Optical flow not available - using background subtraction only")

    def _check_optical_flow_availability(self) -> bool:
        """Check if optical flow functions are available in OpenCV.

        Returns:
            True if optical flow is available, False otherwise
        """
        # Log OpenCV version for diagnostics
        logger.info(f"OpenCV version: {cv2.__version__}")

        # Try multiple optical flow methods in order of preference
        optical_flow_methods = [
            ('calcOpticalFlowPyrFarneback', self._test_farneback),
            ('calcOpticalFlowPyrLK', self._test_lucas_kanade),
            ('optflow.calcOpticalFlowSF', self._test_simple_flow)
        ]

        for method_name, test_func in optical_flow_methods:
            try:
                if test_func():
                    logger.info(f"Optical flow available using: {method_name}")
                    self.optical_flow_method = method_name
                    return True
            except Exception as e:
                logger.debug(f"{method_name} not available: {e}")
                continue

        logger.warning("No optical flow methods available - check OpenCV installation")
        logger.info("Install with: pip install opencv-contrib-python>=4.5.0")
        return False

    def _test_farneback(self) -> bool:
        """Test Farneback optical flow method."""
        if not hasattr(cv2, 'calcOpticalFlowPyrFarneback'):
            return False

        dummy1 = np.zeros((10, 10), dtype=np.uint8)
        dummy2 = np.zeros((10, 10), dtype=np.uint8)
        flow = cv2.calcOpticalFlowPyrFarneback(dummy1, dummy2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return flow is not None

    def _test_lucas_kanade(self) -> bool:
        """Test Lucas-Kanade optical flow method."""
        if not hasattr(cv2, 'calcOpticalFlowPyrLK'):
            return False

        dummy1 = np.zeros((10, 10), dtype=np.uint8)
        dummy2 = np.zeros((10, 10), dtype=np.uint8)
        # Need some points for LK
        points = np.array([[5.0, 5.0]], dtype=np.float32).reshape(-1, 1, 2)
        new_points, status, error = cv2.calcOpticalFlowPyrLK(dummy1, dummy2, points, None)
        return new_points is not None

    def _test_simple_flow(self) -> bool:
        """Test SimpleFlow optical flow method (contrib module)."""
        try:
            if hasattr(cv2, 'optflow') and hasattr(cv2.optflow, 'calcOpticalFlowSF'):
                dummy1 = np.zeros((10, 10), dtype=np.uint8)
                dummy2 = np.zeros((10, 10), dtype=np.uint8)
                flow = cv2.optflow.calcOpticalFlowSF(dummy1, dummy2, 3, 2, 4)
                return flow is not None
        except:
            pass
        return False

    def _calculate_optical_flow(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> Optional[np.ndarray]:
        """Calculate optical flow magnitude using the available method.

        Args:
            prev_gray: Previous frame (grayscale)
            curr_gray: Current frame (grayscale)

        Returns:
            Magnitude array or None if calculation fails
        """
        try:
            if self.optical_flow_method == 'calcOpticalFlowPyrFarneback':
                flow = cv2.calcOpticalFlowPyrFarneback(
                    prev_gray, curr_gray, None, 0.5, 5, 21, 3, 7, 1.5, 0
                )
                magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                return magnitude

            elif self.optical_flow_method == 'calcOpticalFlowPyrLK':
                # For Lucas-Kanade, we need feature points
                # Use goodFeaturesToTrack to find points
                points = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3,
                                               minDistance=7, blockSize=7)
                if points is not None and len(points) > 0:
                    new_points, status, error = cv2.calcOpticalFlowPyrLK(
                        prev_gray, curr_gray, points, None
                    )

                    # Create magnitude map from point movements
                    magnitude = np.zeros_like(prev_gray, dtype=np.float32)
                    good_points = status.ravel() == 1

                    if np.any(good_points):
                        old_pts = points[good_points].reshape(-1, 2)
                        new_pts = new_points[good_points].reshape(-1, 2)

                        # Calculate point displacements
                        displacements = np.linalg.norm(new_pts - old_pts, axis=1)

                        # Map displacements to image
                        for i, (pt, disp) in enumerate(zip(old_pts.astype(int), displacements)):
                            if 0 <= pt[1] < magnitude.shape[0] and 0 <= pt[0] < magnitude.shape[1]:
                                magnitude[pt[1], pt[0]] = disp

                        # Dilate to spread motion information
                        kernel = np.ones((15, 15), np.uint8)
                        magnitude = cv2.dilate(magnitude, kernel, iterations=1)

                    return magnitude
                else:
                    return np.zeros_like(prev_gray, dtype=np.float32)

            elif self.optical_flow_method == 'optflow.calcOpticalFlowSF':
                flow = cv2.optflow.calcOpticalFlowSF(prev_gray, curr_gray, 3, 2, 4)
                magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                return magnitude

            else:
                logger.warning(f"Unknown optical flow method: {self.optical_flow_method}")
                return None

        except Exception as e:
            logger.error(f"Optical flow calculation failed: {e}")
            return None

    def process_frame(self, frame: np.ndarray) -> List[DetectionResult]:
        """Process frame with optical flow + background subtraction.

        Args:
            frame: Input frame

        Returns:
            List of optical flow detection results
        """
        try:
            results = []

            # 1. Background subtraction
            fg_mask = self.back_sub.apply(frame)

            # 2. Optical flow (if available and previous frame exists)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            motion_mask = np.zeros_like(fg_mask)

            if self.optical_flow_available and self.prev_gray is not None:
                try:
                    magnitude = self._calculate_optical_flow(self.prev_gray, gray)
                    if magnitude is not None:
                        motion_mask = (magnitude > self.params.motion_threshold).astype(np.uint8) * 255
                        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                        motion_mask = cv2.dilate(motion_mask, dilate_kernel, iterations=1)
                    else:
                        self.optical_flow_available = False
                except Exception as e:
                    logger.warning(f"Optical flow calculation failed: {e}")
                    self.optical_flow_available = False

            # 3. Combine masks (if optical flow is available) or use background subtraction only
            if self.optical_flow_available and self.prev_gray is not None:
                combined_mask = cv2.bitwise_or(fg_mask, motion_mask)
            else:
                combined_mask = fg_mask

            # 4. Clean up and find contours
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (self.params.morph_kernel_size, self.params.morph_kernel_size))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 5. Generate oriented bounding boxes
            for contour in contours:
                if cv2.contourArea(contour) > self.params.min_area:
                    # Get minimum area rectangle (OBB)
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.array(box, dtype=np.int32)

                    # Calculate center and angle
                    (center_x, center_y), (width, height), angle = rect

                    # Assign object ID (simple tracking)
                    obj_id = self.object_tracker.get_id((center_x, center_y))

                    # Calculate confidence based on contour area (normalized)
                    area = cv2.contourArea(contour)
                    confidence = min(0.9, max(0.3, area / 10000.0))  # Simple area-based confidence

                    results.append(DetectionResult(
                        object_id=obj_id,
                        class_id=0,  # Generic "moving object" class
                        confidence=confidence,
                        oriented_bbox=box.astype(np.float32),
                        center=(center_x, center_y),
                        angle=angle,
                        source_engine='optical_flow'
                    ))

            self.prev_gray = gray.copy()
            return results

        except Exception as e:
            logger.error(f"Error processing frame with optical flow: {e}")
            return []

    def cleanup(self) -> None:
        """Clean up optical flow engine resources."""
        self.back_sub = None
        self.prev_gray = None


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
        self.out = None
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0.0

        # Initialize detection engines based on configuration
        if config.use_yolo:
            self.engines.append(YOLOEngine(config.weights_path))
        if config.use_optical_flow:
            self.engines.append(OpticalFlowEngine(config.optical_flow_params))

        # Initialize conflict resolver if multiple engines and conflict resolution enabled
        if len(self.engines) > 1 and config.enable_conflict_resolution:
            conflict_config = ConflictResolutionConfig(
                iou_threshold=config.iou_threshold,
                yolo_precedence=True,
                enable_logging=config.verbose_conflicts
            )
            self.conflict_resolver = DetectionConflictResolver(conflict_config)

    def initialize(self) -> None:
        """Initialize video capture and writer."""
        try:
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

    def write_frame(self, frame: np.ndarray) -> None:
        """Write frame to output video.

        Args:
            frame: Frame to write
        """
        if self.out is not None:
            self.out.write(frame)

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

        if self.out is not None:
            self.out.release()
            logger.info("Video writer released")

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


class PostprocessingPass(ABC):
    """Abstract base class for postprocessing passes."""

    def set_video_properties(self, frame_width: int, frame_height: int, fps: float) -> None:
        """Set video properties for passes that need them.

        Args:
            frame_width: Video frame width in pixels
            frame_height: Video frame height in pixels
            fps: Video frames per second
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps

    @abstractmethod
    def process(self, openlabel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process OpenLabel data and return modified version.

        Args:
            openlabel_data: Complete OpenLabel data structure

        Returns:
            Modified OpenLabel data structure
        """
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about what this pass did.

        Returns:
            Dictionary with processing statistics
        """
        pass


class GapDetectionPass(PostprocessingPass):
    """Detect gaps in object ID frame sequences."""

    def __init__(self):
        self.gaps_detected = {}
        self.objects_with_gaps = set()

    def process(self, openlabel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect gaps in object tracking sequences.

        Args:
            openlabel_data: Complete OpenLabel data structure

        Returns:
            Unmodified OpenLabel data (detection only, no fixes yet)
        """
        frames = openlabel_data.get("openlabel", {}).get("frames", {})

        object_frames = defaultdict(list)

        for frame_idx_str, frame_data in frames.items():
            frame_idx = int(frame_idx_str)
            objects = frame_data.get("objects", {})

            for obj_id_str in objects.keys():
                object_frames[obj_id_str].append(frame_idx)

        for obj_id, frame_list in object_frames.items():
            if len(frame_list) < 2:
                continue

            frame_list_sorted = sorted(frame_list)
            gaps = []

            for i in range(len(frame_list_sorted) - 1):
                current_frame = frame_list_sorted[i]
                next_frame = frame_list_sorted[i + 1]
                gap_size = next_frame - current_frame - 1

                if gap_size > 0:
                    gaps.append({
                        'start_frame': current_frame,
                        'end_frame': next_frame,
                        'gap_size': gap_size
                    })

            if gaps:
                self.gaps_detected[obj_id] = {
                    'frame_range': (frame_list_sorted[0], frame_list_sorted[-1]),
                    'total_frames': len(frame_list_sorted),
                    'gaps': gaps
                }
                self.objects_with_gaps.add(obj_id)

                logger.warning(
                    f"Object ID {obj_id}: detected {len(gaps)} gap(s) in frame sequence "
                    f"[{frame_list_sorted[0]}-{frame_list_sorted[-1]}]"
                )
                for gap in gaps:
                    logger.warning(
                        f"  Gap: frames {gap['start_frame']} -> {gap['end_frame']} "
                        f"(missing {gap['gap_size']} frame(s))"
                    )

        return openlabel_data

    def get_statistics(self) -> Dict[str, Any]:
        """Get gap detection statistics.

        Returns:
            Dictionary with gap detection statistics
        """
        total_gaps = sum(len(info['gaps']) for info in self.gaps_detected.values())

        return {
            'objects_with_gaps': len(self.objects_with_gaps),
            'total_gaps_detected': total_gaps,
            'gap_details': self.gaps_detected
        }


class GapFillingPass(PostprocessingPass):
    """Fill gaps in object ID frame sequences by interpolating positions."""

    def __init__(self):
        self.gaps_filled = 0
        self.frames_added = 0
        self.objects_processed = set()

    def process(self, openlabel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fill gaps in object tracking sequences by interpolating positions.

        Args:
            openlabel_data: Complete OpenLabel data structure

        Returns:
            Modified OpenLabel data with gaps filled
        """
        frames = openlabel_data.get("openlabel", {}).get("frames", {})

        object_frames = defaultdict(list)

        for frame_idx_str, frame_data in frames.items():
            frame_idx = int(frame_idx_str)
            objects = frame_data.get("objects", {})

            for obj_id_str in objects.keys():
                object_frames[obj_id_str].append(frame_idx)

        for obj_id, frame_list in object_frames.items():
            if len(frame_list) < 2:
                continue

            frame_list_sorted = sorted(frame_list)

            for i in range(len(frame_list_sorted) - 1):
                frame_before = frame_list_sorted[i]
                frame_after = frame_list_sorted[i + 1]
                gap_size = frame_after - frame_before - 1

                if gap_size > 0:
                    self._fill_gap(
                        openlabel_data,
                        obj_id,
                        frame_before,
                        frame_after,
                        gap_size
                    )

        return openlabel_data

    def _fill_gap(self, openlabel_data: Dict[str, Any], obj_id: str,
                  frame_before: int, frame_after: int, gap_size: int) -> None:
        """Fill a specific gap by interpolating object positions.

        Args:
            openlabel_data: OpenLabel data structure
            obj_id: Object ID string
            frame_before: Last frame before gap
            frame_after: First frame after gap
            gap_size: Number of missing frames
        """
        frames = openlabel_data["openlabel"]["frames"]

        obj_data_before = frames[str(frame_before)]["objects"][obj_id]["object_data"]
        obj_data_after = frames[str(frame_after)]["objects"][obj_id]["object_data"]

        rbbox_before = obj_data_before["rbbox"][0]["val"]
        rbbox_after = obj_data_after["rbbox"][0]["val"]

        x_before, y_before, w_before, h_before, r_before = rbbox_before
        x_after, y_after, w_after, h_after, r_after = rbbox_after

        delta_x = x_after - x_before
        delta_y = y_after - y_before

        total_steps = gap_size + 1

        for step in range(1, gap_size + 1):
            interpolation_factor = step / total_steps

            x_interpolated = int(x_before + delta_x * interpolation_factor)
            y_interpolated = int(y_before + delta_y * interpolation_factor)

            missing_frame_idx = frame_before + step
            missing_frame_str = str(missing_frame_idx)

            if missing_frame_str not in frames:
                frames[missing_frame_str] = {"objects": {}}

            frames[missing_frame_str]["objects"][obj_id] = {
                "object_data": {
                    "rbbox": [{
                        "name": "shape",
                        "val": [x_interpolated, y_interpolated, w_before, h_before, r_before]
                    }],
                    "vec": [
                        {
                            "name": "annotator",
                            "val": ["markit_housekeeping(gap)"]
                        },
                        {
                            "name": "confidence",
                            "val": [0.6666]
                        }
                    ]
                }
            }

            self.frames_added += 1

        self.gaps_filled += 1
        self.objects_processed.add(obj_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get gap filling statistics.

        Returns:
            Dictionary with gap filling statistics
        """
        return {
            'objects_processed': len(self.objects_processed),
            'gaps_filled': self.gaps_filled,
            'frames_added': self.frames_added
        }


class DuplicateRemovalPass(PostprocessingPass):
    """Remove duplicate bounding boxes based on IOU threshold."""

    def __init__(self, avg_iou_threshold: float = 0.7, min_iou_threshold: float = 0.3):
        self.objects_deleted = 0
        self.duplicate_pairs_found = 0
        self.frames_modified = 0
        self.iou_calculator = BBoxOverlapCalculator()
        self.deletion_details = []
        self.avg_iou_threshold = avg_iou_threshold
        self.min_iou_threshold = min_iou_threshold

    def process(self, openlabel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove duplicate objects based on IOU analysis.

        Args:
            openlabel_data: Complete OpenLabel data structure

        Returns:
            Modified OpenLabel data with duplicates removed
        """
        frames = openlabel_data.get("openlabel", {}).get("frames", {})
        objects = openlabel_data.get("openlabel", {}).get("objects", {})

        object_frame_map = defaultdict(list)

        for frame_idx_str, frame_data in frames.items():
            frame_idx = int(frame_idx_str)
            frame_objects = frame_data.get("objects", {})

            for obj_id_str in frame_objects.keys():
                object_frame_map[obj_id_str].append(frame_idx)

        objects_to_delete = set()
        object_ids = list(objects.keys())

        for i in range(len(object_ids)):
            for j in range(i + 1, len(object_ids)):
                obj_a = object_ids[i]
                obj_b = object_ids[j]

                if obj_a in objects_to_delete or obj_b in objects_to_delete:
                    continue

                if self._are_duplicates(obj_a, obj_b, object_frame_map, frames):
                    self.duplicate_pairs_found += 1

                    obj_to_delete = self._choose_object_to_delete(
                        obj_a, obj_b, object_frame_map, frames
                    )
                    obj_to_keep = obj_b if obj_to_delete == obj_a else obj_a
                    objects_to_delete.add(obj_to_delete)

                    frames_list = sorted(object_frame_map[obj_to_delete])
                    self.deletion_details.append({
                        'deleted_object': obj_to_delete,
                        'kept_object': obj_to_keep,
                        'frame_start': frames_list[0] if frames_list else None,
                        'frame_end': frames_list[-1] if frames_list else None
                    })

        for obj_id in objects_to_delete:
            if obj_id in objects:
                del objects[obj_id]
                self.objects_deleted += 1

            for frame_idx_str, frame_data in frames.items():
                frame_objects = frame_data.get("objects", {})
                if obj_id in frame_objects:
                    del frame_objects[obj_id]
                    self.frames_modified += 1

        for detail in self.deletion_details:
            logger.info(
                f"Deleted object {detail['deleted_object']} (duplicate of {detail['kept_object']}) "
                f"from frames {detail['frame_start']}-{detail['frame_end']}"
            )

        return openlabel_data

    def _are_duplicates(self, obj_a: str, obj_b: str,
                       object_frame_map: Dict[str, List[int]],
                       frames: Dict[str, Any]) -> bool:
        """Check if two objects are duplicates based on IOU thresholds.

        Args:
            obj_a: First object ID
            obj_b: Second object ID
            object_frame_map: Mapping of object IDs to frame lists
            frames: Frame data

        Returns:
            True if objects are duplicates (avg IOU > 0.8 and min IOU > 0.5)
        """
        frames_a = set(object_frame_map.get(obj_a, []))
        frames_b = set(object_frame_map.get(obj_b, []))
        shared_frames = frames_a.intersection(frames_b)

        if len(shared_frames) == 0:
            return False

        ious = []

        for frame_idx in shared_frames:
            frame_str = str(frame_idx)
            frame_data = frames[frame_str]
            frame_objects = frame_data.get("objects", {})

            bbox_a = self._extract_bbox(frame_objects[obj_a])
            bbox_b = self._extract_bbox(frame_objects[obj_b])

            if bbox_a is not None and bbox_b is not None:
                iou = self.iou_calculator.calculate_intersection_over_union(bbox_a, bbox_b)
                ious.append(iou)

        if len(ious) == 0:
            return False

        avg_iou = sum(ious) / len(ious)
        min_iou = min(ious)

        return avg_iou > self.avg_iou_threshold and min_iou > self.min_iou_threshold

    def _extract_bbox(self, object_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract bounding box from object data and convert to corner points.

        Args:
            object_data: Object data containing rbbox

        Returns:
            4 corner points as numpy array, or None if extraction fails
        """
        try:
            rbbox_val = object_data["object_data"]["rbbox"][0]["val"]
            x, y, w, h, r = rbbox_val

            cos_r = np.cos(r)
            sin_r = np.sin(r)

            hw = w / 2
            hh = h / 2

            corners = np.array([
                [-hw, -hh],
                [hw, -hh],
                [hw, hh],
                [-hw, hh]
            ])

            rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            rotated_corners = corners @ rotation_matrix.T

            oriented_bbox = rotated_corners + np.array([x, y])

            return oriented_bbox.astype(np.float32)

        except (KeyError, IndexError, ValueError) as e:
            logger.debug(f"Failed to extract bbox: {e}")
            return None

    def _choose_object_to_delete(self, obj_a: str, obj_b: str,
                                 object_frame_map: Dict[str, List[int]],
                                 frames: Dict[str, Any]) -> str:
        """Choose which object to delete from a duplicate pair.

        Args:
            obj_a: First object ID
            obj_b: Second object ID
            object_frame_map: Mapping of object IDs to frame lists
            frames: Frame data

        Returns:
            Object ID to delete
        """
        frames_a = len(object_frame_map.get(obj_a, []))
        frames_b = len(object_frame_map.get(obj_b, []))

        if frames_a != frames_b:
            return obj_a if frames_a < frames_b else obj_b

        conf_a = self._calculate_average_confidence(obj_a, object_frame_map, frames)
        conf_b = self._calculate_average_confidence(obj_b, object_frame_map, frames)

        return obj_a if conf_a < conf_b else obj_b

    def _calculate_average_confidence(self, obj_id: str,
                                     object_frame_map: Dict[str, List[int]],
                                     frames: Dict[str, Any]) -> float:
        """Calculate average confidence for an object across all its frames.

        Args:
            obj_id: Object ID
            object_frame_map: Mapping of object IDs to frame lists
            frames: Frame data

        Returns:
            Average confidence value
        """
        confidences = []

        for frame_idx in object_frame_map.get(obj_id, []):
            frame_str = str(frame_idx)
            frame_data = frames[frame_str]
            frame_objects = frame_data.get("objects", {})

            if obj_id in frame_objects:
                try:
                    vec_list = frame_objects[obj_id]["object_data"]["vec"]
                    for vec_item in vec_list:
                        if vec_item.get("name") == "confidence":
                            conf_values = vec_item.get("val", [])
                            if conf_values:
                                confidences.append(conf_values[-1])
                            break
                except (KeyError, IndexError):
                    pass

        return sum(confidences) / len(confidences) if confidences else 0.0

    def get_statistics(self) -> Dict[str, Any]:
        """Get duplicate removal statistics.

        Returns:
            Dictionary with duplicate removal statistics
        """
        return {
            'objects_deleted': self.objects_deleted,
            'duplicate_pairs_found': self.duplicate_pairs_found,
            'frames_modified': self.frames_modified
        }


class RotationAdjustmentPass(PostprocessingPass):
    """Adjust rotation values based on movement direction."""

    def __init__(self, rotation_threshold: float = 0.01):
        self.rotations_adjusted = 0
        self.rotations_kept = 0
        self.rotations_copied = 0
        self.objects_processed = 0
        self.rotation_threshold = rotation_threshold

    def process(self, openlabel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust rotation values based on movement direction.

        Args:
            openlabel_data: Complete OpenLabel data structure

        Returns:
            Modified OpenLabel data with adjusted rotations
        """
        frames = openlabel_data.get("openlabel", {}).get("frames", {})

        object_frame_map = defaultdict(list)

        for frame_idx_str, frame_data in frames.items():
            frame_idx = int(frame_idx_str)
            frame_objects = frame_data.get("objects", {})

            for obj_id_str in frame_objects.keys():
                object_frame_map[obj_id_str].append(frame_idx)

        for obj_id, frame_list in object_frame_map.items():
            if len(frame_list) < 2:
                continue

            self.objects_processed += 1
            frame_list_sorted = sorted(frame_list)
            last_valid_angle = None

            for i in range(len(frame_list_sorted)):
                current_frame = frame_list_sorted[i]
                is_last_frame = (i == len(frame_list_sorted) - 1)

                if is_last_frame:
                    if last_valid_angle is not None:
                        current_frame_str = str(current_frame)
                        frame_obj_data = frames[current_frame_str]["objects"][obj_id]
                        rbbox = frame_obj_data["object_data"]["rbbox"][0]["val"]
                        r_current = rbbox[4]

                        if abs(last_valid_angle - r_current) > self.rotation_threshold:
                            self._apply_rotation_adjustment(frame_obj_data, last_valid_angle)
                            self.rotations_copied += 1
                    break

                r_new = self._calculate_smoothed_rotation(
                    frames, obj_id, current_frame, frame_list_sorted, i
                )

                current_frame_str = str(current_frame)
                frame_obj_data = frames[current_frame_str]["objects"][obj_id]
                rbbox = frame_obj_data["object_data"]["rbbox"][0]["val"]
                r_current = rbbox[4]

                if r_new is None:
                    if last_valid_angle is not None:
                        r_new = last_valid_angle
                        self._apply_rotation_adjustment(frame_obj_data, r_new)
                        self.rotations_copied += 1
                    continue
                else:
                    last_valid_angle = r_new

                if abs(r_new - r_current) > self.rotation_threshold:
                    self._apply_rotation_adjustment(frame_obj_data, r_new)
                    self.rotations_adjusted += 1
                else:
                    self.rotations_kept += 1

        return openlabel_data

    def _apply_rotation_adjustment(self, frame_obj_data: Dict[str, Any], r_new: float) -> None:
        """Apply rotation adjustment and update annotator/confidence.

        Args:
            frame_obj_data: Frame object data
            r_new: New rotation value
        """
        rbbox = frame_obj_data["object_data"]["rbbox"][0]["val"]
        rbbox[4] = r_new

        vec_list = frame_obj_data["object_data"]["vec"]
        annotator_found = False
        confidence_found = False

        for vec_item in vec_list:
            if vec_item.get("name") == "annotator":
                vec_item["val"].append("markit_housekeep(rot)")
                annotator_found = True
            elif vec_item.get("name") == "confidence":
                vec_item["val"].append(0.8888)
                confidence_found = True

        if not annotator_found:
            vec_list.insert(0, {
                "name": "annotator",
                "val": ["markit_housekeep(rot)"]
            })

        if not confidence_found:
            vec_list.append({
                "name": "confidence",
                "val": [0.8888]
            })

    def _calculate_smoothed_rotation(self, frames: Dict[str, Any], obj_id: str,
                                     current_frame: int, frame_list_sorted: List[int],
                                     current_idx: int) -> Optional[float]:
        """Calculate smoothed rotation using weighted average of 1-8 frames ahead.

        Args:
            frames: Frame data
            obj_id: Object ID
            current_frame: Current frame index
            frame_list_sorted: Sorted list of frames for this object
            current_idx: Index in frame_list_sorted

        Returns:
            Smoothed rotation angle in radians, or None if no movement detected
        """
        current_frame_str = str(current_frame)
        current_obj = frames[current_frame_str]["objects"][obj_id]
        current_rbbox = current_obj["object_data"]["rbbox"][0]["val"]
        x_current, y_current = current_rbbox[0], current_rbbox[1]

        angles = []
        weights = []

        for lookahead in range(1, 9):
            if current_idx + lookahead >= len(frame_list_sorted):
                break

            future_frame = frame_list_sorted[current_idx + lookahead]
            future_frame_str = str(future_frame)
            future_obj = frames[future_frame_str]["objects"][obj_id]
            future_rbbox = future_obj["object_data"]["rbbox"][0]["val"]
            x_future, y_future = future_rbbox[0], future_rbbox[1]

            delta_x = x_future - x_current
            delta_y = y_future - y_current

            if delta_x != 0 or delta_y != 0:
                angle = np.arctan2(delta_x, delta_y)
                angles.append(angle)
                weights.append(9 - lookahead)

        if not angles:
            return None

        weighted_sin = sum(np.sin(angle) * weight for angle, weight in zip(angles, weights))
        weighted_cos = sum(np.cos(angle) * weight for angle, weight in zip(angles, weights))
        weight_sum = sum(weights)

        avg_sin = weighted_sin / weight_sum
        avg_cos = weighted_cos / weight_sum

        return float(np.arctan2(avg_sin, avg_cos))

    def get_statistics(self) -> Dict[str, Any]:
        """Get rotation adjustment statistics.

        Returns:
            Dictionary with rotation adjustment statistics
        """
        return {
            'objects_processed': self.objects_processed,
            'rotations_adjusted': self.rotations_adjusted,
            'rotations_kept': self.rotations_kept,
            'rotations_copied': self.rotations_copied
        }


class SuddenPass(PostprocessingPass):
    """Detect sudden appearance/disappearance of objects near frame edges."""

    def __init__(self, edge_distance: int = 200):
        self.edge_distance = edge_distance
        self.sudden_appear_count = 0
        self.sudden_disappear_count = 0
        self.objects_with_events = set()

    def process(self, openlabel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect and record sudden appearance/disappearance events.

        Args:
            openlabel_data: Complete OpenLabel data structure

        Returns:
            Modified OpenLabel data with sudden events recorded
        """
        frames = openlabel_data.get("openlabel", {}).get("frames", {})
        objects = openlabel_data.get("openlabel", {}).get("objects", {})

        if not hasattr(self, 'frame_width') or not hasattr(self, 'frame_height'):
            logger.warning("SuddenPass: Video properties not set, skipping")
            return openlabel_data

        object_frame_map = defaultdict(list)

        for frame_idx_str, frame_data in frames.items():
            frame_idx = int(frame_idx_str)
            frame_objects = frame_data.get("objects", {})

            for obj_id_str in frame_objects.keys():
                object_frame_map[obj_id_str].append(frame_idx)

        frame_indices = sorted([int(f) for f in frames.keys()])
        if not frame_indices:
            return openlabel_data

        first_frame = frame_indices[0]
        last_frame = frame_indices[-1]

        for obj_id, frame_list in object_frame_map.items():
            frame_list_sorted = sorted(frame_list)

            sudden_appear_frames = []
            sudden_disappear_frames = []

            for i, frame_idx in enumerate(frame_list_sorted):
                if frame_idx == first_frame:
                    continue

                is_first_appearance = (i == 0)
                is_last_appearance = (i == len(frame_list_sorted) - 1)

                frame_str = str(frame_idx)
                frame_obj = frames[frame_str]["objects"][obj_id]
                rbbox = frame_obj["object_data"]["rbbox"][0]["val"]
                x, y, w, h, r = rbbox

                is_near_edge = self._is_near_edge(x, y, w, h)

                if is_first_appearance and frame_idx != first_frame and is_near_edge:
                    sudden_appear_frames.append(frame_idx)
                    self.sudden_appear_count += 1

                if is_last_appearance and frame_idx != last_frame and is_near_edge:
                    sudden_disappear_frames.append(frame_idx)
                    self.sudden_disappear_count += 1

            if sudden_appear_frames or sudden_disappear_frames:
                self.objects_with_events.add(obj_id)

                if obj_id not in objects:
                    continue

                if "object_data" not in objects[obj_id]:
                    objects[obj_id]["object_data"] = {}

                if "vec" not in objects[obj_id]["object_data"]:
                    objects[obj_id]["object_data"]["vec"] = []

                vec_list = objects[obj_id]["object_data"]["vec"]

                if sudden_appear_frames:
                    vec_list.append({
                        "name": "suddenappear",
                        "val": sudden_appear_frames
                    })

                if sudden_disappear_frames:
                    vec_list.append({
                        "name": "suddendisappear",
                        "val": sudden_disappear_frames
                    })

        return openlabel_data

    def _is_near_edge(self, x: float, y: float, w: float, h: float) -> bool:
        """Check if bounding box is near frame edge.

        Args:
            x: Center x coordinate
            y: Center y coordinate
            w: Width
            h: Height

        Returns:
            True if any part of bbox is within edge_distance of frame edge
        """
        x_min = x - w / 2
        x_max = x + w / 2
        y_min = y - h / 2
        y_max = y + h / 2

        near_left = x_min < self.edge_distance
        near_right = x_max > (self.frame_width - self.edge_distance)
        near_top = y_min < self.edge_distance
        near_bottom = y_max > (self.frame_height - self.edge_distance)

        return near_left or near_right or near_top or near_bottom

    def get_statistics(self) -> Dict[str, Any]:
        """Get sudden event statistics.

        Returns:
            Dictionary with sudden event statistics
        """
        return {
            'objects_with_events': len(self.objects_with_events),
            'sudden_appear_count': self.sudden_appear_count,
            'sudden_disappear_count': self.sudden_disappear_count
        }


class FrameIntervalPass(PostprocessingPass):
    """Add frame_intervals to objects based on their frame appearances."""

    def __init__(self):
        self.intervals_added = 0
        self.intervals_skipped_existing = 0
        self.intervals_skipped_no_frames = 0

    def process(self, openlabel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add frame_intervals to objects based on frame appearances.

        Args:
            openlabel_data: Complete OpenLabel data structure

        Returns:
            Modified OpenLabel data with frame_intervals added
        """
        frames = openlabel_data.get("openlabel", {}).get("frames", {})
        objects = openlabel_data.get("openlabel", {}).get("objects", {})

        object_frame_map = defaultdict(list)

        for frame_idx_str, frame_data in frames.items():
            frame_idx = int(frame_idx_str)
            frame_objects = frame_data.get("objects", {})

            for obj_id_str in frame_objects.keys():
                object_frame_map[obj_id_str].append(frame_idx)

        for obj_id, obj_data in objects.items():
            if "frame_intervals" in obj_data:
                self.intervals_skipped_existing += 1
                continue

            if obj_id not in object_frame_map or len(object_frame_map[obj_id]) == 0:
                self.intervals_skipped_no_frames += 1
                continue

            frame_list = sorted(object_frame_map[obj_id])
            frame_start = frame_list[0]
            frame_end = frame_list[-1]

            obj_data["frame_intervals"] = [
                {
                    "frame_start": frame_start,
                    "frame_end": frame_end
                }
            ]
            self.intervals_added += 1

        return openlabel_data

    def get_statistics(self) -> Dict[str, Any]:
        """Get frame interval addition statistics.

        Returns:
            Dictionary with frame interval statistics
        """
        return {
            'intervals_added': self.intervals_added,
            'intervals_skipped_existing': self.intervals_skipped_existing,
            'intervals_skipped_no_frames': self.intervals_skipped_no_frames
        }


class PostprocessingPipeline:
    """Manages and executes postprocessing passes on OpenLabel data."""

    def __init__(self):
        self.passes = []
        self.frame_width = None
        self.frame_height = None
        self.fps = None

    def set_video_properties(self, frame_width: int, frame_height: int, fps: float) -> None:
        """Set video properties for the pipeline.

        Args:
            frame_width: Video frame width in pixels
            frame_height: Video frame height in pixels
            fps: Video frames per second
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps

    def add_pass(self, postprocessing_pass: PostprocessingPass) -> None:
        """Add a postprocessing pass to the pipeline.

        Args:
            postprocessing_pass: Postprocessing pass instance
        """
        self.passes.append(postprocessing_pass)

    def execute(self, openlabel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all postprocessing passes in sequence.

        Args:
            openlabel_data: Complete OpenLabel data structure

        Returns:
            Processed OpenLabel data structure
        """
        if not self.passes:
            logger.info("No postprocessing passes configured")
            return openlabel_data

        logger.info(f"Running {len(self.passes)} postprocessing pass(es)...")

        processed_data = openlabel_data

        for i, pass_instance in enumerate(self.passes, 1):
            pass_name = pass_instance.__class__.__name__
            logger.info(f"  Pass {i}/{len(self.passes)}: {pass_name}")

            try:
                if self.frame_width and self.frame_height and self.fps:
                    pass_instance.set_video_properties(self.frame_width, self.frame_height, self.fps)

                processed_data = pass_instance.process(processed_data)
                stats = pass_instance.get_statistics()
                logger.info(f"    Statistics: {stats}")
            except Exception as e:
                logger.error(f"    Error in {pass_name}: {e}")
                raise

        logger.info("Postprocessing completed")
        return processed_data


class OpenLabelHandler:
    """Handles OpenLabel JSON structure creation and management."""

    def __init__(self, schema_path: str):
        """Initialize OpenLabel handler.

        Args:
            schema_path: Path to OpenLabel JSON schema file
        """
        self.schema_path = schema_path
        self.openlabel_data = {}
        self.initialize_from_schema()

    def initialize_from_schema(self) -> None:
        """Initialize OpenLabel structure from schema."""
        try:
            self.openlabel_data = self._create_empty_from_schema(self.schema_path)
            logger.info("OpenLabel structure initialized from schema")
        except Exception as e:
            logger.error(f"Failed to initialize from schema: {e}")
            raise

    def _create_empty_from_schema(self, schema_path: str) -> Dict[str, Any]:
        """Create empty OpenLabel structure based on schema."""
        with open(schema_path, 'r') as f:
            schema = json.load(f)

        def build_empty(schema: Dict[str, Any]) -> Any:
            if schema.get("type") == "object":
                result = {}
                properties = schema.get("properties", {})
                for key, prop_schema in properties.items():
                    if prop_schema.get("type") == "array":
                        result[key] = []
                    elif prop_schema.get("type") == "object":
                        result[key] = build_empty(prop_schema)
                    else:
                        result[key] = None
                return result
            elif schema.get("type") == "array":
                return []
            else:
                return None

        return build_empty(schema)

    def add_metadata(self, video_path: str) -> None:
        """Add metadata to OpenLabel structure.

        Args:
            video_path: Path to source video file
        """
        metadata = {
            "name": f"SAVANT Markit {__version__} Analysis",
            "annotator": Constants.ANNOTATOR_NAME,
            "comment": f"Multi-engine object detection and tracking analysis of {os.path.basename(video_path)}",
            "tags": ["object_detection", "tracking", "yolo", "optical_flow", "savant"],
            "schema_version": Constants.SCHEMA_VERSION
        }

        if "openlabel" in self.openlabel_data and "metadata" in self.openlabel_data["openlabel"]:
            self.openlabel_data["openlabel"]["metadata"].update(metadata)

        logger.info("Metadata added to OpenLabel structure")

    def add_frame_objects(self, frame_idx: int, detection_results: List[DetectionResult],
                         class_map: Dict[int, str]) -> None:
        """Add detected objects for a frame to OpenLabel structure (matching markit.py format).

        Args:
            frame_idx: Frame index
            detection_results: Detection results from engines
            class_map: Mapping of class IDs to names
        """
        if self.openlabel_data is None:
            raise RuntimeError("OpenLabel structure not initialized")

        frame_objects = {}
        seen_object = False

        for detection in detection_results:
            if detection.object_id is not None:
                obj_id_str = str(detection.object_id)
                seen_object = True

                # Convert detection data to YOLO-style xywhr format [center_x, center_y, width, height, rotation]
                xywhr_formatted = self._detection_to_xywhr(detection)

                # Map source engine to annotator name
                annotator_map = {
                    'yolo': 'markit_yolo',
                    'optical_flow': 'markit_oflow'
                }
                annotator_name = annotator_map.get(detection.source_engine, f"markit_{detection.source_engine}")

                # Add new object to global objects list if not seen before
                if obj_id_str not in self.openlabel_data["openlabel"]["objects"]:
                    self.openlabel_data["openlabel"]["objects"][obj_id_str] = {
                        "name": f"Object-{obj_id_str}",
                        "type": class_map.get(detection.class_id, str(detection.class_id)),
                        "ontology_uid": "0"
                    }

                # Add object data for this frame (matching original markit.py format exactly)
                frame_objects[obj_id_str] = {
                    "object_data": {
                        "rbbox": [{
                            "name": "shape",
                            "val": xywhr_formatted
                        }],
                        "vec": [
                            {
                                "name": "annotator",
                                "val": [annotator_name]
                            },
                            {
                                "name": "confidence",
                                "val": [float(detection.confidence)]
                            }
                        ]
                    }
                }

        # Only add frame if there are objects (matching original behavior)
        if seen_object:
            self.openlabel_data["openlabel"]["frames"][str(frame_idx)] = {
                "objects": frame_objects
            }

    def _detection_to_xywhr(self, detection: DetectionResult) -> List:
        """Convert DetectionResult to xywhr format [center_x, center_y, width, height, rotation].

        Args:
            detection: Detection result with oriented bbox

        Returns:
            List in xywhr format matching original markit.py
        """
        try:
            # Use the center and angle from detection
            center_x, center_y = detection.center

            # Calculate width and height from oriented bounding box points
            bbox_points = detection.oriented_bbox
            if len(bbox_points) >= 4:
                # Calculate distances between consecutive points to get width and height
                p1, p2, p3, p4 = bbox_points[0], bbox_points[1], bbox_points[2], bbox_points[3]

                # Distance between first two points (width or height)
                d1 = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                # Distance between second and third points (height or width)
                d2 = np.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2)

                # Assign width and height (larger dimension is usually width)
                width = max(d1, d2)
                height = min(d1, d2)
            else:
                # Fallback if bbox points are invalid
                width = height = 50.0

            # Use the angle from detection (convert to same format as original)
            rotation = float(detection.angle)

            # Format as integers for position/size, rounded float for rotation (matching original)
            xywhr_formatted = [
                int(center_x), int(center_y), int(width), int(height), round(rotation, 4)
            ]

            return xywhr_formatted

        except Exception as e:
            logger.error(f"Error converting detection to xywhr: {e}")
            # Return fallback values
            return [int(detection.center[0]), int(detection.center[1]), 50, 50, 0.0]

    def sort_objects(self) -> None:
        """Sort objects dictionary numerically by key (matching original markit.py)."""
        if self.openlabel_data is None:
            return

        try:
            # Sort the main objects dictionary numerically by key (matching original)
            objects = self.openlabel_data["openlabel"]["objects"]
            sorted_objects = dict(sorted(objects.items(), key=lambda item: int(item[0])))
            self.openlabel_data["openlabel"]["objects"] = sorted_objects
        except Exception as e:
            logger.error(f"Error sorting objects: {e}")

    def save_to_file(self, output_path: str) -> None:
        """Save OpenLabel data to JSON file.

        Args:
            output_path: Output file path
        """
        try:
            self.sort_objects()

            with open(output_path, 'w') as f:
                json.dump(self.openlabel_data, f, indent=2)

            logger.info(f"OpenLabel data saved to: {output_path}")

        except Exception as e:
            logger.error(f"Failed to save OpenLabel data: {e}")
            raise


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
  # YOLO only (default)
  python markit.py --weights model.pt --input video.mp4 --output_json output.json --schema schema.json

  # Optical flow only
  python markit.py --detection-method optical_flow --input video.mp4 --output_json output.json --schema schema.json

  # Both engines with default IoU threshold (0.3)
  python markit.py --detection-method both --weights model.pt --input video.mp4 --output_json output.json --schema schema.json

  # Both engines with custom IoU threshold
  python markit.py --detection-method both --weights model.pt --input video.mp4 --output_json output.json --schema schema.json --iou-threshold 0.5

  # Both engines without conflict resolution
  python markit.py --detection-method both --weights model.pt --input video.mp4 --output_json output.json --schema schema.json --disable-conflict-resolution
        """
    )

    # Required arguments
    parser.add_argument('--weights', help='Path to YOLO weights file (.pt)')
    parser.add_argument('--input', required=True, help='Path to input video file')
    parser.add_argument('--output_json', required=True, help='Path to output OpenLabel JSON file')
    parser.add_argument('--schema', required=True, help='Path to OpenLabel JSON schema file')
    parser.add_argument('--output_video', help='Path to output annotated video file (optional)')

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
    parser.add_argument('--rotation-threshold', type=float, default=0.01,
                       help='Rotation angle threshold in radians for adjustment (default: 0.01)')
    parser.add_argument('--edge-distance', type=int, default=200,
                       help='Distance in pixels from frame edge for sudden appear/disappear detection (default: 200)')

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

            # Annotate and write frame if output video requested
            if config.output_video_path:
                annotated_frame = FrameAnnotator.annotate_frame(frame, detection_results)
                video_processor.write_frame(annotated_frame)

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

        logger.info(f"Markit v{__version__} starting with engines: {', '.join(engines)}")
        if config.enable_conflict_resolution and len(engines) > 1:
            logger.info(f"Conflict resolution enabled with IoU threshold: {config.iou_threshold:.2f}")

        # Initialize components
        video_processor = VideoProcessor(config)
        openlabel_handler = OpenLabelHandler(config.schema_path)

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
            postprocessing_pipeline.add_pass(GapDetectionPass())
            postprocessing_pipeline.add_pass(GapFillingPass())
            postprocessing_pipeline.add_pass(
                DuplicateRemovalPass(
                    avg_iou_threshold=config.duplicate_avg_iou,
                    min_iou_threshold=config.duplicate_min_iou
                )
            )
            postprocessing_pipeline.add_pass(
                RotationAdjustmentPass(rotation_threshold=config.rotation_threshold)
            )
            postprocessing_pipeline.add_pass(SuddenPass(edge_distance=config.edge_distance))
            postprocessing_pipeline.add_pass(FrameIntervalPass())

            openlabel_handler.openlabel_data = postprocessing_pipeline.execute(
                openlabel_handler.openlabel_data
            )
        else:
            logger.info("Housekeeping disabled, skipping postprocessing")

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