"""
engines - Object detection engines

Contains YOLO, optical flow, and background subtraction detection engines.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from ..config import DetectionResult, OpticalFlowParams

logger = logging.getLogger(__name__)


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
