"""
engines - Object detection engines

Contains YOLO, optical flow, background subtraction, and ArUco marker detection engines.
"""

import csv
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

    def __init__(self, weights_path: str, class_map: Optional[Dict[int, str]] = None, verbose: bool = False):
        """Initialize YOLO engine.

        Args:
            weights_path: Path to YOLO weights file
            class_map: Expected class mapping from ontology (for validation)
            verbose: Enable verbose logging
        """
        self.weights_path = weights_path
        self.model = None
        self.class_map = class_map
        self.verbose = verbose
        # Track semantic dimensions and angles per object for continuity
        self.object_tracking: Dict[int, Dict[str, float]] = {}
        # Structure: {obj_id: {'w_sem': float, 'h_sem': float, 'angle': float}}
        self._initialize()

    def _initialize(self) -> None:
        """Initialize YOLO model."""
        try:
            logger.info(f"Loading YOLO model from {self.weights_path}")
            self.model = YOLO(self.weights_path)
            self.model.verbose = False

            # Validate model classes against ontology
            self._validate_model_classes()

        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {e}")
            raise

    def _validate_model_classes(self) -> None:
        """Validate YOLO model class names against ontology class map.

        Logs warnings if there are mismatches that could indicate using
        an old/wrong model trained with different class labels.
        """
        if not self.class_map or not self.model or not hasattr(self.model, 'names'):
            return

        model_names = self.model.names  # Dict: {class_id: class_name}

        # Check number of classes
        num_model_classes = len(model_names)
        num_ontology_classes = len([k for k in self.class_map.keys() if k < 100])  # Exclude high UIDs like ArUco

        if num_model_classes != num_ontology_classes:
            logger.warning(f"⚠️  Class count mismatch: YOLO model has {num_model_classes} classes, "
                          f"ontology has {num_ontology_classes} classes (UID < 100)")
            logger.warning("This may indicate using a model trained with different labels!")

        # Compare class names for matching IDs
        mismatches = []
        for class_id, ontology_label in self.class_map.items():
            if class_id >= 100:  # Skip high UIDs like ArUco markers
                continue
            if class_id in model_names:
                model_label = model_names[class_id]
                # Case-insensitive comparison
                if model_label.lower() != ontology_label.lower():
                    mismatches.append((class_id, model_label, ontology_label))

        if mismatches:
            logger.warning(f"⚠️  Found {len(mismatches)} class name mismatches between YOLO model and ontology:")
            for class_id, model_label, ontology_label in mismatches[:10]:  # Show first 10
                logger.warning(f"  Class {class_id}: model='{model_label}' vs ontology='{ontology_label}'")
            if len(mismatches) > 10:
                logger.warning(f"  ... and {len(mismatches) - 10} more mismatches")
            logger.warning("⚠️  This indicates the YOLO model was trained with different class labels!")
            logger.warning("⚠️  Detection results will have incorrect class labels!")

        # If verbose, log all model classes
        if self.verbose and model_names:
            logger.info(f"YOLO model classes ({len(model_names)} total):")
            for class_id in sorted(model_names.keys())[:25]:  # First 25
                ontology_label = self.class_map.get(class_id, "N/A")
                match = "✓" if model_names[class_id].lower() == ontology_label.lower() else "✗"
                logger.info(f"  {match} Class {class_id:3d}: model='{model_names[class_id]:20s}' ontology='{ontology_label}'")
            if len(model_names) > 25:
                logger.info(f"  ... and {len(model_names) - 25} more classes")

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
                        # Extract YOLO format
                        center_x, center_y, w_yolo, h_yolo, r_yolo = xywhr

                        # Convert YOLO format to semantic format with continuity
                        w_sem, h_sem, continuous_angle = self._yolo_to_semantic(
                            obj_id, w_yolo, h_yolo, r_yolo
                        )

                        # Calculate corner points using semantic dimensions
                        cos_r = np.cos(continuous_angle)
                        sin_r = np.sin(continuous_angle)

                        hw = w_sem / 2
                        hh = h_sem / 2

                        # Corners with long axis along x in local coordinates
                        corners = np.array([
                            [-hw, -hh],
                            [hw, -hh],
                            [hw, hh],
                            [-hw, hh]
                        ])

                        rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
                        rotated_corners = corners @ rotation_matrix.T
                        oriented_bbox = rotated_corners + np.array([center_x, center_y])

                        detection_results.append(DetectionResult(
                            object_id=int(obj_id) if obj_id is not None else None,
                            class_id=int(cls),
                            confidence=float(conf),
                            oriented_bbox=oriented_bbox,
                            center=(float(center_x), float(center_y)),
                            angle=float(continuous_angle),  # Continuous semantic angle
                            source_engine='yolo',
                            width=float(w_sem),   # Semantic long axis
                            height=float(h_sem)   # Semantic short axis
                        ))

            return detection_results

        except Exception as e:
            logger.error(f"Error processing frame with YOLO: {e}")
            return []

    def _yolo_to_semantic(self, obj_id: Optional[int],
                          w_yolo: float, h_yolo: float, r_yolo: float) -> Tuple[float, float, float]:
        """Convert YOLO (w, h, angle) to semantic (w_sem, h_sem, continuous_angle).

        Handles YOLO's w/h swapping by adjusting angle instead of swapping dimensions.
        Maintains continuity with previous detections for same object ID.

        Args:
            obj_id: Object tracking ID (None for first detection)
            w_yolo: YOLO width (may swap between frames)
            h_yolo: YOLO height (may swap between frames)
            r_yolo: YOLO rotation in [0, π/2)

        Returns:
            (w_sem, h_sem, continuous_angle) where w_sem is always long axis
        """
        from ..utils import find_continuous_angle, rebase_angle_if_needed

        # First detection for this object
        if obj_id is None or obj_id not in self.object_tracking:
            # Establish semantic dimensions: long axis = width
            if w_yolo >= h_yolo:
                w_sem = w_yolo
                h_sem = h_yolo
                base_angle = r_yolo
            else:
                # YOLO's h is longer - make it our semantic width
                w_sem = h_yolo
                h_sem = w_yolo
                base_angle = r_yolo + np.pi/2

            continuous_angle = base_angle

            # Store tracking data if we have an ID
            if obj_id is not None:
                self.object_tracking[obj_id] = {
                    'w_sem': w_sem,
                    'h_sem': h_sem,
                    'angle': continuous_angle
                }

            return w_sem, h_sem, continuous_angle

        # Subsequent detection - maintain continuity
        prev_data = self.object_tracking[obj_id]
        w_sem_prev = prev_data['w_sem']
        h_sem_prev = prev_data['h_sem']
        angle_prev = prev_data['angle']

        # Detect if YOLO swapped w and h by comparing with previous semantic dims
        # Use absolute difference to find best match
        error_no_swap = abs(w_yolo - w_sem_prev) + abs(h_yolo - h_sem_prev)
        error_swap = abs(w_yolo - h_sem_prev) + abs(h_yolo - w_sem_prev)

        if error_no_swap <= error_swap:
            # No swap: YOLO's w → semantic w
            base_angle = r_yolo
            w_raw = w_yolo
            h_raw = h_yolo
        else:
            # Swap detected: YOLO's w → semantic h, adjust angle by π/2
            base_angle = r_yolo + np.pi/2
            w_raw = h_yolo
            h_raw = w_yolo

        # Apply continuity: find closest equivalent angle considering π/2 ambiguity
        continuous_angle = find_continuous_angle(base_angle, angle_prev, np.pi/2)
        continuous_angle = rebase_angle_if_needed(continuous_angle)

        # Update semantic dimensions with smoothing (allows gradual aspect ratio changes)
        alpha = 0.7  # Weight for previous value
        w_sem = alpha * w_sem_prev + (1 - alpha) * w_raw
        h_sem = alpha * h_sem_prev + (1 - alpha) * h_raw

        # Update tracking
        self.object_tracking[obj_id] = {
            'w_sem': w_sem,
            'h_sem': h_sem,
            'angle': continuous_angle
        }

        return w_sem, h_sem, continuous_angle

    def cleanup(self) -> None:
        """Clean up YOLO engine resources."""
        self.model = None
        self.object_tracking.clear()


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

                    # Convert angle from degrees to radians (cv2.minAreaRect returns degrees)
                    angle_rad = np.radians(angle)

                    results.append(DetectionResult(
                        object_id=obj_id,
                        class_id=0,  # Generic "moving object" class
                        confidence=confidence,
                        oriented_bbox=box.astype(np.float32),
                        center=(center_x, center_y),
                        angle=angle_rad,
                        source_engine='optical_flow',
                        width=float(width),
                        height=float(height)
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


class ArUcoGPSData:
    """Parses and stores ArUco marker GPS position data from CSV file."""

    def __init__(self, csv_path: str):
        """Initialize ArUco GPS data parser.

        Args:
            csv_path: Path to CSV file with ArUco GPS positions

        CSV format:
            point_name,latitude,longitude,altitude,...
            aruco_24a,57.74281389447574,12.894721471928605,...
            aruco_24c,57.74280961617389,12.89475703176673,...

        Naming convention:
            aruco_<ID><corner> where corner is a, b, c, or d
            a = bottom-left, b = top-left, c = top-right, d = bottom-right
        """
        self.csv_path = Path(csv_path)
        self.gps_data = {}  # {aruco_id: {corner: {'lat': ..., 'lon': ...}}}
        self.base_name = self._extract_base_name()
        self._load_csv()

    def _extract_base_name(self) -> str:
        """Extract base name from CSV filename (before first underscore).

        Returns:
            Base name (e.g., 'Zurich' from 'Zurich_markers.csv')
        """
        filename = self.csv_path.stem  # Get filename without extension
        parts = filename.split('_')
        return parts[0] if parts else filename

    def _load_csv(self) -> None:
        """Load and parse CSV file."""
        try:
            with open(self.csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    point_name = row.get('point_name', '').strip()

                    # Parse point_name: aruco_24a -> ID=24, corner='a'
                    if not point_name.startswith('aruco_'):
                        continue

                    # Extract ID and corner
                    suffix = point_name[6:]  # Remove 'aruco_' prefix
                    if not suffix:
                        continue

                    # Corner is last character (a, b, c, d)
                    corner = suffix[-1].lower()
                    if corner not in ['a', 'b', 'c', 'd']:
                        logger.warning(f"Invalid corner '{corner}' in point_name: {point_name}")
                        continue

                    # ID is everything before the corner
                    try:
                        aruco_id = int(suffix[:-1])
                    except ValueError:
                        logger.warning(f"Invalid ArUco ID in point_name: {point_name}")
                        continue

                    # Extract GPS coordinates
                    try:
                        lat = float(row.get('latitude', 0))
                        lon = float(row.get('longitude', 0))
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid GPS coordinates for {point_name}")
                        continue

                    # Store data
                    if aruco_id not in self.gps_data:
                        self.gps_data[aruco_id] = {}

                    self.gps_data[aruco_id][corner] = {
                        'lat': lat,
                        'lon': lon
                    }

            logger.info(f"Loaded GPS data for {len(self.gps_data)} ArUco markers from {self.csv_path}")

        except Exception as e:
            logger.error(f"Failed to load ArUco GPS CSV: {e}")
            raise

    def get_gps_data(self, aruco_id: int) -> Dict[str, Dict[str, float]]:
        """Get GPS data for specific ArUco ID.

        Args:
            aruco_id: ArUco marker ID

        Returns:
            Dictionary of corner positions: {corner: {'lat': ..., 'lon': ...}}
            Empty dict if ID not found
        """
        return self.gps_data.get(aruco_id, {})

    def get_object_name(self, aruco_id: int) -> str:
        """Get OpenLabel object name for ArUco marker.

        Args:
            aruco_id: ArUco marker ID

        Returns:
            Object name in format: "BaseName_ID" (e.g., "Zurich_24")
        """
        return f"{self.base_name}_{aruco_id}"


class ArUcoEngine(BaseDetectionEngine):
    """ArUco marker detection engine with GPS position integration."""

    def __init__(self, csv_path: str, class_id: int):
        """Initialize ArUco detection engine.

        Args:
            csv_path: Path to GPS CSV file with ArUco positions
            class_id: Class ID for ArUco markers from ontology
        """
        self.csv_path = csv_path
        self.class_id = class_id
        self.gps_data = None
        self.aruco_dict = None
        self.aruco_params = None
        self.tracker = SimpleTracker(id_offset=2000000)  # Offset to avoid conflicts
        self._initialize()

    def _initialize(self) -> None:
        """Initialize ArUco detector and GPS data."""
        try:
            # Load GPS data
            logger.info(f"Loading ArUco GPS data from {self.csv_path}")
            self.gps_data = ArUcoGPSData(self.csv_path)

            # Initialize ArUco detector with DICT_4X4_50
            logger.info("Initializing ArUco detector with DICT_4X4_50")
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            self.aruco_params = cv2.aruco.DetectorParameters()

            logger.info("ArUco detection engine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ArUco engine: {e}")
            raise

    def process_frame(self, frame: np.ndarray) -> List[DetectionResult]:
        """Process frame to detect ArUco markers.

        Args:
            frame: Input frame

        Returns:
            List of ArUco detection results with GPS data
        """
        try:
            results = []

            # Convert to grayscale for better detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect ArUco markers
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params
            )

            # Process each detected marker
            if ids is not None:
                for i, aruco_id in enumerate(ids.flatten()):
                    aruco_id = int(aruco_id)
                    marker_corners = corners[i][0]  # Shape: (4, 2)

                    # Get GPS data for this marker (may be empty)
                    gps_data = self.gps_data.get_gps_data(aruco_id)

                    if not gps_data:
                        logger.warning(f"ArUco marker {aruco_id} detected but not found in GPS CSV")

                    # Calculate center point
                    center_x = float(np.mean(marker_corners[:, 0]))
                    center_y = float(np.mean(marker_corners[:, 1]))

                    # Get or assign tracking ID
                    obj_id = self.tracker.get_id((center_x, center_y))

                    # Calculate rotation angle from corner positions
                    # Use vector from bottom-left to bottom-right corner
                    angle = float(np.arctan2(
                        marker_corners[1, 1] - marker_corners[0, 1],
                        marker_corners[1, 0] - marker_corners[0, 0]
                    ))

                    # Calculate confidence based on corner detection quality
                    # Use perimeter consistency as quality metric
                    edge_lengths = []
                    for j in range(4):
                        p1 = marker_corners[j]
                        p2 = marker_corners[(j + 1) % 4]
                        edge_lengths.append(np.linalg.norm(p2 - p1))

                    # Confidence is higher when edges are more uniform
                    avg_length = np.mean(edge_lengths)
                    length_variance = np.var(edge_lengths)
                    # Normalize variance to 0-1 range and invert (lower variance = higher confidence)
                    confidence = float(max(0.5, min(1.0, 1.0 - (length_variance / (avg_length ** 2)))))

                    # Create detection result with GPS data attached
                    detection = DetectionResult(
                        object_id=obj_id,
                        class_id=self.class_id,
                        confidence=confidence,
                        oriented_bbox=marker_corners.astype(np.float32),
                        center=(center_x, center_y),
                        angle=angle,
                        source_engine='aruco'
                    )

                    # Attach GPS data and ArUco ID for OpenLabel export
                    detection.aruco_id = aruco_id
                    detection.gps_data = gps_data
                    detection.aruco_name = self.gps_data.get_object_name(aruco_id)

                    results.append(detection)

            return results

        except Exception as e:
            logger.error(f"Error processing frame with ArUco detector: {e}")
            return []

    def cleanup(self) -> None:
        """Clean up ArUco engine resources."""
        self.aruco_dict = None
        self.aruco_params = None
        self.gps_data = None
