"""
handler - OpenLabel JSON structure handling

Manages OpenLabel data structure creation, frame object addition, and file saving.
"""

import json
import logging
import os
from typing import Any, Dict, List

import numpy as np

from ..config import Constants, DetectionResult
from ..utils import normalize_angle_to_half_pi

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy data types and rounds floats to 4 decimals."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return round(float(obj), 4)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

    def encode(self, obj):
        """Override encode to round all Python floats to 4 decimals."""
        if isinstance(obj, float):
            return format(round(obj, 4), '.4f').rstrip('0').rstrip('.')
        return super().encode(obj)

    def iterencode(self, obj, _one_shot=False):
        """Override iterencode to round floats in nested structures."""
        def round_floats(o):
            if isinstance(o, float):
                return round(o, 4)
            elif isinstance(o, dict):
                return {k: round_floats(v) for k, v in o.items()}
            elif isinstance(o, (list, tuple)):
                return [round_floats(item) for item in o]
            return o

        return super().iterencode(round_floats(obj), _one_shot)

# Get version from config
__version__ = '2.0.0'


class OpenLabelHandler:
    """Handles OpenLabel JSON structure creation and management."""

    def __init__(self, schema_path: str, verbose: bool = False):
        """Initialize OpenLabel handler.

        Args:
            schema_path: Path to OpenLabel JSON schema file
            verbose: Enable verbose logging for angle/detection analysis
        """
        self.schema_path = schema_path
        self.verbose = verbose
        self.openlabel_data = {}
        self.debug_data = {}  # Separate structure for debug info (verbose mode only)
        self._debug_frame_count = 0
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

                # Convert detection data to OpenLabel-style xywhr format [center_x, center_y, width, height, rotation]
                xywhr_formatted = self._detection_to_xywhr(detection, frame_idx)

                # Map source engine to annotator name
                annotator_map = {
                    'yolo': 'markit_yolo',
                    'optical_flow': 'markit_oflow',
                    'aruco': 'markit_aruco'
                }
                annotator_name = annotator_map.get(detection.source_engine, f"markit_{detection.source_engine}")

                # Check if this is an ArUco marker with GPS data
                is_aruco = hasattr(detection, 'aruco_id') and hasattr(detection, 'gps_data')

                # Add new object to global objects list if not seen before
                if obj_id_str not in self.openlabel_data["openlabel"]["objects"]:
                    if is_aruco:
                        # ArUco marker - use special naming and type
                        object_name = getattr(detection, 'aruco_name', f"ArUco_{detection.aruco_id}")
                        self.openlabel_data["openlabel"]["objects"][obj_id_str] = {
                            "name": object_name,
                            "type": "ArUco",
                            "ontology_uid": "0"
                        }
                    else:
                        # Regular detection
                        self.openlabel_data["openlabel"]["objects"][obj_id_str] = {
                            "name": f"Object-{obj_id_str}",
                            "type": class_map.get(detection.class_id, str(detection.class_id)),
                            "ontology_uid": "0"
                        }

                # Build vec data based on detection type
                vec_data = [
                    {
                        "name": "annotator",
                        "val": [annotator_name]
                    },
                    {
                        "name": "confidence",
                        "val": [float(detection.confidence)]
                    }
                ]

                # If verbose, store original YOLO dimensions in separate debug structure
                if self.verbose and detection.width is not None and detection.height is not None:
                    # Ensure frame exists in debug_data
                    if frame_idx not in self.debug_data:
                        self.debug_data[frame_idx] = {}

                    # Store raw xywhr for this object
                    self.debug_data[frame_idx][obj_id_str] = {
                        "raw_xywhr": [
                            float(detection.center[0]),
                            float(detection.center[1]),
                            float(detection.width),
                            float(detection.height),
                            float(detection.angle)
                        ]
                    }

                # Add GPS data for ArUco markers
                if is_aruco:
                    gps_data = detection.gps_data
                    aruco_id = detection.aruco_id

                    # Build corner identifiers and GPS coordinates
                    corner_ids = []
                    latitudes = []
                    longitudes = []

                    # Sort corners alphabetically (a, b, c, d) for consistent output
                    for corner in sorted(gps_data.keys()):
                        corner_data = gps_data[corner]
                        corner_ids.append(f"{aruco_id}{corner}")
                        latitudes.append(str(corner_data['lat']))
                        longitudes.append(str(corner_data['lon']))

                    # Add ArUco-specific vectors (only if GPS data exists)
                    if corner_ids:
                        vec_data.extend([
                            {
                                "name": "arucoID",
                                "val": corner_ids
                            },
                            {
                                "name": "lat",
                                "val": latitudes
                            },
                            {
                                "name": "long",
                                "val": longitudes
                            }
                        ])

                        # Add description (base name from CSV filename)
                        description = getattr(detection, 'aruco_name', '').split('_')[0]
                        if description:
                            vec_data.append({
                                "name": "description",
                                "val": description
                            })

                # Add object data for this frame
                frame_objects[obj_id_str] = {
                    "object_data": {
                        "rbbox": [{
                            "name": "shape",
                            "val": xywhr_formatted
                        }],
                        "vec": vec_data
                    }
                }

        # Only add frame if there are objects (matching original behavior)
        if seen_object:
            self.openlabel_data["openlabel"]["frames"][str(frame_idx)] = {
                "objects": frame_objects
            }

    def _detection_to_xywhr(self, detection: DetectionResult, frame_idx: int) -> List:
        """Convert DetectionResult to OpenLabel xywhr format [center_x, center_y, width, height, rotation].

        Converts from YOLO/OpenCV convention (minimum area rectangle with angle in [0, π/2))
        to OpenLabel convention (width=horizontal extent, height=vertical extent, [-π/4, π/4)).

        OpenLabel convention (I think, but it is not well documented):
        - width (w) is always the horizontal (x-direction) extent
        - height (h) is always the vertical (y-direction) extent
        - rotation (r) is right-handed from x to y axis (positive = clockwise in image coordinates)

        Args:
            detection: Detection result with oriented bbox
            frame_idx: Frame index for logging

        Returns:
            List in xywhr format: [center_x, center_y, width, height, rotation]
        """
        try:
            # Use the center from detection
            center_x, center_y = detection.center

            # Get the oriented bounding box corners
            bbox_points = detection.oriented_bbox

            if len(bbox_points) >= 4:
                # Corners from YOLO: p0, p1, p2, p3
                # Edge p0->p1 is the "width" edge in YOLO's minimum-area-rectangle
                # Edge p1->p2 is the "height" edge in YOLO's minimum-area-rectangle
                p0, p1, p2, p3 = bbox_points[0], bbox_points[1], bbox_points[2], bbox_points[3]

                # Calculate edge vectors and their properties
                edge_01 = p1 - p0  # YOLO's "width" edge
                edge_12 = p2 - p1  # YOLO's "height" edge

                # Calculate lengths
                len_01 = np.sqrt(edge_01[0]**2 + edge_01[1]**2)
                len_12 = np.sqrt(edge_12[0]**2 + edge_12[1]**2)

                # Calculate angles of these edges (arctan2 gives angle in [-π, π])
                angle_01 = np.arctan2(edge_01[1], edge_01[0])
                angle_12 = np.arctan2(edge_12[1], edge_12[0])

                # Determine which edge is more horizontal (for OpenLabel width)
                # An edge is more "horizontal" if its angle is closer to 0
                # Use |cos(angle)| as a measure of horizontalness (1.0 = perfectly horizontal)
                horizontalness_01 = abs(np.cos(angle_01))
                horizontalness_12 = abs(np.cos(angle_12))

                if horizontalness_01 >= horizontalness_12:
                    # Edge 01 is more horizontal -> use as OpenLabel width
                    width = len_01
                    height = len_12
                    rotation = angle_01
                else:
                    # Edge 12 is more horizontal -> use as OpenLabel width
                    # When swapping dimensions, subtract 90° (π/2) from YOLO's angle
                    width = len_12
                    height = len_01
                    rotation = angle_01 - np.pi / 2

                # Normalize rotation to [-π/2, π/2] range for OpenLabel (should not be needed if yolo output is indeed between 0 and π/2 but just in case)
                rotation = normalize_angle_to_half_pi(rotation)
            else:
                # Fallback if bbox points are invalid
                width = height = 50.0
                rotation = 0.0

            # Verbose logging for angle analysis (first 1000 detections)
            if self.verbose and self._debug_frame_count < 1000:
                # Show YOLO original values vs OpenLabel converted values
                yolo_w = detection.width if detection.width is not None else 0.0
                yolo_h = detection.height if detection.height is not None else 0.0
                yolo_r = detection.angle if detection.angle is not None else 0.0

                logger.info(f"Frame {frame_idx}: obj_id={detection.object_id}, "
                           f"YOLO(w={yolo_w:.1f}, h={yolo_h:.1f}, r={yolo_r:.4f}rad/{np.degrees(yolo_r):.1f}°) -> "
                           f"OpenLabel(w={width:.1f}, h={height:.1f}, r={rotation:.4f}rad/{np.degrees(rotation):.1f}°), "
                           f"source={detection.source_engine}")
                self._debug_frame_count += 1

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
                json.dump(self.openlabel_data, f, indent=2, cls=NumpyEncoder)

            logger.info(f"OpenLabel data saved to: {output_path}")

        except Exception as e:
            logger.error(f"Failed to save OpenLabel data: {e}")
            raise
