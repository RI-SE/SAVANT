"""
handler - OpenLabel JSON structure handling

Manages OpenLabel data structure creation, frame object addition, and file saving.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import Constants, DetectionResult
from ..utils import normalize_angle_to_2pi_range
from .. import __version__

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
        self._class_translation_count = 0  # Counter for verbose class translation logging
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
            "schema_version": Constants.SCHEMA_VERSION,
            "tagged_file": os.path.basename(video_path)
        }

        if "openlabel" in self.openlabel_data and "metadata" in self.openlabel_data["openlabel"]:
            self.openlabel_data["openlabel"]["metadata"].update(metadata)

        logger.info("Metadata added to OpenLabel structure")

    def set_ontology(self, ontology_uri: str) -> None:
        """Set the ontology URI in the ontologies section.

        Args:
            ontology_uri: The ontology namespace URI to use
        """
        if ontology_uri:
            if "openlabel" in self.openlabel_data and "ontologies" in self.openlabel_data["openlabel"]:
                self.openlabel_data["openlabel"]["ontologies"]["0"] = ontology_uri
                logger.info(f"Ontology set: {ontology_uri}")
            else:
                logger.warning("Could not set ontology: openlabel.ontologies not found in structure")

    def add_aruco_objects(self, gps_data: Dict[int, Dict], csv_name: str,
                          id_mapping: Optional[Dict[int, int]] = None) -> None:
        """Pre-populate all ArUco markers from GPS data.

        Adds all markers from the CSV file to objects, even if not detected in video.
        This allows manual annotation tools to attach bounding boxes to pre-defined markers.

        Args:
            gps_data: Dict mapping aruco_id -> {corner: {'lat', 'lon', 'alt'}}
            csv_name: CSV filename without extension (used for description field)
            id_mapping: Optional mapping of physical ArUco ID to sequential object ID.
                        If None, uses legacy 2000000 + aruco_id scheme.
        """
        for aruco_id, corners in gps_data.items():
            # Use mapping if provided, otherwise legacy scheme
            if id_mapping and aruco_id in id_mapping:
                obj_id_str = str(id_mapping[aruco_id])
            else:
                obj_id_str = str(2000000 + aruco_id)
            object_name = f"aruco_{aruco_id}"

            # Build GPS vectors from all corners
            corner_ids, latitudes, longitudes, altitudes = [], [], [], []
            for corner in sorted(corners.keys()):
                corner_data = corners[corner]
                corner_ids.append(f"{aruco_id}{corner}")
                latitudes.append(str(corner_data['lat']))
                longitudes.append(str(corner_data['lon']))
                altitudes.append(str(corner_data.get('alt', 0)))

            # Build object with GPS data at object level
            aruco_object = {
                "name": object_name,
                "type": "ArUco",
                "ontology_uid": "0"
            }

            # Add object_data.vec with GPS coordinates
            if corner_ids:
                aruco_object["object_data"] = {
                    "vec": [
                        {"name": "arucoID", "val": corner_ids},
                        {"name": "long", "val": longitudes},
                        {"name": "lat", "val": latitudes},
                        {"name": "alt", "val": altitudes},
                        {"name": "description", "val": csv_name}
                    ]
                }

            self.openlabel_data["openlabel"]["objects"][obj_id_str] = aruco_object

        logger.info(f"Pre-populated {len(gps_data)} ArUco markers from GPS data")

    def add_visual_marker_objects(self, gps_data: Dict[int, Dict],
                                   marker_names: Dict[int, str],
                                   id_mapping: Dict[int, int],
                                   csv_name: str) -> None:
        """Pre-populate visual markers from GPS data.

        Adds all visual markers to objects list. These are reference points like
        lampposts, features, etc. that are not automatically detected but have
        known GPS coordinates.

        Args:
            gps_data: Dict mapping marker_id -> {corner: {'lat', 'lon', 'alt'}}
            marker_names: Dict mapping marker_id -> description (e.g., 'lamppost')
            id_mapping: Mapping of marker ID to sequential object ID
            csv_name: CSV filename without extension (used for description field)
        """
        for marker_id, corners in gps_data.items():
            obj_id_str = str(id_mapping[marker_id])
            marker_type = marker_names.get(marker_id, 'marker')
            object_name = f"{marker_type}_{marker_id}"

            # Build GPS vectors from all corners
            corner_ids, latitudes, longitudes, altitudes = [], [], [], []
            for corner in sorted(corners.keys()):
                corner_data = corners[corner]
                corner_ids.append(f"{marker_id}{corner}")
                latitudes.append(str(corner_data['lat']))
                longitudes.append(str(corner_data['lon']))
                altitudes.append(str(corner_data.get('alt', 0)))

            # Build object with GPS data at object level
            marker_object = {
                "name": object_name,
                "type": "VisualMarker",
                "ontology_uid": "0"
            }

            # Add object_data.vec with GPS coordinates
            if corner_ids:
                marker_object["object_data"] = {
                    "vec": [
                        {"name": "markerID", "val": corner_ids},
                        {"name": "long", "val": longitudes},
                        {"name": "lat", "val": latitudes},
                        {"name": "alt", "val": altitudes},
                        {"name": "description", "val": csv_name}
                    ]
                }

            self.openlabel_data["openlabel"]["objects"][obj_id_str] = marker_object

        logger.info(f"Pre-populated {len(gps_data)} visual markers from GPS data")

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
                        # ArUco marker - use special naming and type with GPS at object level
                        object_name = getattr(detection, 'aruco_name', f"ArUco_{detection.aruco_id}")
                        gps_data = detection.gps_data
                        aruco_id = detection.aruco_id

                        # Build GPS vectors from all corners
                        corner_ids, latitudes, longitudes, altitudes = [], [], [], []
                        for corner in sorted(gps_data.keys()):
                            corner_data = gps_data[corner]
                            corner_ids.append(f"{aruco_id}{corner}")
                            latitudes.append(str(corner_data['lat']))
                            longitudes.append(str(corner_data['lon']))
                            altitudes.append(str(corner_data.get('alt', 0)))

                        description = object_name.split('_')[0]

                        # Build object with GPS data at object level (not per-frame)
                        aruco_object = {
                            "name": object_name,
                            "type": "ArUco",
                            "ontology_uid": "0"
                        }

                        # Add object_data.vec with GPS coordinates if available
                        if corner_ids:
                            aruco_object["object_data"] = {
                                "vec": [
                                    {"name": "arucoID", "val": corner_ids},
                                    {"name": "long", "val": longitudes},
                                    {"name": "lat", "val": latitudes},
                                    {"name": "alt", "val": altitudes},
                                    {"name": "description", "val": description}
                                ]
                            }

                        self.openlabel_data["openlabel"]["objects"][obj_id_str] = aruco_object
                    else:
                        # Regular detection - translate YOLO class ID to ontology label
                        class_label = class_map.get(detection.class_id, str(detection.class_id))

                        # Verbose logging for first 25 class translations
                        if self.verbose and self._class_translation_count < 25:
                            logger.info(f"Class translation: YOLO class_id={detection.class_id} → "
                                       f"ontology_label='{class_label}' (source={detection.source_engine})")
                            self._class_translation_count += 1

                        self.openlabel_data["openlabel"]["objects"][obj_id_str] = {
                            "name": f"Object-{obj_id_str}",
                            "type": class_label,
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

                # Add object data for this frame
                # ArUco: only rbbox + basic vec (GPS data is at object level)
                # Regular objects: full vec data
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
        """Convert DetectionResult to OpenLabel xywhr format.

        New convention:
        - width/height are semantic (long/short axis) and never swap
        - rotation is continuous, normalized to [0, 2π) for output
        - positive x-axis is 0 radians

        Args:
            detection: Detection result with oriented bbox
            frame_idx: Frame index for logging

        Returns:
            List in xywhr format: [center_x, center_y, width, height, rotation]
        """
        try:
            center_x, center_y = detection.center
            width = detection.width if detection.width is not None else 50.0
            height = detection.height if detection.height is not None else 50.0
            rotation = detection.angle if detection.angle is not None else 0.0

            # Normalize rotation to [0, 2π) for OpenLabel output
            rotation_output = normalize_angle_to_2pi_range(rotation)

            # Verbose logging for angle analysis (first 1000 detections)
            if self.verbose and self._debug_frame_count < 1000:
                logger.info(f"Frame {frame_idx}: obj={detection.object_id}, "
                           f"internal_angle={rotation:.4f}rad ({np.degrees(rotation):.1f}°) → "
                           f"output={rotation_output:.4f}rad ({np.degrees(rotation_output):.1f}°), "
                           f"w={width:.1f}px, h={height:.1f}px, source={detection.source_engine}")
                self._debug_frame_count += 1

            # Format for OpenLabel: [center_x, center_y, width, height, rotation]
            xywhr_formatted = [
                int(center_x),
                int(center_y),
                int(width),
                int(height),
                round(rotation_output, 4)
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
