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

logger = logging.getLogger(__name__)

# Get version from config
__version__ = '2.0.0'


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
