"""
openlabel - OpenLabel JSON format handling for SAVANT

This module provides reading and writing capabilities for OpenLabel format files.
It is designed to be a shared library used across SAVANT tools (markit, savant_app, trainit).

Reading:
    - Pydantic models for parsing and validating OpenLabel JSON files
    - load_openlabel() function for loading files
    - get_boxes_with_ids_for_frame() for extracting frame-level annotations

Writing:
    - OpenLabelWriter class for creating OpenLabel JSON files
    - DetectionData dataclass for passing detection information
    - NumpyEncoder for JSON serialization of NumPy types
"""

import json
import logging
import math
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, confloat, conint, model_serializer, model_validator

logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================

def normalize_angle_to_2pi_range(angle: float) -> float:
    """Normalize angle to [0, 2π) range for OpenLabel output.

    Args:
        angle: Angle in radians (can be any value)

    Returns:
        Normalized angle in [0, 2π) range
    """
    result = angle % (2 * math.pi)
    if result < 0:
        result += 2 * math.pi
    return float(result)


# =============================================================================
# Reading Classes (Pydantic Models)
# =============================================================================

class RotatedBBox(BaseModel):
    """Rotated bounding box coordinates (x_center, y_center, width, height, rotation).

    Coordinates are in pixel values. Rotation is in radians.
    """

    x_center: confloat(ge=0)
    y_center: confloat(ge=0)
    width: confloat(gt=0)
    height: confloat(gt=0)
    rotation: float  # radians

    @model_serializer
    def serialize(self) -> list:
        return [self.x_center, self.y_center, self.width, self.height, self.rotation]

    @model_validator(mode="before")
    @classmethod
    def validate_deserialize(cls, data):
        if isinstance(data, list):
            if len(data) != 5:
                raise ValueError(
                    "RotatedBBox requires exactly 5 elements for deserialization"
                )
            return {
                "x_center": data[0],
                "y_center": data[1],
                "width": data[2],
                "height": data[3],
                "rotation": data[4],
            }
        return data


class GeometryData(BaseModel):
    """Represents geometric data for rotated bounding boxes."""

    name: Literal["shape"] = "shape"
    val: RotatedBBox


class ConfidenceData(BaseModel):
    """Contains confidence score for detection."""

    name: Literal["confidence"] = "confidence"
    val: deque[confloat(ge=0, le=1)]  # List of confidence scores


class AnnotatorData(BaseModel):
    """Contains annotator information."""

    name: Literal["annotator"] = "annotator"
    val: deque[str]


class ObjectData(BaseModel):
    """Container for object's geometric and confidence data."""

    rbbox: List[GeometryData]
    vec: List[Union[ConfidenceData, AnnotatorData]]


class FrameLevelObject(BaseModel):
    """Represents an object's data within a specific frame."""

    object_data: ObjectData


class FrameObjects(BaseModel):
    """Represents all objects within a specific frame."""

    objects: Dict[str, FrameLevelObject]


class FrameInterval(BaseModel):
    """Represents the frames in which the object exists."""

    frame_start: conint(ge=0)
    frame_end: conint(ge=0)


class ObjectMetadata(BaseModel):
    """Metadata for tracked objects across all frames."""

    name: str
    type: str
    ontology_uid: Optional[str] = None
    frame_intervals: Optional[List[FrameInterval]] = None


class OpenLabelMetadata(BaseModel):
    """Top-level metadata for the OpenLabel annotation file."""

    schema_version: str
    tagged_file: Optional[str] = None
    annotator: Optional[str] = None
    name: Optional[str] = None
    comment: Optional[str] = None
    tags: Optional[List[str]] = None


class ActionMetadata(BaseModel):
    """Action metadata."""

    name: str
    type: str
    ontology_uid: Optional[str] = None
    frame_intervals: Optional[List[FrameInterval]] = None


class OntologyDetails(BaseModel):
    """Ontology details for specifying namespace and filtering.

    Refer to: https://github.com/RI-SE/SAVANT/tree/main/ontology
    """

    uri: str
    boundary_list: Optional[List[str]] = None
    boundary_mode: Optional[Literal["include", "exclude"]] = None


class OpenLabel(BaseModel):
    """Main model representing the complete OpenLabel structure."""

    metadata: OpenLabelMetadata
    ontologies: Dict[str, Union[str, OntologyDetails]]
    objects: Dict[str, ObjectMetadata]
    actions: Optional[Dict[str, ActionMetadata]] = None
    frames: Dict[str, FrameObjects]

    def model_dump(self, *args, **kwargs) -> dict:
        """Override Pydantic's default model_dump to exclude None values."""
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(*args, **kwargs)

    def get_boxes_with_ids_for_frame(self, frame_idx: int) -> List[Tuple]:
        """Return bounding boxes for a specific frame.

        Args:
            frame_idx: The frame index to get boxes for

        Returns:
            List of tuples containing:
            (object_id: str, object_type: str,
             x_center: float, y_center: float,
             width: float, height: float, rotation: float)
        """
        results = []
        frame_key = str(frame_idx)

        # Check if frame exists
        if frame_key not in self.frames:
            return results

        frame = self.frames[frame_key]

        for object_id, frame_obj in frame.objects.items():
            # Get object metadata
            metadata = self.objects.get(object_id)
            object_type = metadata.type if metadata else "unknown"

            for geometry_data in frame_obj.object_data.rbbox:
                if geometry_data.name != "shape":
                    continue

                # Extract RotatedBBox values
                rbbox = geometry_data.val
                bbox_info = (
                    object_id,
                    object_type,
                    rbbox.x_center,
                    rbbox.y_center,
                    rbbox.width,
                    rbbox.height,
                    rbbox.rotation,
                )
                results.append(bbox_info)

        return results

    def get_frame_indices(self) -> List[int]:
        """Return sorted list of all frame indices in the annotation."""
        return sorted(int(k) for k in self.frames.keys())


def load_openlabel(json_path: Union[str, Path]) -> OpenLabel:
    """Load and validate an OpenLabel JSON file.

    Args:
        json_path: Path to the OpenLabel JSON file

    Returns:
        Validated OpenLabel model

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is not valid OpenLabel format
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"OpenLabel file not found: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    # OpenLabel files have an "openlabel" root key
    if "openlabel" not in data:
        raise ValueError(f"Not a valid OpenLabel file (missing 'openlabel' key): {json_path}")

    try:
        return OpenLabel.model_validate(data["openlabel"])
    except Exception as e:
        raise ValueError(f"Failed to parse OpenLabel file: {json_path}: {e}")


# =============================================================================
# Writing Classes
# =============================================================================

@dataclass
class DetectionData:
    """Generic detection data for OpenLabel writing.

    This is a framework-agnostic representation of a detection that can be
    used to create OpenLabel annotations.
    """
    object_id: int
    class_id: int
    center: Tuple[float, float]  # (x, y) in pixels
    width: float  # in pixels
    height: float  # in pixels
    angle: float  # in radians
    confidence: float
    source_engine: str = "unknown"


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


class OpenLabelWriter:
    """Handles OpenLabel JSON structure creation and management.

    This class provides methods for building OpenLabel files programmatically.
    It is compatible with the interface used in markit for easy migration.
    """

    def __init__(self, schema_path: str, verbose: bool = False):
        """Initialize OpenLabel writer.

        Args:
            schema_path: Path to OpenLabel JSON schema file
            verbose: Enable verbose logging for angle/detection analysis
        """
        self.schema_path = schema_path
        self.verbose = verbose
        self.openlabel_data: Dict[str, Any] = {}
        self._debug_frame_count = 0
        self._class_translation_count = 0
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

    def add_metadata(
        self,
        video_path: str,
        annotator: str,
        schema_version: str = "1.0",
        name: Optional[str] = None,
        comment: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """Add metadata to OpenLabel structure.

        Args:
            video_path: Path to source video file
            annotator: Name of the annotator/tool
            schema_version: OpenLabel schema version
            name: Optional name for the annotation
            comment: Optional comment
            tags: Optional list of tags
        """
        video_basename = os.path.basename(video_path)

        metadata = {
            "schema_version": schema_version,
            "tagged_file": video_basename,
            "annotator": annotator,
        }

        if name:
            metadata["name"] = name
        if comment:
            metadata["comment"] = comment
        if tags:
            metadata["tags"] = tags

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

    def add_frame_objects(
        self,
        frame_idx: int,
        detections: List[DetectionData],
        class_map: Dict[int, str]
    ) -> None:
        """Add detected objects for a frame to OpenLabel structure.

        Args:
            frame_idx: Frame index
            detections: List of DetectionData objects
            class_map: Mapping of class IDs to names
        """
        if self.openlabel_data is None:
            raise RuntimeError("OpenLabel structure not initialized")

        frame_objects = {}
        seen_object = False

        for detection in detections:
            if detection.object_id is not None:
                obj_id_str = str(detection.object_id)
                seen_object = True

                # Convert detection data to OpenLabel-style xywhr format
                xywhr_formatted = self._detection_to_xywhr(detection, frame_idx)

                # Add new object to global objects list if not seen before
                if obj_id_str not in self.openlabel_data["openlabel"]["objects"]:
                    class_label = class_map.get(detection.class_id, str(detection.class_id))

                    if self.verbose and self._class_translation_count < 25:
                        logger.info(
                            f"Class translation: class_id={detection.class_id} → "
                            f"label='{class_label}' (source={detection.source_engine})"
                        )
                        self._class_translation_count += 1

                    self.openlabel_data["openlabel"]["objects"][obj_id_str] = {
                        "name": f"Object-{obj_id_str}",
                        "type": class_label,
                        "ontology_uid": "0"
                    }

                # Build vec data
                vec_data = [
                    {
                        "name": "annotator",
                        "val": [detection.source_engine]
                    },
                    {
                        "name": "confidence",
                        "val": [float(detection.confidence)]
                    }
                ]

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

        # Only add frame if there are objects
        if seen_object:
            self.openlabel_data["openlabel"]["frames"][str(frame_idx)] = {
                "objects": frame_objects
            }

    def _detection_to_xywhr(self, detection: DetectionData, frame_idx: int) -> List:
        """Convert DetectionData to OpenLabel xywhr format.

        Args:
            detection: Detection data
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

            if self.verbose and self._debug_frame_count < 1000:
                logger.info(
                    f"Frame {frame_idx}: obj={detection.object_id}, "
                    f"angle={rotation:.4f}rad → output={rotation_output:.4f}rad, "
                    f"w={width:.1f}px, h={height:.1f}px"
                )
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
            return [int(detection.center[0]), int(detection.center[1]), 50, 50, 0.0]

    def sort_objects(self) -> None:
        """Sort objects dictionary numerically by key."""
        if self.openlabel_data is None:
            return

        try:
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
