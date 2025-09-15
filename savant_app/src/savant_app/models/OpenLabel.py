from pydantic import (
    BaseModel,
    conint,
    confloat,
    model_serializer,
    model_validator,
)
from typing import Dict, List, Literal, Union, Optional
from math import isfinite


class RotatedBBox(BaseModel):
    """Rotated bounding box coordinates (x_center, y_center, width, height, rotation)"""

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
    """Represents geometric data for rotated bounding boxes"""

    name: Literal["shape"] = "shape"
    val: RotatedBBox


class ConfidenceData(BaseModel):
    """Contains confidence score for detection"""

    name: Literal["confidence"] = "confidence"
    val: List[confloat(ge=0, le=1)]  # List of confidence scores


class AnnotatorData(BaseModel):
    """Contains annotator information"""

    name: Literal["annotator"] = "annotator"
    val: List[str]


class ObjectData(BaseModel):
    """Container for object's geometric and confidence data"""

    rbbox: List[GeometryData]
    vec: List[Union[ConfidenceData, AnnotatorData]]


class FrameLevelObject(BaseModel):
    """Represents an object's data within a specific frame"""

    object_data: ObjectData


class FrameObjects(BaseModel):
    """Represents all objects within a specific frame"""

    objects: Dict[str, FrameLevelObject]


class FrameInterval(BaseModel):
    """Represents the frames in which the object exists"""

    frame_start: conint(ge=0)
    frame_end: conint(ge=0)


class ObjectMetadata(BaseModel):
    """Metadata for tracked objects across all frames"""

    name: str
    type: str
    ontology_uid: Optional[str] = None
    frame_intervals: Optional[List[FrameInterval]] = None


class OpenLabelMetadata(BaseModel):
    """Top-level metadata for the OpenLabel annotation file"""

    schema_version: str
    tagged_file: Optional[str] = None
    annotator: Optional[str] = None


class ActionMetadata(BaseModel):
    """Action metadata"""

    name: str
    type: str
    ontology_uid: Optional[str] = None
    frame_intervals: Optional[List[FrameInterval]] = None


class OntologyDetails(BaseModel):
    """
    Ontology details which are not yet in use, but are ready for when needed.
    Refer to the following section in the readme:
    https://github.com/fwrise/SAVANT/tree/main/Specification#savant-ontology
    """

    uri: str
    boundary_list: Optional[List[str]] = None
    boundary_mode: Optional[Literal["include", "exclude"]] = None


class OpenLabel(BaseModel):
    """Main model representing the complete OpenLabel structure"""

    metadata: OpenLabelMetadata
    ontologies: Dict[str, Union[str, OntologyDetails]]
    objects: Dict[str, ObjectMetadata]
    actions: Optional[Dict[str, ActionMetadata]] = (
        None  # Made optional as they are not being used yet according to the spec.
    )
    frames: Dict[str, FrameObjects]

    def model_dump(self, *args, **kwargs) -> dict:
        """ "
        This overrides Pydantic's default model_dump, such that
        we exclude fields with a None value.
        """
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(*args, **kwargs)

    def add_new_object(self, obj_type: str, obj_id: str):

        new_obj_metadata = ObjectMetadata(
            name=f"Object-{obj_id}",
            type=obj_type,
        )

        self.objects[obj_id] = new_obj_metadata

    def append_object_bbox(
        self,
        frame_id: int,
        bbox_coordinates: dict,
        confidence_data: dict,
        annotater_data: dict,
        obj_id: str,
    ):
        """
        Adds a new bounding box for an object with no existing
        annotations.
        """
        # TODO: Add rotation. Hard coded to 0 for now.
        rbbox = RotatedBBox(
            x_center=bbox_coordinates[0],
            y_center=bbox_coordinates[1],
            width=bbox_coordinates[2],
            height=bbox_coordinates[3],
            rotation=0,
        )

        bbox_geometry_data = GeometryData(val=rbbox)
        new_obj_data = ObjectData(
            rbbox=[bbox_geometry_data], vec=[annotater_data, confidence_data]
        )
        new_frame_obj = FrameLevelObject(object_data=new_obj_data)

        # Adds a NEW bounding box.
        # Will overwrite if a bounding box ID already exists.
        self.frames[str(frame_id)].objects[new_bbox_key] = new_frame_obj

    def _get_frame_object(
        self,
        frame_key: Union[int, str],
        object_key: Union[int, str],
    ) -> FrameLevelObject:
        """
        Internal: fetch a FrameLevelObject by frame/object keys or raise a clear error.
        """
        frame_id_str = str(frame_key)
        object_id_str = str(object_key)

        if frame_id_str not in self.frames:
            raise KeyError(f"Frame '{frame_id_str}' not found")

        frame_objects = self.frames[frame_id_str].objects
        if object_id_str not in frame_objects:
            raise KeyError(f"Object '{object_id_str}' not found in frame '{frame_id_str}'")

        return frame_objects[object_id_str]

    def get_bbox(
        self,
        frame_key: Union[int, str],
        object_key: Union[int, str],
        bbox_index: int = 0,
    ) -> RotatedBBox:
        """
        Return the RotatedBBox for (frame, object, bbox_index).

        Args:
            frame_key: Frame identifier (int or str).
            object_key: Object identifier (int or str).
            bbox_index: Index within the object's rbbox list (default 0).
        """
        frame_object = self._get_frame_object(frame_key, object_key)
        geometry_items = frame_object.object_data.rbbox

        if not geometry_items:
            raise IndexError(
                f"Object '{object_key}' in frame '{frame_key}' has no rbbox entries"
            )

        try:
            geometry_entry = geometry_items[bbox_index]
        except IndexError as exc:
            raise IndexError(
                f"rbbox index {bbox_index} out of range for object "
                f"'{object_key}' in frame '{frame_key}'"
            ) from exc

        return geometry_entry.val

    def update_bbox(
        self,
        frame_key: Union[int, str],
        object_key: Union[int, str],
        *,
        bbox_index: int = 0,
        # absolute values (optional)
        x_center: Optional[float] = None,
        y_center: Optional[float] = None,
        width:    Optional[float] = None,
        height:   Optional[float] = None,
        rotation: Optional[float] = None,
        # deltas values (optional)
        delta_x: float = 0.0,
        delta_y: float = 0.0,
        delta_w: float = 0.0,
        delta_h: float = 0.0,
        delta_theta: float = 0.0,
        # clamps
        min_width: float = 1e-6,
        min_height: float = 1e-6,
    ) -> RotatedBBox:
        """
        Update a rotated bbox in-place (pure domain logic) and return the updated box.
        Uses absolute values if provided; then applies deltas; clamps width/height.
        """
        frame_object = self._get_frame_object(frame_key, object_key)
        geometry_items = frame_object.object_data.rbbox

        if not geometry_items:
            raise IndexError(
                f"Object '{object_key}' in frame '{frame_key}' has no rbbox entries"
            )

        try:
            geometry_entry = geometry_items[bbox_index]
        except IndexError as exc:
            raise IndexError(
                f"rbbox index {bbox_index} out of range for object "
                f"'{object_key}' in frame '{frame_key}'"
            ) from exc

        current_bbox: RotatedBBox = geometry_entry.val

        new_x_center = x_center if (x_center is not None and isfinite(
            x_center)) else current_bbox.x_center
        new_y_center = y_center if (y_center is not None and isfinite(
            y_center)) else current_bbox.y_center
        new_width = width if (width is not None and isfinite(width)) else current_bbox.width
        new_height = height if (height is not None and isfinite(height)) else current_bbox.height
        new_rotation = rotation if (rotation is not None and isfinite(
            rotation)) else current_bbox.rotation

        new_x_center += delta_x
        new_y_center += delta_y
        new_width += delta_w
        new_height += delta_h
        new_rotation += delta_theta

        new_width = max(min_width, new_width)
        new_height = max(min_height, new_height)

        updated_bbox = RotatedBBox.model_validate(
            [new_x_center, new_y_center, new_width, new_height, new_rotation]
        )
        geometry_entry.val = updated_bbox
        return updated_bbox
