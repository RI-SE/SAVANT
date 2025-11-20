from collections import deque
from math import isfinite
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, confloat, conint, model_serializer, model_validator


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
    val: deque[confloat(ge=0, le=1)]  # List of confidence scores


class AnnotatorData(BaseModel):
    """Contains annotator information"""

    name: Literal["annotator"] = "annotator"
    val: deque[str]


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


class RDFItem(BaseModel):
    """Represents an item in rdf_subjects or rdf_objects"""

    type: Literal["object", "action", "event", "context"]
    uid: str


class RelationMetadata(BaseModel):
    """Relation metadata"""

    name: str
    type: str
    ontology_uid: str
    frame_intervals: Optional[List[FrameInterval]] = None
    rdf_subjects: List[RDFItem]
    rdf_objects: List[RDFItem]


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
    relations: Optional[Dict[str, RelationMetadata]] = None

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
        bbox_coordinates: List[float],
        obj_id: str,
        annotator: str,
    ):
        """
        Adds a new bounding box for an object with no existing
        annotations.
        """
        # Handle bbox_coordinates: support both list and dict
        if isinstance(bbox_coordinates, dict):
            # Extract values from dictionary
            x_center_val = bbox_coordinates.get("x_center", 0.0)
            y_center_val = bbox_coordinates.get("y_center", 0.0)
            width_val = bbox_coordinates.get("width", 0.0)
            height_val = bbox_coordinates.get("height", 0.0)
            rotation_val = bbox_coordinates.get("rotation", 0.0)
        else:
            try:
                # Try to treat as list
                x_center_val = (
                    bbox_coordinates[0]
                    if bbox_coordinates and len(bbox_coordinates) > 0
                    else 0.0
                )
                y_center_val = (
                    bbox_coordinates[1]
                    if bbox_coordinates and len(bbox_coordinates) > 1
                    else 0.0
                )
                width_val = (
                    bbox_coordinates[2]
                    if bbox_coordinates and len(bbox_coordinates) > 2
                    else 0.0
                )
                height_val = (
                    bbox_coordinates[3]
                    if bbox_coordinates and len(bbox_coordinates) > 3
                    else 0.0
                )
                rotation_val = (
                    bbox_coordinates[4]
                    if bbox_coordinates and len(bbox_coordinates) > 4
                    else 0.0
                )
            except (TypeError, IndexError):
                # If there's an exception, set to default
                x_center_val, y_center_val, width_val, height_val, rotation_val = (
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )

        rbbox = RotatedBBox(
            x_center=x_center_val,
            y_center=y_center_val,
            width=width_val,
            height=height_val,
            rotation=rotation_val,
        )

        bbox_geometry_data = GeometryData(val=rbbox)
        confidence_data = ConfidenceData(
            val=deque([1.0])
        )  # Confidence set to 1 for humans
        annotator_data = AnnotatorData(val=deque([annotator]))

        new_obj_data = ObjectData(
            rbbox=[bbox_geometry_data], vec=[annotator_data, confidence_data]
        )
        new_frame_obj = FrameLevelObject(object_data=new_obj_data)

        # Adds a NEW bounding box.
        # Will overwrite if a bounding box ID already exists.
        self.frames[str(frame_id)].objects[obj_id] = new_frame_obj

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
            raise KeyError(
                f"Object '{object_id_str}' not found in frame '{frame_id_str}'"
            )

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
        annotator: str,
        # absolute values (optional)
        x_center: Optional[float] = None,
        y_center: Optional[float] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
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

        new_x_center = (
            x_center
            if (x_center is not None and isfinite(x_center))
            else current_bbox.x_center
        )
        new_y_center = (
            y_center
            if (y_center is not None and isfinite(y_center))
            else current_bbox.y_center
        )
        new_width = (
            width if (width is not None and isfinite(width)) else current_bbox.width
        )
        new_height = (
            height if (height is not None and isfinite(height)) else current_bbox.height
        )
        new_rotation = (
            rotation
            if (rotation is not None and isfinite(rotation))
            else current_bbox.rotation
        )

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

        vec_data = frame_object.object_data.vec
        annotator_data = None
        confidence_data = None

        for item in vec_data:
            if item.name == "annotator":
                annotator_data = item
            elif item.name == "confidence":
                confidence_data = item

        if annotator_data is not None and confidence_data is not None:
            self.update_annotator(annotator, annotator_data, 1.0, confidence_data)

        return updated_bbox

    def delete_bbox(
        self, frame_key: int | str, object_key: str
    ) -> Optional["FrameLevelObject"]:
        """
        Remove a bbox (FrameLevelObject) from a specific frame by object_key.
        Returns the removed FrameLevelObject if it existed, else None.
        """
        fkey = str(frame_key)
        if fkey not in self.frames:
            return None
        frame = self.frames[fkey]
        return frame.objects.pop(object_key, None)

    def restore_bbox(
        self, frame_key: int | str, object_key: str, frame_obj: "FrameLevelObject"
    ) -> None:
        """
        Restore a previously removed bbox (FrameLevelObject) at the same frame/object key.
        Overwrites if something is already there.
        """
        fkey = str(frame_key)
        if fkey not in self.frames:
            # Create empty frame if needed
            self.frames[fkey] = FrameObjects(objects={})
        self.frames[fkey].objects[object_key] = frame_obj

    def get_boxes_with_ids_for_frame(self, frame_idx: int) -> list:
        """Return bounding boxes for a specific frame.

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

    def update_annotator(
        self,
        annotater: str,
        current_annotator_data: AnnotatorData,
        confidence: float,
        current_confidence_data: ConfidenceData,
    ):
        """Internal: Update annotator data for a bbox."""
        # Check if the annotator making the update is already
        # the latest annotator.
        if annotater not in current_annotator_data.val[0]:
            current_annotator_data.val.appendleft(annotater)
            current_confidence_data.val.appendleft(confidence)

    def add_object_relationship(
        self,
        relationship_type: str,
        ontology_uid: str,
        subject_object_id: str,
        object_object_id: str,
        frame_intervals: list[tuple],
    ) -> str:
        """
        Add a new object relationship and return its unique ID. 
        """

        def _generate_relation_id(self) -> str:
            """
            Get the latest key of the relationships, and increment it by 1
            to generate a unqiue relation ID.
            """
            # If there are no relations, the first ID should be 0.
            if not self.relations or not self.relations.keys():
                return str(0)
            # Sorted ensures that the last key is always the largest.
            # This prevents bugs if we delete keys in the middle of
            # the dict.
            return str(sorted(self.relations.keys())[-1] + 1)

        # Convert the received frame intervals into the OL spec
        # type.
        ol_frame_intervals = [
            FrameInterval(frame_start=interval[0], frame_end=interval[1])
            for interval in frame_intervals
        ]

        # Create a new unique relation ID.
        new_id = _generate_relation_id(self)

        # Generate name for the relationship based on the ID
        relationship_name = f"Relation-{new_id}"
        

        new_relationship = RelationMetadata(
            name=relationship_name,
            type=relationship_type,
            ontology_uid=ontology_uid,
            rdf_subjects=[RDFItem(type="object", uid=subject_object_id)],
            rdf_objects=[RDFItem(type="object", uid=object_object_id)],
            frame_intervals=ol_frame_intervals,
        )

        # If we dont have any relations, initialize an empty list.
        # This is preferred over setting a default empty dict as we want to
        # exclude the field in the output if it is None.
        if not self.relations:
            self.relations = {}

        self.relations[new_id] = new_relationship

        return str(new_id)
    
    def restore_object_relationship(
        self,
        relation_id: str,
        relationship_type: str,
        ontology_uid: str,
        subject_object_id: str,
        object_object_id: str,
        frame_intervals: list[tuple],
    ) -> None:
        """
        Restore a previously deleted object relationship with the specified ID.
        """
        # Convert the received frame intervals into the OL spec type.
        ol_frame_intervals = [
            FrameInterval(frame_start=interval[0], frame_end=interval[1])
            for interval in frame_intervals
        ]

        # Generate name for the relationship based on the ID
        relationship_name = f"Relation-{relation_id}"

        new_relationship = RelationMetadata(
            name=relationship_name,
            type=relationship_type,
            ontology_uid=ontology_uid,
            rdf_subjects=[RDFItem(type="object", uid=subject_object_id)],
            rdf_objects=[RDFItem(type="object", uid=object_object_id)],
            frame_intervals=ol_frame_intervals,
        )

        # If we don't have any relations, initialize an empty dict
        if not self.relations:
            self.relations = {}

        # Restore the relationship with the specified ID
        self.relations[relation_id] = new_relationship

    def get_object_relationships(self, object_id: str) -> List[RelationMetadata]:
        """
        Get all relationships for a given object_id.
        """
        if not self.relations:
            return []

        matching_relations = []
        for relation in self.relations.values():
            for subject in relation.rdf_subjects:
                if subject.uid == object_id:
                    matching_relations.append(relation)
                    break 
            for obj in relation.rdf_objects:
                if obj.uid == object_id:
                    matching_relations.append(relation)
                    break
        return matching_relations

    def delete_relationship(self, relation_id: str) -> bool:
        """
        Delete a relationship by its ID.
        """
        if self.relations and relation_id in self.relations:
            del self.relations[relation_id]
            return True
        return False
