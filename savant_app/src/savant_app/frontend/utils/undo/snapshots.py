"""Snapshot helpers used by undo/redo commands."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from savant_app.models.OpenLabel import FrameLevelObject, ObjectMetadata, RotatedBBox


@dataclass(frozen=True)
class BBoxGeometrySnapshot:
    """Geometry for a bounding box that can be re-applied later."""

    center_x: float
    center_y: float
    width: float
    height: float
    rotation: float

    @classmethod
    def from_rotated_bbox(cls, bbox: RotatedBBox) -> "BBoxGeometrySnapshot":
        return cls(
            center_x=float(bbox.x_center),
            center_y=float(bbox.y_center),
            width=float(bbox.width),
            height=float(bbox.height),
            rotation=float(bbox.rotation),
        )


@dataclass(frozen=True)
class FrameObjectSnapshot:
    """Snapshot of a frame-level object (bbox entry)."""

    frame_number: int
    object_id: str
    frame_object: FrameLevelObject

    def clone(self) -> "FrameObjectSnapshot":
        return FrameObjectSnapshot(
            frame_number=self.frame_number,
            object_id=self.object_id,
            frame_object=deepcopy(self.frame_object),
        )


@dataclass(frozen=True)
class ObjectMetadataSnapshot:
    """Snapshot of an object's metadata."""

    object_id: str
    metadata: ObjectMetadata

    def clone(self) -> "ObjectMetadataSnapshot":
        return ObjectMetadataSnapshot(
            object_id=self.object_id,
            metadata=deepcopy(self.metadata),
        )


@dataclass(frozen=True)
class CreatedObjectSnapshot:
    """Combined snapshot produced when a brand new object is created."""

    frame_snapshot: FrameObjectSnapshot
    metadata_snapshot: ObjectMetadataSnapshot

    @property
    def object_id(self) -> str:
        return self.frame_snapshot.object_id


@dataclass(frozen=True)
class CreatedRelationshipSnapshot:
    """
    Snapshot representing a created object relationship.

    NOTE: frame intervals are calculated in the backend. This
    snapshot provides the necessary info for this!

    """

    relationship_id: str
    relationship_type: str
    ontology_uid: str
    subject_object_id: str
    object_object_id: str


@dataclass(frozen=True)
class FrameTagSnapshot:
    """Snapshot representing a frame-tag interval."""

    tag_name: str
    start_frame: int
    end_frame: int
