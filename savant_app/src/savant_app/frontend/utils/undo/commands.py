"""Concrete undoable commands."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Protocol, runtime_checkable
from .snapshots import (
    BBoxGeometrySnapshot,
    CreatedObjectSnapshot,
    FrameObjectSnapshot,
    ObjectMetadataSnapshot,
    FrameTagSnapshot,
    CreatedRelationshipSnapshot,
)
from .gateways import GatewayHolder, UndoGatewayError


@runtime_checkable
class UndoableCommand(Protocol):
    """Protocol that every undoable command must follow."""

    description: str

    def do(self, context: GatewayHolder) -> None: ...

    def undo(self, context: GatewayHolder) -> None: ...


@dataclass
class CreateNewObjectBBoxCommand:
    frame_number: int
    bbox_info: dict
    annotator: str
    description: str = "Create new object bounding box"
    _snapshot: Optional[CreatedObjectSnapshot] = field(
        default=None, init=False, repr=False
    )

    def do(self, context: GatewayHolder) -> None:
        gateway = context.annotation_gateway
        # On first execution, create the object and capture snapshots for undo.
        if self._snapshot is None:
            self._snapshot = gateway.create_new_object_bbox(
                frame_number=self.frame_number,
                bbox_info=self.bbox_info,
                annotator=self.annotator,
            )
            return

        # On subsequent executions (redo), restore previously captured state.
        gateway.ensure_object_metadata(self._snapshot.metadata_snapshot)
        gateway.restore_bbox(self._snapshot.frame_snapshot)

    def undo(self, context: GatewayHolder) -> None:
        gateway = context.annotation_gateway
        if self._snapshot is None:
            return
        gateway.delete_bbox(
            frame_number=self._snapshot.frame_snapshot.frame_number,
            object_id=self._snapshot.frame_snapshot.object_id,
        )
        gateway.remove_object_metadata_if_unused(self._snapshot.object_id)


@dataclass
class CreateExistingObjectBBoxCommand:
    frame_number: int
    bbox_info: dict
    annotator: str
    description: str = "Create bounding box for existing object"
    _snapshot: Optional[FrameObjectSnapshot] = field(
        default=None, init=False, repr=False
    )

    def do(self, context: GatewayHolder) -> None:
        gateway = context.annotation_gateway
        if self._snapshot is None:
            self._snapshot = gateway.add_bbox_to_existing_object(
                frame_number=self.frame_number,
                bbox_info=self.bbox_info,
                annotator=self.annotator,
            )
            return
        gateway.restore_bbox(self._snapshot)

    def undo(self, context: GatewayHolder) -> None:
        gateway = context.annotation_gateway
        if self._snapshot is None:
            return
        gateway.delete_bbox(
            frame_number=self._snapshot.frame_number,
            object_id=self._snapshot.object_id,
        )


@dataclass
class DeleteBBoxCommand:
    frame_number: int
    object_id: str
    description: str = "Delete bounding box"
    _snapshot: Optional[FrameObjectSnapshot] = field(
        default=None, init=False, repr=False
    )
    _exists: bool = field(default=False, init=False, repr=False)

    def do(self, context: GatewayHolder) -> None:
        gateway = context.annotation_gateway
        snapshot = gateway.delete_bbox(
            frame_number=self.frame_number,
            object_id=self.object_id,
        )
        self._snapshot = snapshot
        self._exists = snapshot is not None

    def undo(self, context: GatewayHolder) -> None:
        gateway = context.annotation_gateway
        if not self._exists or self._snapshot is None:
            return
        gateway.restore_bbox(self._snapshot)


@dataclass
class UpdateBBoxGeometryCommand:
    frame_number: int
    object_id: str
    before: BBoxGeometrySnapshot
    after: BBoxGeometrySnapshot
    annotator: str
    description: str = "Update bounding box geometry"

    def do(self, context: GatewayHolder) -> None:
        context.annotation_gateway.apply_geometry(
            frame_number=self.frame_number,
            object_id=self.object_id,
            geometry=self.after,
            annotator=self.annotator,
        )

    def undo(self, context: GatewayHolder) -> None:
        context.annotation_gateway.apply_geometry(
            frame_number=self.frame_number,
            object_id=self.object_id,
            geometry=self.before,
            annotator=self.annotator,
        )


@dataclass
class CompositeCommand:
    description: str
    commands: Sequence[UndoableCommand]

    def do(self, context: GatewayHolder) -> None:
        for command in self.commands:
            command.do(context)

    def undo(self, context: GatewayHolder) -> None:
        for command in reversed(self.commands):
            command.undo(context)


@dataclass
class AddFrameTagCommand:
    snapshot: FrameTagSnapshot
    description: str = "Add frame tag"

    def do(self, context: GatewayHolder) -> None:
        gateway = context.frame_tag_gateway
        if gateway is None:
            raise RuntimeError("No frame tag gateway configured.")
        gateway.add_frame_tag(self.snapshot)

    def undo(self, context: GatewayHolder) -> None:
        gateway = context.frame_tag_gateway
        if gateway is None:
            raise RuntimeError("No frame tag gateway configured.")
        gateway.remove_frame_tag(self.snapshot)


@dataclass
class CascadeBBoxCommand:
    object_id: str
    frame_start: int
    frame_end: Optional[int]
    width: Optional[float]
    height: Optional[float]
    rotation: Optional[float]
    annotator: str
    description: str = "Cascade bounding box update"
    _before: Dict[int, BBoxGeometrySnapshot] = field(
        default_factory=dict, init=False, repr=False
    )
    _after: Dict[int, BBoxGeometrySnapshot] = field(
        default_factory=dict, init=False, repr=False
    )
    _modified_frames: List[int] = field(default_factory=list, init=False, repr=False)

    def do(self, context: GatewayHolder) -> None:
        gateway = context.annotation_gateway
        if not self._before:
            frames = gateway.frames_for_object(self.object_id)
            frames_to_update = [
                frame
                for frame in frames
                if frame >= self.frame_start
                and (self.frame_end is None or frame <= self.frame_end)
            ]
            self._before = {
                frame: gateway.capture_geometry(frame, self.object_id)
                for frame in frames_to_update
            }
            modified_frames = gateway.cascade_bbox_edit(
                frame_start=self.frame_start,
                frame_end=self.frame_end,
                object_id=self.object_id,
                width=self.width,
                height=self.height,
                rotation=self.rotation,
                annotator=self.annotator,
            )
            self._modified_frames = list(modified_frames)
            self._after = {
                frame: gateway.capture_geometry(frame, self.object_id)
                for frame in modified_frames
            }
            return

        for frame, geometry in self._after.items():
            gateway.apply_geometry(
                frame_number=frame,
                object_id=self.object_id,
                geometry=geometry,
                annotator=self.annotator,
            )

    def undo(self, context: GatewayHolder) -> None:
        if not self._before:
            return
        gateway = context.annotation_gateway
        for frame, geometry in self._before.items():
            gateway.apply_geometry(
                frame_number=frame,
                object_id=self.object_id,
                geometry=geometry,
                annotator=self.annotator,
            )

    @property
    def modified_frames(self) -> Sequence[int]:
        return tuple(self._modified_frames)


@dataclass
class LinkObjectIdsCommand:
    primary_object_id: str
    secondary_object_id: str
    description: str = "Link object IDs"
    _frame_snapshots: List[FrameObjectSnapshot] = field(
        default_factory=list, init=False, repr=False
    )
    _metadata_snapshot: Optional[ObjectMetadataSnapshot] = field(
        default=None, init=False, repr=False
    )
    _affected_frames: List[int] = field(default_factory=list, init=False, repr=False)

    def do(self, context: GatewayHolder) -> None:
        gateway = context.annotation_gateway
        if not self._frame_snapshots:
            frames = gateway.frames_for_object(self.secondary_object_id)
            for frame in frames:
                snapshot = gateway.capture_frame_object(frame, self.secondary_object_id)
                if snapshot is not None:
                    self._frame_snapshots.append(snapshot.clone())
            self._metadata_snapshot = gateway.capture_object_metadata(
                self.secondary_object_id
            )
        affected = gateway.link_object_ids(
            self.primary_object_id,
            self.secondary_object_id,
        )
        self._affected_frames = list(affected or [])

    def undo(self, context: GatewayHolder) -> None:
        gateway = context.annotation_gateway
        if not self._frame_snapshots and self._metadata_snapshot is None:
            return
        for frame in self._affected_frames:
            gateway.delete_bbox(frame, self.primary_object_id)
        for snapshot in self._frame_snapshots:
            gateway.restore_bbox(snapshot.clone())
        if self._metadata_snapshot is not None:
            gateway.ensure_object_metadata(self._metadata_snapshot)
        self._affected_frames = sorted(
            {snapshot.frame_number for snapshot in self._frame_snapshots}
        )

    @property
    def affected_frames(self) -> Sequence[int]:
        return tuple(sorted(set(self._affected_frames)))


@dataclass
class InterpolateAnnotationsCommand:
    object_id: str
    start_frame: int
    end_frame: int
    annotator: str
    description: str = "Interpolate bounding boxes"
    _before_snapshots: Dict[int, Optional[FrameObjectSnapshot]] = field(
        default_factory=dict, init=False, repr=False
    )
    _before_flags: Dict[int, bool] = field(default_factory=dict, init=False, repr=False)
    _after_snapshots: Dict[int, FrameObjectSnapshot] = field(
        default_factory=dict, init=False, repr=False
    )
    _after_flags: Dict[int, bool] = field(default_factory=dict, init=False, repr=False)
    _affected_frames: List[int] = field(default_factory=list, init=False, repr=False)

    def _intermediate_frames(self) -> List[int]:
        return [frame for frame in range(self.start_frame + 1, self.end_frame)]

    def do(self, context: GatewayHolder) -> None:
        frames = self._intermediate_frames()
        if not frames:
            return
        gateway = context.annotation_gateway
        if not self._after_snapshots:
            for frame in frames:
                snapshot = gateway.capture_frame_object(frame, self.object_id)
                self._before_snapshots[frame] = snapshot.clone() if snapshot else None
                self._before_flags[frame] = gateway.is_interpolated(
                    frame, self.object_id
                )
        gateway.interpolate_annotations(
            self.object_id,
            self.start_frame,
            self.end_frame,
            self.annotator,
        )
        self._after_snapshots = {}
        self._after_flags = {}
        for frame in frames:
            snapshot = gateway.capture_frame_object(frame, self.object_id)
            if snapshot is None:
                raise UndoGatewayError(
                    f"Interpolated bbox for frame {frame} and object {self.object_id} missing."
                )
            self._after_snapshots[frame] = snapshot.clone()
            self._after_flags[frame] = gateway.is_interpolated(frame, self.object_id)
            gateway.set_interpolated_flag(
                frame, self.object_id, self._after_flags[frame]
            )
        self._affected_frames = frames

    def undo(self, context: GatewayHolder) -> None:
        frames = self._intermediate_frames()
        if not frames:
            return
        gateway = context.annotation_gateway
        for frame in frames:
            gateway.delete_bbox(frame, self.object_id)
        for frame in frames:
            snapshot = self._before_snapshots.get(frame)
            if snapshot is not None:
                gateway.restore_bbox(snapshot.clone())
        for frame in frames:
            flag = self._before_flags.get(frame, False)
            gateway.set_interpolated_flag(frame, self.object_id, flag)

    @property
    def affected_frames(self) -> Sequence[int]:
        return tuple(self._affected_frames)


@dataclass
class RemoveFrameTagCommand:
    snapshot: FrameTagSnapshot
    description: str = "Remove frame tag"

    def do(self, context: GatewayHolder) -> None:
        gateway = context.frame_tag_gateway
        if gateway is None:
            raise RuntimeError("No frame tag gateway configured.")
        gateway.remove_frame_tag(self.snapshot)

    def undo(self, context: GatewayHolder) -> None:
        gateway = context.frame_tag_gateway
        if gateway is None:
            raise RuntimeError("No frame tag gateway configured.")
        gateway.add_frame_tag(self.snapshot)


@dataclass
class ResolveConfidenceCommand:
    frame_number: int
    object_id: str
    annotator: str
    description: str = "Resolve confidence issue"
    _before_snapshot: Optional[FrameObjectSnapshot] = field(
        default=None, init=False, repr=False
    )
    _after_snapshot: Optional[FrameObjectSnapshot] = field(
        default=None, init=False, repr=False
    )

    def do(self, context: GatewayHolder) -> None:
        gateway = context.annotation_gateway
        if self._after_snapshot is None:
            self._before_snapshot = gateway.capture_frame_object(
                self.frame_number, self.object_id
            )
            gateway.mark_confidence_resolved(
                self.frame_number, self.object_id, self.annotator
            )
            snapshot = gateway.capture_frame_object(self.frame_number, self.object_id)
            if snapshot is None:
                raise UndoGatewayError(
                    "Failed to capture snapshot after resolving confidence."
                )
            self._after_snapshot = snapshot
        else:
            gateway.restore_bbox(self._after_snapshot.clone())

    def undo(self, context: GatewayHolder) -> None:
        if self._before_snapshot is None:
            return
        gateway = context.annotation_gateway
        gateway.restore_bbox(self._before_snapshot.clone())


@dataclass
class CreateObjectRelationshipCommand:
    relationship_type: str
    ontology_uid: str
    subject_object_id: str
    object_object_id: str
    description: str = "Create object relationship"
    _snapshot: Optional[CreatedRelationshipSnapshot] = field(
        default=None, init=False, repr=False
    )

    def do(self, context: GatewayHolder) -> None:
        gateway = context.annotation_gateway

        # Create a new object relationship.
        if self._snapshot is None:
            relationship_id = gateway.create_object_relationship(
                self.relationship_type,
                self.ontology_uid,
                self.subject_object_id,
                self.object_object_id,
            )
            # Create snapshot with the relationship details
            self._snapshot = CreatedRelationshipSnapshot(
                relationship_id=relationship_id,
                relationship_type=self.relationship_type,
                ontology_uid=self.ontology_uid,
                subject_object_id=self.subject_object_id,
                object_object_id=self.object_object_id,
            )
            return

        # On redo, restore the relationship
        gateway.restore_object_relationship(
            self._snapshot.relationship_id,
            self._snapshot.relationship_type,
            self._snapshot.ontology_uid,
            self._snapshot.subject_object_id,
            self._snapshot.object_object_id,
        )

    def undo(self, context: GatewayHolder) -> None:
        gateway = context.annotation_gateway
        if self._snapshot is not None:
            gateway.delete_object_relationship(self._snapshot.relationship_id)
