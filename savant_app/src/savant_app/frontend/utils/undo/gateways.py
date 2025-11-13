"""Gateway protocols and controller-backed implementations for undo/redo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, runtime_checkable
from copy import deepcopy
from .snapshots import (
    BBoxGeometrySnapshot,
    CreatedObjectSnapshot,
    FrameObjectSnapshot,
    ObjectMetadataSnapshot,
    FrameTagSnapshot,
)


@runtime_checkable
class AnnotationGateway(Protocol):
    """Minimal API the commands expect for bbox/object changes."""

    def capture_geometry(self, frame_number: int, object_id: str) -> BBoxGeometrySnapshot:
        ...

    def capture_frame_object(
        self, frame_number: int, object_id: str
    ) -> Optional[FrameObjectSnapshot]:
        ...

    def capture_object_metadata(
        self, object_id: str
    ) -> Optional[ObjectMetadataSnapshot]:
        ...

    def apply_geometry(
        self,
        frame_number: int,
        object_id: str,
        geometry: BBoxGeometrySnapshot,
        annotator: str,
    ) -> None:
        ...

    def delete_bbox(self, frame_number: int, object_id: str) -> Optional[FrameObjectSnapshot]:
        ...

    def restore_bbox(self, snapshot: FrameObjectSnapshot) -> None:
        ...

    def ensure_object_metadata(self, metadata_snapshot: ObjectMetadataSnapshot) -> None:
        ...

    def remove_object_metadata_if_unused(self, object_id: str) -> None:
        ...

    def create_new_object_bbox(
        self, frame_number: int, bbox_info: dict, annotator: str
    ) -> CreatedObjectSnapshot:
        ...

    def create_existing_object_bbox(
        self, frame_number: int, bbox_info: dict, annotator: str
    ) -> FrameObjectSnapshot:
        ...

    def frames_for_object(self, object_id: str) -> List[int]:
        ...

    def interpolate_annotations(
        self,
        object_id: str,
        start_frame: int,
        end_frame: int,
        annotator: str,
    ) -> None:
        ...

    def is_interpolated(self, frame_number: int, object_id: str) -> bool:
        ...

    def set_interpolated_flag(
        self, frame_number: int, object_id: str, interpolated: bool
    ) -> None:
        ...

    def cascade_bbox_edit(
        self,
        frame_start: int,
        frame_end: Optional[int],
        object_id: str,
        width: Optional[float],
        height: Optional[float],
        rotation: Optional[float],
        annotator: str,
    ) -> List[int]:
        ...

    def link_object_ids(
        self,
        primary_object_id: str,
        secondary_object_id: str,
    ) -> List[int]:
        ...

    def mark_confidence_resolved(
        self, frame_number: int, object_id: str, annotator: str
    ) -> None:
        ...


@runtime_checkable
class FrameTagGateway(Protocol):
    """API for adding/removing frame tags."""

    def add_frame_tag(self, snapshot: FrameTagSnapshot) -> None:
        ...

    def remove_frame_tag(self, snapshot: FrameTagSnapshot) -> None:
        ...


@dataclass
class GatewayHolder:
    """Gateway bundle exposed to undo/redo commands."""

    annotation_gateway: AnnotationGateway
    frame_tag_gateway: Optional[FrameTagGateway] = None


class UndoGatewayError(RuntimeError):
    """Raised when the undo gateway encounters an inconsistent state."""


@dataclass
class ControllerAnnotationGateway:
    """Adapter that exposes the AnnotationController/ProjectState to commands."""

    annotation_controller: object
    project_state_controller: object

    def _annotation_config(self):
        project_state = getattr(self.project_state_controller, "project_state", None)
        if project_state is None:
            raise UndoGatewayError("Project state is not initialised.")
        config = getattr(project_state, "annotation_config", None)
        if config is None:
            raise UndoGatewayError("Annotation configuration is not loaded.")
        return config

    def _interpolation_metadata(self) -> set:
        project_state = getattr(self.project_state_controller, "project_state", None)
        if project_state is None:
            raise UndoGatewayError("Project state is not initialised.")
        metadata = getattr(project_state, "interpolation_metadata", None)
        if metadata is None:
            metadata = set()
            project_state.interpolation_metadata = metadata
        return metadata

    def capture_geometry(self, frame_number: int, object_id: str) -> BBoxGeometrySnapshot:
        bbox = self.annotation_controller.get_bbox(
            frame_key=frame_number,
            object_key=object_id,
        )
        return BBoxGeometrySnapshot.from_rotated_bbox(bbox)

    def capture_frame_object(
        self, frame_number: int, object_id: str
    ) -> Optional[FrameObjectSnapshot]:
        config = self._annotation_config()
        frame_key = str(frame_number)
        frame = getattr(config, "frames", {}).get(frame_key)
        if frame is None:
            return None
        frame_objects = getattr(frame, "objects", {})
        frame_object = frame_objects.get(object_id)
        if frame_object is None:
            return None
        return FrameObjectSnapshot(
            frame_number=frame_number,
            object_id=object_id,
            frame_object=deepcopy(frame_object),
        )

    def capture_object_metadata(
        self, object_id: str
    ) -> Optional[ObjectMetadataSnapshot]:
        config = self._annotation_config()
        objects = getattr(config, "objects", {})
        metadata = objects.get(object_id)
        if metadata is None:
            return None
        return ObjectMetadataSnapshot(
            object_id=object_id,
            metadata=deepcopy(metadata),
        )

    def apply_geometry(
        self,
        frame_number: int,
        object_id: str,
        geometry: BBoxGeometrySnapshot,
        annotator: str,
    ) -> None:
        self.annotation_controller.move_resize_bbox(
            frame_key=frame_number,
            object_key=object_id,
            x_center=geometry.center_x,
            y_center=geometry.center_y,
            width=geometry.width,
            height=geometry.height,
            rotation=geometry.rotation,
            annotator=annotator,
        )

    def delete_bbox(self, frame_number: int, object_id: str) -> Optional[FrameObjectSnapshot]:
        removed = self.annotation_controller.delete_bbox(
            frame_key=frame_number, object_key=object_id)
        if removed is None:
            return None
        return FrameObjectSnapshot(
            frame_number=frame_number,
            object_id=object_id,
            frame_object=deepcopy(removed),
        )

    def restore_bbox(self, snapshot: FrameObjectSnapshot) -> None:
        self.annotation_controller.restore_bbox(
            frame_key=snapshot.frame_number,
            object_key=snapshot.object_id,
            frame_obj=deepcopy(snapshot.frame_object),
        )

    def ensure_object_metadata(self, metadata_snapshot: ObjectMetadataSnapshot) -> None:
        config = self._annotation_config()
        objects = getattr(config, "objects", {})
        objects[metadata_snapshot.object_id] = deepcopy(metadata_snapshot.metadata)

    def remove_object_metadata_if_unused(self, object_id: str) -> None:
        config = self._annotation_config()
        frames = getattr(config, "frames", {})
        for frame in frames.values():
            if object_id in getattr(frame, "objects", {}):
                return
        objects = getattr(config, "objects", {})
        objects.pop(object_id, None)

    def create_new_object_bbox(
        self, frame_number: int, bbox_info: dict, annotator: str
    ) -> CreatedObjectSnapshot:
        config = self._annotation_config()
        before_ids = set(getattr(config, "objects", {}).keys())
        self.annotation_controller.create_new_object_bbox(
            frame_number=frame_number,
            bbox_info=bbox_info,
            annotator=annotator,
        )
        after_ids = set(getattr(config, "objects", {}).keys())
        new_ids = after_ids - before_ids
        if len(new_ids) != 1:
            raise UndoGatewayError("Unable to determine newly created object ID.")
        object_id = next(iter(new_ids))
        frame_key = str(frame_number)
        frame_objects = getattr(config.frames.get(frame_key), "objects", None)
        if not frame_objects or object_id not in frame_objects:
            raise UndoGatewayError("Newly created object is missing from frame data.")
        frame_snapshot = FrameObjectSnapshot(
            frame_number=frame_number,
            object_id=object_id,
            frame_object=deepcopy(frame_objects[object_id]),
        )
        metadata = config.objects.get(object_id)
        if metadata is None:
            raise UndoGatewayError("Newly created object metadata is missing.")
        metadata_snapshot = ObjectMetadataSnapshot(
            object_id=object_id,
            metadata=deepcopy(metadata),
        )
        return CreatedObjectSnapshot(
            frame_snapshot=frame_snapshot,
            metadata_snapshot=metadata_snapshot,
        )

    def create_existing_object_bbox(
        self, frame_number: int, bbox_info: dict, annotator: str
    ) -> FrameObjectSnapshot:
        object_name = str(bbox_info.get("object_id"))
        self.annotation_controller.create_bbox_existing_object(
            frame_number=frame_number,
            bbox_info=bbox_info,
            annotator=annotator,
        )
        config = self._annotation_config()
        objects_map = getattr(config, "objects", {})
        object_id = None
        for key, meta in objects_map.items():
            if getattr(meta, "name", None) == object_name or key == object_name:
                object_id = key
                break
        if object_id is None:
            raise UndoGatewayError(
                f"Failed to resolve object ID for existing object '{object_name}'."
            )
        frame_key = str(frame_number)
        frame = config.frames.get(frame_key)
        if frame is None or object_id not in frame.objects:
            raise UndoGatewayError("Failed to locate created bbox for existing object.")
        return FrameObjectSnapshot(
            frame_number=frame_number,
            object_id=object_id,
            frame_object=deepcopy(frame.objects[object_id]),
        )

    def frames_for_object(self, object_id: str) -> List[int]:
        result = self.annotation_controller.frames_for_object(object_id)
        return list(result or [])

    def cascade_bbox_edit(
        self,
        frame_start: int,
        frame_end: Optional[int],
        object_id: str,
        width: Optional[float],
        height: Optional[float],
        rotation: Optional[float],
        annotator: str,
    ) -> List[int]:
        updated_frames = self.annotation_controller.cascade_bbox_edit(
            frame_start=frame_start,
            frame_end=frame_end,
            object_key=object_id,
            width=width,
            height=height,
            rotation=rotation,
            annotator=annotator,
        )
        return list(updated_frames or [])

    def link_object_ids(
        self,
        primary_object_id: str,
        secondary_object_id: str,
    ) -> List[int]:
        affected = self.annotation_controller.link_object_ids(
            primary_object_id,
            secondary_object_id,
        )
        return list(affected or [])

    def interpolate_annotations(
        self,
        object_id: str,
        start_frame: int,
        end_frame: int,
        annotator: str,
    ) -> None:
        self.annotation_controller.interpolate_annotations(
            object_id,
            start_frame,
            end_frame,
            annotator,
        )

    def is_interpolated(self, frame_number: int, object_id: str) -> bool:
        metadata = self._interpolation_metadata()
        return (int(frame_number), str(object_id)) in metadata

    def set_interpolated_flag(
        self, frame_number: int, object_id: str, interpolated: bool
    ) -> None:
        metadata = self._interpolation_metadata()
        key = (int(frame_number), str(object_id))
        if interpolated:
            metadata.add(key)
        else:
            metadata.discard(key)

    def mark_confidence_resolved(
        self, frame_number: int, object_id: str, annotator: str
    ) -> None:
        self.annotation_controller.mark_confidence_resolved(
            frame_number,
            object_id,
            annotator,
        )


@dataclass
class ControllerFrameTagGateway:
    annotation_controller: object

    def add_frame_tag(self, snapshot: FrameTagSnapshot) -> None:
        self.annotation_controller.add_frame_tag(
            snapshot.tag_name,
            snapshot.start_frame,
            snapshot.end_frame,
        )

    def remove_frame_tag(self, snapshot: FrameTagSnapshot) -> None:
        removed = self.annotation_controller.remove_frame_tag(
            snapshot.tag_name,
            snapshot.start_frame,
            snapshot.end_frame,
        )
        if removed is False:
            raise UndoGatewayError("Frame tag removal failed; snapshot not present.")
