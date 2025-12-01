from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from pydantic import ValidationError

from savant_app.frontend.utils.ontology_utils import (
    get_action_labels,
    get_bbox_type_labels,
)
from savant_app.frontend.utils.settings_store import get_ontology_path
from savant_app.models.OpenLabel import (
    ActionMetadata,
    AnnotatorData,
    ConfidenceData,
    FrameInterval,
    FrameLevelObject,
    OpenLabel,
    RotatedBBox,
)

from .exceptions import (
    BBoxNotFoundError,
    FrameNotFoundError,
    InvalidFrameRangeError,
    InvalidInputError,
    NoFrameLabelFoundError,
    ObjectInFrameError,
    ObjectLinkConflictError,
    ObjectNotFoundError,
    OntologyNotFound,
    UnsportedTagTypeError,
)
from .interpolation_service import InterpolationService
from .project_state import ProjectState


class AnnotationService:
    def __init__(self, project_state: ProjectState) -> None:
        # This is a temporary cache for unsaved annotations.
        # It may hold several annotations for a given user session.
        self.project_state = project_state

        # This is a temporary cache for unsaved annotations.
        # It may hold several annotations for a given user session.
        self.cached_annotations: list[dict] = []
        self._tag_cache_key: tuple[str, float] | None = None
        self._tag_cache_vals: list[str] | None = None
        self._bbox_cache_key: tuple[str, float] | None = None
        self._bbox_cache_vals: Dict[str, List[str]] | None = None

    def create_new_object_bbox(
        self, frame_number: int, obj_type: str, coordinates: tuple, annotator: str
    ) -> None:
        """Handles both, the creation of a new object and adding a bbox for it."""
        try:
            # Generate ID
            obj_id = self._generate_new_object_id()

            # Add new object and bbox
            self._add_new_object(obj_type=obj_type, obj_id=obj_id)
            self._add_object_bbox(
                frame_number=frame_number,
                bbox_coordinates=coordinates,
                obj_id=obj_id,
                annotator=annotator,
            )
        except ValidationError as e:
            errors = "; ".join(
                f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()
            )
            raise InvalidInputError(f"Invalid input data: {errors}", e)

    def add_bbox_to_existing_object(
        self, frame_number: int, coordinates: tuple, object_id: str, annotator: str
    ) -> None:
        """Handles adding a bbox for an existing object."""

        # Verify if current frame already has the object
        if self._does_object_exist_in_frame(frame_number, object_id):
            raise ObjectInFrameError(
                f"Object ID {object_id} already has a bbox in frame {frame_number}."
            )

        self._add_object_bbox(
            frame_number=frame_number,
            bbox_coordinates=coordinates,
            obj_id=object_id,
            annotator=annotator,
        )

    def get_active_objects(self, frame_number: int) -> list[dict]:
        """Get a list of active objects for the given frame number.

        Note, filtering of active objects is done at the service level,
        due to being related to the context of the application,
        whereas the model handles what the data is and manipulating itself.
        """
        # Get frame object keys
        frame = self.project_state.annotation_config.frames.get(str(frame_number))
        if frame is None:
            return []
        active_object_keys = frame.objects.keys()

        return [
            {"type": obj.type, "name": obj.name, "id": key}
            for key, obj in self.project_state.annotation_config.objects.items()
            if key in active_object_keys
        ]

    def get_frame_objects(self, frame_limit: int, current_frame: int) -> list[str]:
        """
        Get a list of all objects with bboxes in the frame range between the
        current frame and frame_limit.
        """
        frames = self.project_state.annotation_config.frames
        global_objects = self.project_state.annotation_config.objects

        if frame_limit < 0 or current_frame < 0:
            raise InvalidFrameRangeError("Frame numbers must be non-negative.")

        start_frame = max(0, current_frame - frame_limit)

        # Convert to string keys for consistency
        frame_keys = [str(k) for k in range(start_frame, current_frame + 1)]
        frame_subset = {k: frames[k] for k in frame_keys if k in frames}

        object_ids = set()
        for frame_data in frame_subset.values():  # Use new variable name
            object_ids.update(frame_data.objects.keys())  # Use update() for set

        # Correct list comprehension:
        return [obj_id for obj_id in object_ids if obj_id in global_objects]

    def get_bbox(
        self,
        frame_key: Union[int, str],
        object_key: Union[int, str],
        bbox_index: int = 0,
    ) -> Optional[RotatedBBox]:
        """
        Read a bbox from the current annotation config (model).
        """
        openlabel_model: OpenLabel = self.project_state.annotation_config
        try:
            if object_key is None:
                return None
            return openlabel_model.get_bbox(
                frame_key=frame_key,
                object_key=object_key,
                bbox_index=bbox_index,
            )
        except KeyError as e:
            raise BBoxNotFoundError(
                f"BBox not found for object {object_key} in frame {frame_key}.", e
            )

    def move_resize_bbox(
        self,
        frame_key: Union[int, str],
        object_key: Union[int, str],
        *,
        bbox_index: int = 0,
        # absolute (optional)
        x_center: Optional[float] = None,
        y_center: Optional[float] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        rotation: Optional[float] = None,
        # deltas (optional)
        delta_x: float = 0.0,
        delta_y: float = 0.0,
        delta_w: float = 0.0,
        delta_h: float = 0.0,
        delta_theta: float = 0.0,
        # clamps
        min_width: float = 1e-6,
        min_height: float = 1e-6,
        annotator: str,
    ) -> RotatedBBox:
        """
        Update bbox geometry in the model; return the updated RotatedBBox.
        """
        openlabel_model: OpenLabel = self.project_state.annotation_config
        try:
            updated_bbox = openlabel_model.update_bbox(
                frame_key=frame_key,
                object_key=object_key,
                bbox_index=bbox_index,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
                rotation=rotation,
                delta_x=delta_x,
                delta_y=delta_y,
                delta_w=delta_w,
                delta_h=delta_h,
                delta_theta=delta_theta,
                min_width=min_width,
                min_height=min_height,
                annotator=annotator,
            )
            # place for side-effects, e.g. mark project dirty, log, mirror, etc.
            return updated_bbox
        except ValidationError as e:
            raise InvalidInputError(
                f"Invalid input data: {[err["msg"] for err in e.errors()]}", e
            )

    def cascade_bbox_edit(
        self,
        frame_start: int,
        object_key: Union[int, str],
        frame_end: Optional[int],
        annotator: str,
        *,
        # size values
        width: Optional[float] = None,
        height: Optional[float] = None,
        rotation: Optional[float] = None,
        # clamps
        min_width: float = 1e-6,
        min_height: float = 1e-6,
    ):
        """Apply size changes to frames from frame_start to frame_end (inclusive)."""
        # Get all frames that contain this object (after frame_start)
        frames_with_object = sorted(
            int(frame_key)
            for frame_key, frame_data in getattr(
                self.project_state.annotation_config, "frames", {}
            ).items()
            if getattr(frame_data, "objects", None) and object_key in frame_data.objects
        )
        frames_with_object = [
            frame
            for frame in frames_with_object
            if frame >= frame_start and frame <= frame_end
        ]

        if not frames_with_object:
            return []

        edited_frames = []
        for frame_num in frames_with_object:
            frame_objects = self.project_state.annotation_config.frames[
                str(frame_num)
            ].objects
            if object_key not in frame_objects:
                continue

            # Apply size changes only
            self.project_state.annotation_config.update_bbox(
                frame_key=str(frame_num),
                object_key=object_key,
                bbox_index=0,
                width=width,
                height=height,
                rotation=rotation,
                min_width=min_width,
                min_height=min_height,
                annotator=annotator,
            )
            edited_frames.append(frame_num)

        return edited_frames

    def delete_bbox(
        self, frame_key: int, object_key: str
    ) -> Optional[FrameLevelObject]:
        """No try-except here; let the controller handle exceptions."""

        if not self.project_state.annotation_config:
            return None
        return self.project_state.annotation_config.delete_bbox(frame_key, object_key)

    def restore_bbox(
        self, frame_key: int, object_key: str, frame_obj: FrameLevelObject
    ) -> None:
        if not self.project_state.annotation_config:
            return
        self.project_state.annotation_config.restore_bbox(
            frame_key, object_key, frame_obj
        )

    def list_object_ids(self) -> list[str]:
        """Return all object IDs currently defined in the annotation config."""
        config = self.project_state.annotation_config
        if not config or not getattr(config, "objects", None):
            return []

        def _sort_key(value: str) -> tuple[int, str]:
            text = str(value)
            if text.isdigit():
                return (0, f"{int(text):010d}")
            return (1, text)

        return sorted(config.objects.keys(), key=_sort_key)

    def frames_for_object(self, object_id: str) -> list[int]:
        """Return sorted frame indices that contain the specified object ID."""
        config = self.project_state.annotation_config
        if config is None:
            raise FrameNotFoundError("No annotation configuration loaded.")
        if object_id not in getattr(config, "objects", {}):
            raise ObjectNotFoundError(f"Object ID '{object_id}' does not exist.")

        frames_with_object: list[int] = []
        for frame_key, frame in config.frames.items():
            if object_id in getattr(frame, "objects", {}):
                try:
                    frames_with_object.append(int(frame_key))
                except (TypeError, ValueError):
                    continue
        return sorted(frames_with_object)

    def link_object_ids(
        self,
        primary_object_id: str,
        secondary_object_id: str,
    ) -> list[int]:
        """
        Replace all occurrences of secondary_object_id with primary_object_id.

        Args:
            primary_object_id: The object ID that should remain.
            secondary_object_id: The object ID to replace.

        Returns:
            A sorted list of affected frame indices.
        """
        if primary_object_id == secondary_object_id:
            raise ObjectLinkConflictError("Cannot link an object ID to itself.")

        config = self.project_state.annotation_config
        if config is None:
            raise FrameNotFoundError("No annotation configuration loaded.")

        objects_map = getattr(config, "objects", {})
        if primary_object_id not in objects_map:
            raise ObjectNotFoundError(
                f"Object ID '{primary_object_id}' does not exist."
            )
        if secondary_object_id not in objects_map:
            raise ObjectNotFoundError(
                f"Object ID '{secondary_object_id}' does not exist."
            )

        affected_frames: list[int] = []
        for frame_key, frame in config.frames.items():
            frame_objects = getattr(frame, "objects", {})
            if secondary_object_id in frame_objects:
                if primary_object_id in frame_objects:
                    raise ObjectLinkConflictError(
                        f"Frame {frame_key} already contains both object IDs."
                    )
                try:
                    affected_frames.append(int(frame_key))
                except (TypeError, ValueError):
                    continue

        if not affected_frames:
            return []

        for frame_key in list(config.frames.keys()):
            frame = config.frames[frame_key]
            frame_objects = getattr(frame, "objects", {})
            if secondary_object_id not in frame_objects:
                continue

            frame_obj = frame_objects.pop(secondary_object_id)
            frame_objects[primary_object_id] = frame_obj

        # Remove secondary metadata; primary metadata remains authoritative.
        if secondary_object_id in objects_map:
            objects_map.pop(secondary_object_id)

        return sorted(affected_frames)

    def mark_confidence_resolved(
        self, frame_number: int, object_id: str, annotator: str
    ) -> None:
        """Mark a warning/error as resolved by setting confidence to 1.0."""
        openlabel_model: OpenLabel = self.project_state.annotation_config
        if openlabel_model is None:
            raise FrameNotFoundError("No annotation configuration loaded.")

        frame_key = str(frame_number)
        frame = openlabel_model.frames.get(frame_key)
        if frame is None:
            raise FrameNotFoundError(f"Frame {frame_number} not found.")

        frame_obj = frame.objects.get(object_id)
        if frame_obj is None:
            raise ObjectNotFoundError(
                f"Object ID {object_id} not found in frame {frame_number}."
            )

        vec_entries = getattr(frame_obj.object_data, "vec", None)
        if vec_entries is None:
            vec_entries = []
            frame_obj.object_data.vec = vec_entries

        confidence_entry = None
        for entry in vec_entries:
            if getattr(entry, "name", None) == "confidence":
                confidence_entry = entry
                break

        if confidence_entry is None:
            confidence_entry = ConfidenceData(val=deque())
            vec_entries.append(confidence_entry)

        annotator_entry = AnnotatorData(val=deque())
        vec_entries.insert(0, annotator_entry)

        openlabel_model.update_annotator(
            annotator, annotator_entry, 1.0, confidence_entry
        )

    def _add_new_object(self, obj_type: str, obj_id: str) -> None:
        self.project_state.annotation_config.add_new_object(
            obj_type=obj_type, obj_id=obj_id
        )

    def _add_object_bbox(
        self, frame_number: int, bbox_coordinates: dict, obj_id: str, annotator: str
    ) -> None:
        """
        Service function to add new annotations to the config.
        """

        # Append new bounding box (under frames)
        self.project_state.annotation_config.append_object_bbox(
            frame_id=frame_number,
            bbox_coordinates=bbox_coordinates,
            obj_id=obj_id,
            annotator=annotator,
        )

    def _does_object_exist_in_frame(self, frame_number: int, object_id: str) -> bool:
        """Check if an object exists in a specific frame."""
        try:
            frame = self.project_state.annotation_config.frames[str(frame_number)]
        except KeyError:
            raise FrameNotFoundError(f"Frame number {frame_number} does not exist.")

        return object_id in [key for key in frame.objects.keys()]

    def _generate_new_object_id(self) -> str:
        return str(
            int(list(self.project_state.annotation_config.objects.keys())[-1]) + 1
        )

    def get_frame_tags(self) -> List[str]:
        """
        Return allowed frame tag labels from the ontology (Action class labels).
        """
        path = Path(str(get_ontology_path()))
        modified_time = path.stat().st_mtime
        key = (str(path), float(modified_time))

        if self._tag_cache_key == key and self._tag_cache_vals is not None:
            return self._tag_cache_vals

        labels = get_action_labels(path)
        if not labels:
            raise NoFrameLabelFoundError(
                "No Action labels found in ontology,\n"
                "please ensure an ontology file is selected in settings."
            )
        self._tag_cache_key, self._tag_cache_vals = key, labels
        return labels

    def add_frame_tag(self, tag_name: str, frame_start: int, frame_end: int) -> None:
        """
        Append a frame interval [frame_start, frame_end] to the given tag.
        Validates against ontology-derived allowed list.
        """
        tag = (tag_name or "").strip()
        allowed_tags = set(self.get_frame_tags())
        if tag not in allowed_tags:
            raise UnsportedTagTypeError(f"Unsupported tag '{tag}' (not in ontology).")
        if frame_start is None or frame_end is None:
            raise InvalidFrameRangeError("Both start and end frames must be provided.")
        if frame_start > frame_end:
            raise InvalidFrameRangeError("Start frame cannot be after end frame.")

        openlabel_config: OpenLabel = self.project_state.annotation_config
        if openlabel_config.actions is None:
            openlabel_config.actions = {}

        if tag not in openlabel_config.actions:
            openlabel_config.actions[tag] = ActionMetadata(
                name=tag, type="FrameTag", frame_intervals=[]
            )

        intervals = openlabel_config.actions[tag].frame_intervals or []
        intervals.append(FrameInterval(frame_start=frame_start, frame_end=frame_end))
        openlabel_config.actions[tag].frame_intervals = intervals

    def get_active_frame_tags(self, frame_index: int):
        """
        Return all frame-tag intervals that include the given frame index.

        Args:
            frame_index: 0-based index of the current frame.

        Returns:
            List of (tag_name, start, end) tuples where start <= frame_index <= end.
            Empty list if none.
        """
        openlabel_config = self.project_state.annotation_config
        if not openlabel_config or not openlabel_config.actions:
            return []

        active = []
        for tag_name, action in openlabel_config.actions.items():
            intervals = getattr(action, "frame_intervals", None) or []
            for iv in intervals:
                start = iv.frame_start
                end = iv.frame_end
                if start is None or end is None:
                    continue
                if int(start) <= frame_index <= int(end):
                    active.append((tag_name, int(start), int(end)))
        return active

    def bbox_types(self) -> Dict[str, List[str]]:
        """
        Return bbox type labels from the ontology (DynamicObject + StaticObject).

        Returns:
            {
              "DynamicObject": [...],
              "StaticObject":  [...],
            }

        Raises:
            OntologyNotFound / ValueError if ontology is missing/invalid.
        """
        try:
            path = Path(str(get_ontology_path())).resolve()
            modified_time = path.stat().st_mtime
            key = (str(path), float(modified_time))
        except FileNotFoundError:
            raise OntologyNotFound(
                "Ontology file not found, please select an ontology file in settings."
            )

        if self._bbox_cache_key == key and self._bbox_cache_vals is not None:
            return self._bbox_cache_vals

        vals = get_bbox_type_labels(path)
        self._bbox_cache_key, self._bbox_cache_vals = key, vals
        return vals

    def remove_frame_tag(self, tag_name: str, frame_start: int, frame_end: int) -> bool:
        """
        Remove the exact [frame_start, frame_end] interval for tag_name.
        If the tag has no intervals left, remove the tag entry.
        Returns True if something was removed.
        """
        openlabel_config = self.project_state.annotation_config
        if (
            not openlabel_config
            or not openlabel_config.actions
            or tag_name not in openlabel_config.actions
        ):
            return False

        action = openlabel_config.actions[tag_name]
        intervals = list(getattr(action, "frame_intervals", []) or [])
        new_intervals = [
            iv
            for iv in intervals
            if not (
                int(iv.frame_start) == int(frame_start)
                and int(iv.frame_end) == int(frame_end)
            )
        ]
        if len(new_intervals) == len(intervals):
            return False

        if new_intervals:
            action.frame_intervals = new_intervals
        else:
            del openlabel_config.actions[tag_name]
            if not openlabel_config.actions:
                openlabel_config.actions = None
        return True

    def delete_bboxes_by_object(
        self, object_key: str
    ) -> List[Tuple[int, FrameLevelObject]]:
        """
        Remove all FrameLevelObject instances for the given object_key across all frames.
        Returns a list of (frame_index, removed_frame_obj) for undo purposes.
        """
        out: List[Tuple[int, FrameLevelObject]] = []
        if not self.project_state.annotation_config:
            return out

        openlabel_annotation = self.project_state.annotation_config
        for frame_key in list(openlabel_annotation.frames.keys()):
            removed = openlabel_annotation.delete_bbox(frame_key, object_key)
            if removed is not None:
                out.append((int(frame_key), removed))
        return out

    def get_object_metadata(self, object_id: str) -> dict:
        openlabel_config = self.project_state.annotation_config
        if not openlabel_config or object_id not in openlabel_config.objects:
            raise ObjectNotFoundError(f"Object ID '{object_id}' does not exist.")
        meta = openlabel_config.objects[object_id]
        return {"id": object_id, "name": meta.name, "type": meta.type}

    def update_object_name(self, object_id: str, new_name: str) -> None:
        new_name = (new_name or "").strip()
        if not new_name:
            raise InvalidInputError("Name cannot be empty.")
        openlabel_config = self.project_state.annotation_config
        if not openlabel_config or object_id not in openlabel_config.objects:
            raise ObjectNotFoundError(f"Object ID '{object_id}' does not exist.")
        openlabel_config.objects[object_id].name = new_name

    def update_object_type(self, object_id: str, new_type: str) -> None:
        openlabel_config = self.project_state.annotation_config
        if not openlabel_config or object_id not in openlabel_config.objects:
            raise ObjectNotFoundError(f"Object ID '{object_id}' does not exist.")

        allowed = self.bbox_types()
        allowed_set = set(allowed.get("DynamicObject", [])) | set(
            allowed.get("StaticObject", [])
        )
        target_type_lower = (new_type or "").strip().lower()
        canonical = None
        for type in allowed_set:
            if type.lower() == target_type_lower:
                canonical = type
                break

        if canonical is None:
            raise InvalidInputError(f"Unsupported type '{new_type}' (not in ontology).")

        openlabel_config.objects[object_id].type = canonical

    def interpolate_annotations(
        self,
        object_id: str,
        start_frame: int,
        end_frame: int,
        # control_points: Dict[str, List],
        annotator: str,
    ) -> None:
        """Create interpolated annotations between two frames using Bezier splines"""
        if start_frame >= end_frame:
            raise InvalidFrameRangeError("Start frame must be before end frame")

        # Validate frames exist
        if str(start_frame) not in self.project_state.annotation_config.frames:
            raise FrameNotFoundError(f"Start frame {start_frame} does not exist")
        if str(end_frame) not in self.project_state.annotation_config.frames:
            raise FrameNotFoundError(f"End frame {end_frame} does not exist")

        # Validate object exists in frames
        if not self._does_object_exist_in_frame(start_frame, object_id):
            raise ObjectNotFoundError(
                f"Object {object_id} not found in start frame {start_frame}"
            )
        if not self._does_object_exist_in_frame(end_frame, object_id):
            raise ObjectNotFoundError(
                f"Object {object_id} not found in end frame {end_frame}"
            )

        start_bbox = self.get_bbox(start_frame, object_id)
        end_bbox = self.get_bbox(end_frame, object_id)

        # Calculate number of frames to interpolate
        num_frames = end_frame - start_frame - 1
        if num_frames <= 0:
            raise InvalidFrameRangeError(
                "No frames to interpolate between start and end frame"
            )

        # Extract center control points from the control_points dictionary
        # center_control_points = []
        # if 'center_x' in control_points and 'center_y' in control_points:
        #    xs = control_points['center_x']
        #    ys = control_points['center_y']
        #    if len(xs) == len(ys):
        #        center_control_points = list(zip(xs, ys))

        # print(xs, ys, center_control_points)

        # Interpolate bboxes for intermediate frames using the new spline method
        interpolated_bboxes = InterpolationService.interpolate_annotations(
            start_bbox,
            end_bbox,
            num_frames,
            #    center_control_points
        )

        # Save interpolated bboxes
        for i, bbox in enumerate(interpolated_bboxes):
            frame_num = start_frame + i + 1
            self._add_object_bbox(frame_num, bbox, object_id, annotator)

        # Store interpolation metadata for visual distinction
        for frame_num in range(start_frame + 1, end_frame):
            self.project_state.interpolation_metadata.add((frame_num, object_id))

    def is_interpolated(self, frame_num: int, object_id: str) -> bool:
        """Check if annotation is interpolated at given frame"""
        return (frame_num, object_id) in self.project_state.interpolation_metadata

    def add_object_relationship(
        self,
        relationship_type: str,
        ontology_uid: str,
        subject_object_id: str,
        object_object_id: str,
    ) -> str:
        """Add a new relationship between objects and return its ID"""
        openlabel = self.project_state.annotation_config

        # Calculate frame interval stuff.
        frame_intervals = self._calculate_relation_frame_interval(
            subject_object_id, object_object_id
        )

        return openlabel.add_object_relationship(
            relationship_type,
            ontology_uid,
            subject_object_id,
            object_object_id,
            frame_intervals,
        )

    def restore_object_relationship(
        self,
        relation_id: str,
        relationship_type: str,
        ontology_uid: str,
        subject_object_id: str,
        object_object_id: str,
    ) -> None:
        """Restore a previously deleted relationship."""
        openlabel = self.project_state.annotation_config

        # Calculate frame interval stuff.
        frame_intervals = self._calculate_relation_frame_interval(
            subject_object_id, object_object_id
        )

        openlabel.restore_object_relationship(
            relation_id,
            relationship_type,
            ontology_uid,
            subject_object_id,
            object_object_id,
            frame_intervals,
        )

    def get_object_relationships(self, object_id: str):
        """Get all relationships for a given object_id."""
        openlabel = self.project_state.annotation_config
        return openlabel.get_object_relationships(object_id)

    def delete_relationship(self, relation_id: str) -> bool:
        """Delete a relationship by its ID."""
        openlabel = self.project_state.annotation_config
        return openlabel.delete_relationship(relation_id)

    def _calculate_relation_frame_interval(
        self, subject_object_id: str, object_object_id: str
    ):
        """
        Given two objects of a relationship, calculate
        the frame intervals in which the relationship holds.
        """

        def _get_frame_intersection(subject_object_id: str, object_object_id: str):
            """calculate the intersection of two object annotations"""
            subject_object_frames = set(self.frames_for_object(subject_object_id))
            object_object_frames = set(self.frames_for_object(object_object_id))
            return subject_object_frames.intersection(object_object_frames)

        frame_intersection = _get_frame_intersection(
            subject_object_id, object_object_id
        )

        if not frame_intersection:
            return []

        sorted_frame_intersection = sorted(
            [int(frame_number) for frame_number in frame_intersection]
        )

        def _calculate_continuous_frame_intervals(
            sorted_frames: list[int],
        ) -> list[tuple]:
            """
            From a sorted list of frames, calculate the frame
            intervals.

            return a list of tuples (a, b), where
            a = interval start frame, b = interval end frame .
            """
            current_start_frame = sorted_frames[0]
            current_end_frame = sorted_frames[0]

            intervals: list[tuple] = []

            for frame in sorted_frames[1:]:
                if frame == current_end_frame + 1:
                    # Frame is continous,
                    # so step the current_end.
                    current_end_frame = frame
                else:
                    # Gap in frame interval
                    intervals.append((current_start_frame, current_end_frame))

                    # Set the start and end frame to
                    # the current frame, so we can loop for
                    # the next interval
                    current_start_frame = frame
                    current_end_frame = frame

            # Append the final interval.
            intervals.append((current_start_frame, current_end_frame))
            return intervals

        return _calculate_continuous_frame_intervals(sorted_frame_intersection)

    def get_frame_relationships(self, frame_index):
        openlabel = self.project_state.annotation_config

        frame_relationships = openlabel.get_relationships_from_frame(frame_index)

        if not frame_relationships:
            return None

        return [
            {
                "subject": relationship.rdf_subjects[0].uid,
                "relationship_type": relationship.type,
                "object": relationship.rdf_objects[0].uid,
            }
            for relationship in frame_relationships
        ]
