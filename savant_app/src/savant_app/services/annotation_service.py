from .project_state import ProjectState
from .exceptions import (
    ObjectInFrameError,
    ObjectNotFoundError,
    FrameNotFoundError,
    InvalidFrameRangeError,
    InvalidInputError,
    BBoxNotFoundError,
    NoFrameLabelFoundError,
    UnsportedTagTypeError,
)
from typing import Optional, Union, Tuple
from savant_app.models.OpenLabel import OpenLabel, RotatedBBox
from savant_app.models.OpenLabel import FrameLevelObject
from pydantic import ValidationError
from savant_app.models.OpenLabel import ActionMetadata, FrameInterval
from savant_app.frontend.utils.settings_store import get_ontology_path
from savant_app.frontend.utils.ontology_utils import get_action_labels
from typing import Dict, List
from savant_app.frontend.utils.ontology_utils import get_bbox_type_labels
from pathlib import Path


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
        self, frame_number: int, obj_type: str, coordinates: tuple
    ) -> None:
        """Handles both, the creation of a new object and adding a bbox for it."""
        try:
            # Generate ID
            obj_id = self._generate_new_object_id()

            # Add new object and bbox
            self._add_new_object(obj_type=obj_type, obj_id=obj_id)
            self._add_object_bbox(
                frame_number=frame_number, bbox_coordinates=coordinates, obj_id=obj_id
            )
        except ValidationError as e:
            errors = "; ".join(
                f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()
            )
            raise InvalidInputError(f"Invalid input data: {errors}", e)

    def create_existing_object_bbox(
        self, frame_number: int, coordinates: tuple, object_name: str
    ) -> None:
        """Handles adding a bbox for an existing object."""

        # verify the object exists
        # if not self._does_object_exist(object_name):
        #    raise ObjectNotFoundError(f"Object: {object_name} does not exist.")
        try:
            object_id = self._get_objectid_by_name(object_name)

            # Verify if current frame already has the object
            if self._does_object_exist_in_frame(frame_number, object_id):
                raise ObjectInFrameError(
                    f"Object ID {object_name} already has a bbox in frame {frame_number}."
                )

            self._add_object_bbox(
                frame_number=frame_number,
                bbox_coordinates=coordinates,
                obj_id=object_id,
            )
        except ValidationError as e:
            errors = "; ".join(
                f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()
            )
            raise InvalidInputError(f"Invalid input data: {errors}", e)

    def get_active_objects(self, frame_number: int) -> list[dict]:
        """Get a list of active objects for the given frame number.

        Note, filtering of active objects is done at the service level,
        due to being related to the context of the application,
        whereas the model handles what the data is and manipulating itself.
        """
        # Get frame object keys
        frame = self.project_state.annotation_config.frames[str(frame_number)]
        active_object_keys = frame.objects.keys()

        return [
            {"type": obj.type, "name": obj.name}
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
        return [
            global_objects[obj_id].name
            for obj_id in object_ids
            if obj_id in global_objects
        ]

    def get_bbox(
        self,
        frame_key: Union[int, str],
        object_key: Union[int, str],
        bbox_index: int = 0,
    ) -> RotatedBBox:
        """
        Read a bbox from the current annotation config (model).
        """
        openlabel_model: OpenLabel = self.project_state.annotation_config
        try:
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
            int(frame_key) for frame_key, frame_data in getattr(self.project_state.annotation_config, "frames", {}).items()
            if getattr(frame_data, "objects", None) and object_key in frame_data.objects
        )
        frames_with_object = [frame for frame in frames_with_object if frame >= frame_start and frame <= frame_end]
        
        if not frames_with_object:
            return []

        edited_frames = []
        for frame_num in frames_with_object:
            #frame_str = str(frame_num)
            #if frame_str not in self.project_state.annotation_config.frames:
            #    continue
            frame_objects = self.project_state.annotation_config.frames[str(frame_num)].objects
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

    def _add_new_object(self, obj_type: str, obj_id: str) -> None:
        self.project_state.annotation_config.add_new_object(
            obj_type=obj_type, obj_id=obj_id
        )

    def _add_object_bbox(
        self, frame_number: int, bbox_coordinates: dict, obj_id: str
    ) -> None:
        """
        Service function to add new annotations to the config.
        """
        # Temporary hard code of annotater and confidence score.
        annotater_data = {"val": ["example_name"]}
        confidence_data = {"val": [0.9]}

        # Append new bounding box (under frames)
        self.project_state.annotation_config.append_object_bbox(
            frame_id=frame_number,
            bbox_coordinates=bbox_coordinates,
            confidence_data=confidence_data,
            annotater_data=annotater_data,
            obj_id=obj_id,
        )

    def _does_object_exist(self, object_name: str) -> bool:
        """Check if an object exists in the annotation config."""
        existing_ids = [
            value.name
            for _, value in self.project_state.annotation_config.objects.items()
        ]
        return object_name in existing_ids

    def _does_object_exist_in_frame(self, frame_number: int, object_id: str) -> bool:
        """Check if an object exists in a specific frame."""
        try:
            frame = self.project_state.annotation_config.frames[str(frame_number)]
        except KeyError:
            raise FrameNotFoundError(f"Frame number {frame_number} does not exist.")

        return object_id in [key for key in frame.objects.keys()]

    def _get_objectid_by_name(self, object_name: str):
        """Get the object ID given the object name."""
        for key, value in self.project_state.annotation_config.objects.items():
            if value.name == object_name:
                return key
        raise ObjectNotFoundError(f"Object: {object_name}, does not exist.")

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
            raise NoFrameLabelFoundError(f"No Action labels found in ontology: {path}")
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
            FileNotFoundError / ValueError if ontology is missing/invalid.
        """
        path = Path(str(get_ontology_path())).resolve()
        modified_time = path.stat().st_mtime
        key = (str(path), float(modified_time))

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
        ol = self.project_state.annotation_config
        if not ol or not ol.actions or tag_name not in ol.actions:
            return False

        action = ol.actions[tag_name]
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
            del ol.actions[tag_name]
            if not ol.actions:
                ol.actions = None
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
