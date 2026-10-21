"""Class representing and managing the complete state of an annotation project."""

import hashlib
import json
import logging
import os
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple

from edit.models.OpenLabel import OpenLabel
from savant_common.openlabel import EDIT_METADATA_KEY, contiguous_intervals, source_confidence
from edit.utils import read_json
from edit.version import get_version
from pydantic import ValidationError

from .exceptions import OpenLabelFileNotValid, OverlayIndexError
from .types import VideoMetadata

logger = logging.getLogger(__name__)


@dataclass
class BBoxDimensionData:
    cx: float
    cy: float
    width: float
    height: float
    theta: float


@dataclass
class FrameBBoxData:
    object_id: str
    object_type: str
    bbox: BBoxDimensionData
    confidence: float | None = None


@dataclass
class FrameConfidenceIssue:
    object_id: str
    confidence: float
    severity: Literal["warning", "error"]


@dataclass
class FrameTagDetail:
    frame_index: int
    category: Literal["frame_tag", "object_tag"]
    tag_name: str
    object_id: str | None = None
    object_name: str | None = None
    description: str | None = None


def _normalize_tag_frames(value) -> list[int]:
    frames: set[int] = set()
    if value is None:
        return []
    if isinstance(value, (int, float)):
        if int(value) >= 0:
            frames.add(int(value))
        return sorted(frames)
    if isinstance(value, str):
        try:
            num = int(value.strip())
        except ValueError:
            return []
        if num >= 0:
            frames.add(num)
        return sorted(frames)
    if isinstance(value, (list, tuple, set)):
        for item in value:
            frames.update(_normalize_tag_frames(item))
        return sorted(frames)
    if hasattr(value, "__iter__") and not isinstance(value, (bytes, bytearray)):
        try:
            for item in value:
                frames.update(_normalize_tag_frames(item))
        except TypeError:
            pass
    return sorted(frames)


class ProjectState:
    def __init__(self):
        self.annotation_config: OpenLabel = None
        self.open_label_path: str = None
        self.video_metadata: VideoMetadata = VideoMetadata()
        self.interpolation_metadata = (
            set()
        )  # Stores frame numbers with interpolated metadata
        self._saved_digest: str | None = None  # digest of the config as last loaded/saved

    def load_openlabel_config(self, path: str) -> None:
        """Load and validate OpenLabel configuration from JSON file.
        Args:
            path: Path to JSON file containing a SAVANT OpenLabel configuration

        Raises:
            FileNotFoundError: If specified path doesn't exist
            ValidationError: If configuration fails OpenLabel schema validation
            ValueError: If path does not point to a JSON file.

        Initializes:
            self.open_label: New OpenLabel instance with loaded configuration
        """
        try:
            config = read_json(path)
            self.annotation_config = OpenLabel(**config["openlabel"])
            self.open_label_path = path
            self._saved_digest = self._config_digest()
        except KeyError as e:
            raise OpenLabelFileNotValid(
                f"File does not appear to be a SAVANT OpenLabel file "
                f"(missing key: {e}). Please select a valid project JSON."
            ) from e
        except json.decoder.JSONDecodeError:
            raise OpenLabelFileNotValid(
                "Please ensure a valid json file exists in the config dir."
            )
        except ValidationError as e:
            raise OpenLabelFileNotValid(
                "Config file contains incorrect OpenLabel syntax."
            ) from e

    def save_openlabel_config(self) -> None:
        """Save the OpenLabel configuration; the file is replaced only after a complete write."""
        payload = {"openlabel": self._export_config()}
        target = Path(self.open_label_path)
        fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2)
            if target.exists():
                shutil.copymode(target, tmp_name)
            os.replace(tmp_name, target)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        self._saved_digest = self._config_digest()

    def _export_config(self) -> dict:
        """Serialize the config with object frame_intervals rebuilt from the boxes.

        Edits change boxes but never the intervals markit wrote, so they are recomputed.
        Objects left without any box are dropped together with relations that reference
        them; downstream tools would silently ignore them anyway. Only the written copy
        changes: the in-memory model keeps them, so undo can still restore their boxes.
        """
        data = self.annotation_config.model_dump(mode="json")
        data["metadata"][EDIT_METADATA_KEY] = {"version": get_version()}
        frames_by_object = defaultdict(list)
        for frame_key, frame in (data.get("frames") or {}).items():
            if not str(frame_key).isdigit():
                continue
            for obj_id in (frame.get("objects") or {}):
                frames_by_object[obj_id].append(int(frame_key))

        objects = data.get("objects") or {}
        orphans = {obj_id for obj_id in objects if obj_id not in frames_by_object}
        for obj_id in orphans:
            del objects[obj_id]
        for obj_id, metadata in objects.items():
            metadata["frame_intervals"] = contiguous_intervals(frames_by_object[obj_id])

        relations = data.get("relations") or {}
        dangling = [
            rel_id for rel_id, rel in relations.items()
            if any(item["uid"] in orphans for item in rel["rdf_subjects"] + rel["rdf_objects"])
        ]
        for rel_id in dangling:
            del relations[rel_id]

        if orphans:
            logger.warning(
                f"Not saving {len(orphans)} object(s) without any bounding box: "
                f"{sorted(orphans)}; {len(dangling)} relation(s) referencing them dropped"
            )
        return data

    def _config_digest(self) -> str | None:
        if self.annotation_config is None:
            return None
        return hashlib.sha256(self.annotation_config.model_dump_json().encode()).hexdigest()

    def has_unsaved_changes(self) -> bool:
        """Compare against the last load/save, so edits made outside the undo stack count too."""
        return self._config_digest() != self._saved_digest

    def get_tag_categories(self) -> dict[str, dict[str, list[int]]]:
        """Collect object metadata tags and action tags mapped to their frames."""
        if not self.annotation_config:
            return {"frame": {}, "object": {}}

        tags: dict[str, dict[str, set[int]]] = {
            "frame": {},
            "object": {},
        }

        def ensure_category_entry(category: str, name: str) -> set[int]:
            category_map = tags.setdefault(category, {})
            return category_map.setdefault(name, set())

        for metadata in self.annotation_config.objects.values():
            object_data = getattr(metadata, "object_data", None)
            if not object_data:
                continue
            vec_entries = getattr(object_data, "vec", None) or []
            for entry in vec_entries:
                name = getattr(entry, "name", None)
                if not name:
                    continue
                frames = _normalize_tag_frames(getattr(entry, "val", None))
                if frames:
                    bucket = ensure_category_entry("object", str(name))
                    bucket.update(frames)

        actions = getattr(self.annotation_config, "actions", None) or {}
        for action in actions.values():
            action_name = getattr(action, "name", None)
            if not action_name:
                continue
            intervals = getattr(action, "frame_intervals", None) or []
            starts: set[int] = set()
            for interval in intervals:
                start_val = getattr(interval, "frame_start", None)
                if isinstance(start_val, (int, float)) and int(start_val) >= 0:
                    starts.add(int(start_val))
            if starts:
                bucket = ensure_category_entry("frame", str(action_name))
                bucket.update(starts)

        normalized: dict[str, dict[str, list[int]]] = {
            "frame": {},
            "object": {},
        }
        for category, name_map in tags.items():
            normalized[category] = {
                name: sorted(values)
                for name, values in sorted(
                    name_map.items(), key=lambda item: item[0].lower()
                )
            }

        return normalized

    def get_tag_frame_details(self) -> dict[int, list[dict]]:
        """Return descriptive tag details for each frame."""
        if not self.annotation_config:
            return {}

        details: dict[int, list[FrameTagDetail]] = {}

        def add_detail(frame_idx: int, detail: FrameTagDetail) -> None:
            details.setdefault(int(frame_idx), []).append(detail)

        for object_id, metadata in self.annotation_config.objects.items():
            object_data = getattr(metadata, "object_data", None)
            if not object_data:
                continue
            vec_entries = getattr(object_data, "vec", None) or []
            for entry in vec_entries:
                name = getattr(entry, "name", None)
                if not name:
                    continue
                frames = _normalize_tag_frames(getattr(entry, "val", None))
                if not frames:
                    continue
                for frame in frames:
                    add_detail(
                        frame,
                        FrameTagDetail(
                            frame_index=frame,
                            category="object_tag",
                            tag_name=str(name),
                            object_id=str(object_id),
                            object_name=getattr(metadata, "name", None),
                            description=str(name),
                        ),
                    )

        actions = getattr(self.annotation_config, "actions", None) or {}
        for action in actions.values():
            action_name = getattr(action, "name", None)
            if not action_name:
                continue
            intervals = getattr(action, "frame_intervals", None) or []
            for interval in intervals:
                frame_start = getattr(interval, "frame_start", None)
                if not isinstance(frame_start, (int, float)):
                    continue
                frame_start = int(frame_start)
                if frame_start < 0:
                    continue
                frame_end = getattr(interval, "frame_end", None)
                description = (
                    f"{action_name} frames {frame_start}-{frame_end}"
                    if isinstance(frame_end, (int, float)) and frame_end >= frame_start
                    else f"{action_name} frame {frame_start}"
                )
                add_detail(
                    frame_start,
                    FrameTagDetail(
                        frame_index=frame_start,
                        category="frame_tag",
                        tag_name=str(action_name),
                        object_id=None,
                        object_name="Frame",
                        description=description,
                    ),
                )

        return {
            frame: [detail.__dict__ for detail in entries]
            for frame, entries in details.items()
        }

    def boxes_for_frame(
        self, frame_idx: int
    ) -> List[Tuple[float, float, float, float, float]]:
        """
        Return list of rotated boxes for a frame in video pixel coords:
        (cx, cy, w, h, theta_radians).
        """
        boxes = self.boxes_with_ids_for_frame(frame_idx)
        return [
            (
                bbox.bbox.cx,
                bbox.bbox.cy,
                bbox.bbox.width,
                bbox.bbox.height,
                bbox.bbox.theta,
            )
            for bbox in boxes
        ]

    def boxes_with_ids_for_frame(self, frame_idx: int) -> List[FrameBBoxData]:
        results: List[FrameBBoxData] = []

        frame = self.annotation_config.frames.get(str(frame_idx))
        if not frame:
            return []

        for object_id, frame_obj in frame.objects.items():
            metadata = self.annotation_config.objects.get(object_id)
            object_type = metadata.type if metadata else "unknown"

            confidence_value: float | None = None
            vec_entries = getattr(frame_obj.object_data, "vec", []) or []
            for vec_entry in vec_entries:
                if getattr(vec_entry, "name", None) != "confidence":
                    continue
                confidence_values = getattr(vec_entry, "val", None) or []
                if confidence_values:
                    confidence_value = float(confidence_values[0])
                break

            for geometry_data in frame_obj.object_data.rbbox:
                if geometry_data.name != "shape":
                    continue
                rbbox_dimensions = geometry_data.val
                bbox_dimension_data = BBoxDimensionData(
                    cx=rbbox_dimensions.x_center,
                    cy=rbbox_dimensions.y_center,
                    width=rbbox_dimensions.width,
                    height=rbbox_dimensions.height,
                    theta=rbbox_dimensions.rotation,
                )
                results.append(
                    FrameBBoxData(
                        object_id=object_id,
                        object_type=object_type,
                        bbox=bbox_dimension_data,
                        confidence=confidence_value,
                    )
                )

        return results

    def confidence_issues(
        self,
        *,
        warning_range: tuple[float, float],
        error_range: tuple[float, float],
        show_warnings: bool = True,
        show_errors: bool = True,
    ) -> Dict[int, List[FrameConfidenceIssue]]:
        """
        Collect confidence-based warnings/errors for each frame.

        Returns:
            Dict mapping frame index to a list of FrameConfidenceIssue objects.
            Frames without warnings or errors are omitted.
        """
        issues_by_frame: Dict[int, List[FrameConfidenceIssue]] = {}
        if not self.annotation_config or not self.annotation_config.frames:
            return issues_by_frame

        warning_min, warning_max = (float(warning_range[0]), float(warning_range[1]))
        error_min, error_max = (float(error_range[0]), float(error_range[1]))

        for frame_key, frame in self.annotation_config.frames.items():
            try:
                frame_index = int(frame_key)
            except (TypeError, ValueError):
                continue

            frame_issues: List[FrameConfidenceIssue] = []
            for object_id, frame_obj in frame.objects.items():
                histories = {
                    getattr(vec_entry, "name", None): getattr(vec_entry, "val", None)
                    for vec_entry in getattr(frame_obj.object_data, "vec", []) or []
                }
                # Judge the detector's score, not the fixed value markit housekeeping puts on top
                confidence_value = source_confidence(histories.get("annotator"), histories.get("confidence"))

                if confidence_value is None:
                    continue

                severity: Literal["warning", "error"] | None = None
                if show_errors and error_min <= confidence_value <= error_max:
                    severity = "error"
                elif show_warnings and warning_min <= confidence_value <= warning_max:
                    severity = "warning"
                else:
                    continue

                frame_issues.append(
                    FrameConfidenceIssue(
                        object_id=object_id,
                        confidence=confidence_value,
                        severity=severity,
                    )
                )

            if frame_issues:
                issues_by_frame[frame_index] = frame_issues

        return issues_by_frame

    def object_id_for_frame_index(self, frame_idx: int, overlay_index: int) -> str:
        """
        Map overlay row index -> object_id_str for the given frame.
        Raises IndexError if overlay_index is out of range.
        """
        pairs = self.boxes_with_ids_for_frame(frame_idx)
        if overlay_index < 0 or overlay_index >= len(pairs):
            raise OverlayIndexError(
                f"overlay_index {overlay_index} out of range for frame {frame_idx}"
            )
        return pairs[overlay_index].object_id

    def validate_before_save(self) -> None:
        """
        Validate action frame intervals:
        - start/end present
        - start <= end
        """
        if not self.annotation_config or not self.annotation_config.actions:
            return
        errs = []
        for key, action in self.annotation_config.actions.items():
            if not action.frame_intervals:
                continue
            for interval_index, frame_interval in enumerate(action.frame_intervals):
                start_frame = frame_interval.frame_start
                end_frame = frame_interval.frame_end
                if start_frame is None or end_frame is None:
                    errs.append(
                        f"Action '{key}' interval #{interval_index+1} missing start or end"
                    )
                elif start_frame > end_frame:
                    errs.append(
                        f"""
                        Action '{key}' interval #{interval_index + 1} has
                        start {start_frame} > end {end_frame}
                        """
                    )
        if errs:
            raise ValueError("Invalid frame tags:\n- " + "\n- ".join(errs))
