"""Class representing and managing the complete state of an annotation project."""

import json
from savant_app.models.OpenLabel import OpenLabel
from savant_app.utils import read_json
from .exceptions import (
    OpenLabelFileNotValid,
    OverlayIndexError,
)
from pydantic import ValidationError
from typing import Dict, List, Tuple, Literal
from dataclasses import dataclass


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


class ProjectState:
    def __init__(self):
        self.annotation_config: OpenLabel = None
        self.open_label_path: str = None

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
        except json.decoder.JSONDecodeError:
            raise OpenLabelFileNotValid(
                "Please ensure a valid json file exists in the config dir."
            )
        except ValidationError as e:
            raise OpenLabelFileNotValid(
                "Config file contains incorrect OpenLabel syntax."
            ) from e

    def save_openlabel_config(self) -> None:
        """Save the adjusted OpenLabel configuration to a JSON file.

        Args:
            adjusted_config: The OpenLabel instance containing the adjusted configuration
        """
        # Save the configuration to a JSON file
        with open(self.open_label_path, "w") as f:
            f.write(
                json.dumps(
                    {"openlabel": self.annotation_config.model_dump(mode="json")}
                )
            )

    def get_actor_types(self) -> list[str]:
        """Get the list of all possible actor types.

        Returns:
            A list of actor type strings.
        """
        return self.__ACTORS.copy()

    def boxes_for_frame(
        self, frame_idx: int
    ) -> List[Tuple[float, float, float, float, float]]:
        """
        Return list of rotated boxes for a frame in video pixel coords:
        (cx, cy, w, h, theta_radians).
        """
        if not self.annotation_config or not self.annotation_config.frames:
            return []

        fkey = str(frame_idx)
        if fkey not in self.annotation_config.frames:
            alt = str(frame_idx + 1)
            if alt not in self.annotation_config.frames:
                return []
            fkey = alt

        out: List[Tuple[float, float, float, float, float]] = []
        frame = self.annotation_config.frames[fkey]
        for _obj_id, fobj in frame.objects.items():
            for geom in fobj.object_data.rbbox:
                if geom.name != "shape":
                    continue
                rb = geom.val
                out.append(
                    (
                        float(rb.x_center),
                        float(rb.y_center),
                        float(rb.width),
                        float(rb.height),
                        float(rb.rotation),
                    )
                )
        return out

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
                confidence_value: float | None = None
                vec_entries = getattr(frame_obj.object_data, "vec", []) or []
                for vec_entry in vec_entries:
                    if getattr(vec_entry, "name", None) != "confidence":
                        continue
                    confidence_values = getattr(vec_entry, "val", None) or []
                    if confidence_values:
                        confidence_value = float(confidence_values[0])
                    break

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
        return pairs[overlay_index][0]

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
                        f"Action '{key}' interval #{interval_index+1
                                                    } has start {start_frame} > end {end_frame}"
                    )
        if errs:
            raise ValueError("Invalid frame tags:\n- " + "\n- ".join(errs))
