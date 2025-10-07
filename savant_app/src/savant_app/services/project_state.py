"""Class representing and managing the complete state of an annotation project."""

import json
from savant_app.models.OpenLabel import OpenLabel
from savant_app.utils import read_json
from typing import List, Tuple


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
        config = read_json(path)
        self.annotation_config = OpenLabel(**config["openlabel"])
        self.open_label_path = path

    def save_openlabel_config(self) -> None:
        """Save the adjusted OpenLabel configuration to a JSON file.

        Args:
            adjusted_config: The OpenLabel instance containing the adjusted configuration

        Raises:
            ValueError: If adjusted_config is not a valid OpenLabel instance
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

    def boxes_with_ids_for_frame(
        self, frame_idx: int
    ) -> List[Tuple[str, Tuple[float, float, float, float, float]]]:
        """
        Return [(object_id_str, (cx, cy, w, h, theta)), ...] for the given frame,
        in the same order you'll draw them in the overlay.
        Uses the same frame-key fallback logic as boxes_for_frame().
        """
        if not self.annotation_config or not self.annotation_config.frames:
            return []

        fkey = str(frame_idx)
        if fkey not in self.annotation_config.frames:
            alt = str(frame_idx + 1)
            if alt not in self.annotation_config.frames:
                return []
            fkey = alt

        out: List[Tuple[str, Tuple[float, float, float, float, float]]] = []
        frame = self.annotation_config.frames[fkey]

        # Preserve dict iteration order (same as you already draw)
        for object_id_str, fobj in frame.objects.items():
            for geom in fobj.object_data.rbbox:
                if geom.name != "shape":
                    continue
                rb = geom.val
                out.append(
                    (
                        object_id_str,
                        (
                            float(rb.x_center),
                            float(rb.y_center),
                            float(rb.width),
                            float(rb.height),
                            float(rb.rotation),
                        ),
                    )
                )
        return out

    def object_id_for_frame_index(self, frame_idx: int, overlay_index: int) -> str:
        """
        Map overlay row index -> object_id_str for the given frame.
        Raises IndexError if overlay_index is out of range.
        """
        pairs = self.boxes_with_ids_for_frame(frame_idx)
        if overlay_index < 0 or overlay_index >= len(pairs):
            raise IndexError(
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
                    errs.append(f"Action '{key}' interval #{interval_index+1} missing start or end")
                elif start_frame > end_frame:
                    errs.append(
                        f"Action '{key}' interval #{interval_index+1
                                                    } has start {start_frame} > end {end_frame}")
        if errs:
            raise ValueError("Invalid frame tags:\n- " + "\n- ".join(errs))
