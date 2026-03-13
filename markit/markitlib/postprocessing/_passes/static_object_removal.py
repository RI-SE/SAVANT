"""
StaticObjectRemovalPass - Static object removal postprocessing pass.
"""

import logging
from typing import Any, Dict
from collections import defaultdict

from ..base import PostprocessingPass
from ._common import _get_object_source_engine

logger = logging.getLogger(__name__)


class StaticObjectRemovalPass(PostprocessingPass):
    """Remove DynamicObject instances that don't move beyond threshold.
    NOTE: This pass will remove e.g. parked cars, pedestrians standing still, etc. If this
    is not desired, do not use this pass or use --static-mark to mark instead of remove.
    """

    def __init__(self, static_threshold: int = 20, mark_only: bool = False):
        """Initialize static object removal pass.

        Args:
            static_threshold: Movement threshold in pixels (default: 20)
            mark_only: If True, mark static objects instead of removing them (default: False)
        """
        self.static_threshold = static_threshold
        self.mark_only = mark_only
        self.objects_checked = 0
        self.objects_removed = 0
        self.objects_marked = 0
        self.frames_modified = 0
        self.removal_details = []
        self.marking_details = []

    def process(self, openlabel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove DynamicObjects that don't move beyond threshold.

        Args:
            openlabel_data: Complete OpenLabel data structure

        Returns:
            Modified OpenLabel data with static DynamicObjects removed
        """
        # Check if ontology path is set
        if not hasattr(self, "ontology_path") or not self.ontology_path:
            logger.warning("StaticObjectRemovalPass: Ontology path not set, skipping")
            return openlabel_data

        # Import ontology functions (done here to avoid circular imports)
        import os
        from savant_common.ontology import get_class_by_label

        # Check if ontology file exists
        if not os.path.exists(self.ontology_path):
            logger.warning(
                f"StaticObjectRemovalPass: Ontology file not found: {self.ontology_path}"
            )
            return openlabel_data

        frames = openlabel_data.get("openlabel", {}).get("frames", {})
        objects = openlabel_data.get("openlabel", {}).get("objects", {})

        # Build object-to-frames mapping
        object_frame_map = defaultdict(list)
        for frame_idx_str, frame_data in frames.items():
            frame_idx = int(frame_idx_str)
            frame_objects = frame_data.get("objects", {})
            for obj_id_str in frame_objects.keys():
                object_frame_map[obj_id_str].append(frame_idx)

        # Check each object
        objects_to_remove = []

        for obj_id, obj_data in objects.items():
            obj_type = obj_data.get("type", "")

            # Look up class in ontology
            try:
                class_info = get_class_by_label(
                    self.ontology_path, obj_type, case_sensitive=False
                )

                # Skip if not found in ontology
                if class_info is None:
                    logger.debug(
                        f"StaticObjectRemovalPass: Class '{obj_type}' not found in ontology, skipping object {obj_id}"
                    )
                    continue

                # Check if top-level class is DynamicObject
                top_level_class_name = class_info.get("top_level_class_name")
                if top_level_class_name != "DynamicObject":
                    continue

                # This is a DynamicObject, check movement
                self.objects_checked += 1

                # Get all frame appearances
                frame_list = object_frame_map.get(obj_id, [])
                if len(frame_list) == 0:
                    continue

                # Calculate movement
                x_positions = []
                y_positions = []

                for frame_idx in frame_list:
                    frame_str = str(frame_idx)
                    if frame_str not in frames:
                        continue

                    frame_obj = frames[frame_str]["objects"].get(obj_id)
                    if not frame_obj:
                        continue

                    try:
                        rbbox = frame_obj["object_data"]["rbbox"][0]["val"]
                        x, y = rbbox[0], rbbox[1]  # Center coordinates
                        x_positions.append(x)
                        y_positions.append(y)
                    except (KeyError, IndexError, TypeError) as e:
                        logger.debug(
                            f"StaticObjectRemovalPass: Error extracting position for object {obj_id} in frame {frame_idx}: {e}"
                        )
                        continue

                # Check if we have position data
                if len(x_positions) == 0 or len(y_positions) == 0:
                    continue

                # Calculate max-min movement in each dimension
                delta_x = max(x_positions) - min(x_positions)
                delta_y = max(y_positions) - min(y_positions)

                # Check if both dimensions are below threshold
                if (
                    delta_x <= self.static_threshold
                    and delta_y <= self.static_threshold
                ):
                    # Store first frame for marking
                    first_frame = min(frame_list)

                    if self.mark_only:
                        # OF-origin statics are always deleted even in mark_only mode —
                        # they are noise, not real objects that should be preserved.
                        source_engine = _get_object_source_engine(
                            obj_id, object_frame_map, frames
                        )
                        if source_engine == "oflow":
                            objects_to_remove.append(obj_id)
                            self.removal_details.append(
                                {
                                    "object_id": obj_id,
                                    "type": obj_type,
                                    "delta_x": delta_x,
                                    "delta_y": delta_y,
                                    "frame_count": len(frame_list),
                                }
                            )
                            logger.info(
                                f"Removed static OF object {obj_id} (type: {obj_type}) - "
                                f"movement: dx={delta_x}px, dy={delta_y}px, frames={len(frame_list)}"
                            )
                        else:
                            self.marking_details.append(
                                {
                                    "object_id": obj_id,
                                    "type": obj_type,
                                    "delta_x": delta_x,
                                    "delta_y": delta_y,
                                    "frame_count": len(frame_list),
                                    "first_frame": first_frame,
                                }
                            )
                            logger.info(
                                f"Marked static object {obj_id} (type: {obj_type}) - "
                                f"movement: dx={delta_x}px, dy={delta_y}px, frames={len(frame_list)}"
                            )
                    else:
                        objects_to_remove.append(obj_id)
                        self.removal_details.append(
                            {
                                "object_id": obj_id,
                                "type": obj_type,
                                "delta_x": delta_x,
                                "delta_y": delta_y,
                                "frame_count": len(frame_list),
                            }
                        )
                        logger.info(
                            f"Removed static object {obj_id} (type: {obj_type}) - "
                            f"movement: dx={delta_x}px, dy={delta_y}px, frames={len(frame_list)}"
                        )

            except Exception as e:
                logger.warning(
                    f"StaticObjectRemovalPass: Error processing object {obj_id}: {e}"
                )
                continue

        # Mark non-oflow static objects (when mark_only=True)
        if self.mark_only:
            for detail in self.marking_details:
                obj_id = detail["object_id"]
                first_frame = detail["first_frame"]

                if obj_id not in objects:
                    continue

                if "object_data" not in objects[obj_id]:
                    objects[obj_id]["object_data"] = {}

                if "vec" not in objects[obj_id]["object_data"]:
                    objects[obj_id]["object_data"]["vec"] = []

                vec_list = objects[obj_id]["object_data"]["vec"]
                vec_list.append({"name": "staticdynamic", "val": [first_frame]})

                self.objects_marked += 1

        # Remove static objects: always when mark_only=False; OF-origin only when mark_only=True
        for obj_id in objects_to_remove:
            if obj_id in objects:
                del objects[obj_id]
                self.objects_removed += 1

            for frame_idx_str, frame_data in frames.items():
                frame_objects = frame_data.get("objects", {})
                if obj_id in frame_objects:
                    del frame_objects[obj_id]
                    self.frames_modified += 1

        return openlabel_data

    def get_statistics(self) -> Dict[str, Any]:
        """Get static object removal/marking statistics.

        Returns:
            Dictionary with removal or marking statistics
        """
        return {
            "objects_checked": self.objects_checked,
            "objects_removed": self.objects_removed,
            "objects_marked": self.objects_marked,
            "frames_modified": self.frames_modified,
        }
