"""
_common - Shared constants and helper functions for postprocessing passes.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Confidence value used for all housekeeping operations
HOUSEKEEPING_CONFIDENCE = 0.8888


def update_housekeeping_annotator(obj_data: Dict[str, Any], tag: str) -> None:
    """Update annotator field to add a housekeeping tag.

    Combines all housekeeping tags into a single entry: markit_housekeeping(rot,90fix,smooth)
    Only adds confidence value when creating the housekeeping entry (first tag).

    Args:
        obj_data: Object data dictionary containing object_data.vec
        tag: Short tag to add (e.g., "rot", "90fix", "smooth")
    """
    import re

    vec_list = obj_data.get("object_data", {}).get("vec", [])
    if not vec_list:
        # No vec list, create one with housekeeping entry
        obj_data.setdefault("object_data", {})["vec"] = [
            {"name": "annotator", "val": [f"markit_housekeeping({tag})"]},
            {"name": "confidence", "val": [HOUSEKEEPING_CONFIDENCE]},
        ]
        return

    # Find annotator and confidence entries
    annotator_item = None
    confidence_item = None
    for vec_item in vec_list:
        if vec_item.get("name") == "annotator":
            annotator_item = vec_item
        elif vec_item.get("name") == "confidence":
            confidence_item = vec_item

    if annotator_item is None:
        # No annotator field, add new housekeeping entry at beginning
        vec_list.insert(0, {"name": "annotator", "val": [f"markit_housekeeping({tag})"]})
        if confidence_item:
            confidence_item["val"].insert(0, HOUSEKEEPING_CONFIDENCE)
        else:
            vec_list.insert(1, {"name": "confidence", "val": [HOUSEKEEPING_CONFIDENCE]})
        return

    # Look for existing markit_housekeeping(...) entry
    annotator_vals = annotator_item.get("val", [])
    housekeeping_idx = None
    housekeeping_tags = []

    for i, val in enumerate(annotator_vals):
        match = re.match(r"markit_housekeeping\(([^)]*)\)", val)
        if match:
            housekeeping_idx = i
            existing_tags = match.group(1)
            if existing_tags:
                housekeeping_tags = [t.strip() for t in existing_tags.split(",")]
            break

    if housekeeping_idx is not None:
        # Found existing housekeeping entry - add tag if not present
        if tag not in housekeeping_tags:
            housekeeping_tags.append(tag)
            annotator_vals[housekeeping_idx] = f"markit_housekeeping({','.join(housekeeping_tags)})"
        # Don't add confidence - it was already added when housekeeping was created
    else:
        # No housekeeping entry yet - create one at position 0
        annotator_vals.insert(0, f"markit_housekeeping({tag})")
        # Add corresponding confidence at position 0
        if confidence_item:
            confidence_item["val"].insert(0, HOUSEKEEPING_CONFIDENCE)
        else:
            # Find annotator position to insert confidence after it
            for i, vec_item in enumerate(vec_list):
                if vec_item.get("name") == "annotator":
                    vec_list.insert(i + 1, {"name": "confidence", "val": [HOUSEKEEPING_CONFIDENCE]})
                    break


def _get_object_source_engine(
    obj_id: str,
    object_frame_map: Dict[str, List[int]],
    frames: Dict[str, Any],
) -> str:
    """Get the source detection engine for an object from its first non-gap frame.

    Inspects the annotator vec entry, skipping housekeeping/gap frames, to determine
    whether the object originated from yolo, oflow, aruco, or is unknown.

    Args:
        obj_id: Object ID to look up.
        object_frame_map: Mapping of object IDs to their frame index lists.
        frames: Full frame data dictionary from OpenLabel.

    Returns:
        Source engine name: "yolo", "oflow", "aruco", or "unknown".
    """
    import re

    for frame_idx in object_frame_map.get(obj_id, []):
        frame_str = str(frame_idx)
        frame_objects = frames[frame_str].get("objects", {})

        if obj_id not in frame_objects:
            continue

        try:
            vec_list = frame_objects[obj_id]["object_data"]["vec"]
            for vec_item in vec_list:
                if vec_item.get("name") == "annotator":
                    annotators = vec_item.get("val", [])
                    for ann in reversed(annotators):
                        if "markit_housekeeping" in ann:
                            match = re.match(r"markit_housekeeping\(([^)]*)\)", ann)
                            if match and match.group(1) == "gap":
                                continue
                            continue
                        if "yolo" in ann.lower():
                            return "yolo"
                        elif "oflow" in ann.lower() or "optical_flow" in ann.lower():
                            return "oflow"
                        elif "aruco" in ann.lower():
                            return "aruco"
                    break
        except (KeyError, IndexError):
            pass

    return "unknown"
