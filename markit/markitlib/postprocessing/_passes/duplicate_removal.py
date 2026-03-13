"""
DuplicateRemovalPass - Duplicate removal postprocessing pass.
"""

import logging
from typing import Any, Dict, List, Optional
from collections import defaultdict
import numpy as np

from ..base import PostprocessingPass
from ...geometry import BBoxOverlapCalculator
from ._common import _get_object_source_engine

logger = logging.getLogger(__name__)


class DuplicateRemovalPass(PostprocessingPass):
    """Remove duplicate bounding boxes based on IOU threshold.

    Detects objects that overlap significantly across multiple frames and removes
    the lower-priority one. Priority order: yolo > aruco > oflow > unknown.
    """

    # Source engine priority (higher = keep)
    ENGINE_PRIORITY = {
        "yolo": 3,
        "aruco": 2,
        "oflow": 1,
        "optical_flow": 1,
        "unknown": 0,
    }

    def __init__(
        self,
        avg_iou_threshold: float = 0.3,
        min_iou_threshold: float = 0.2,
        min_shared_ratio: float = 0.5,
        iomin_threshold: float = 0.7,
    ):
        """Initialize duplicate removal pass.

        Args:
            avg_iou_threshold: Average IoU across shared frames to consider duplicate.
            min_iou_threshold: Minimum IoU in any shared frame to consider duplicate.
            min_shared_ratio: Minimum ratio of shared frames to the shorter object's
                total frames. Prevents removing objects that only briefly overlap.
            iomin_threshold: Average intersection-over-minimum-area threshold.
                Detects containment where a large bbox envelops a smaller one,
                which suppresses IoU despite being the same object.
        """
        self.objects_deleted = 0
        self.duplicate_pairs_found = 0
        self.frames_modified = 0
        self.frames_merged = 0
        self.iou_calculator = BBoxOverlapCalculator()
        self.deletion_details = []
        self.avg_iou_threshold = avg_iou_threshold
        self.min_iou_threshold = min_iou_threshold
        self.min_shared_ratio = min_shared_ratio
        self.iomin_threshold = iomin_threshold

    def process(self, openlabel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove duplicate objects based on IOU analysis.

        Args:
            openlabel_data: Complete OpenLabel data structure

        Returns:
            Modified OpenLabel data with duplicates removed
        """
        frames = openlabel_data.get("openlabel", {}).get("frames", {})
        objects = openlabel_data.get("openlabel", {}).get("objects", {})

        object_frame_map = defaultdict(list)

        for frame_idx_str, frame_data in frames.items():
            frame_idx = int(frame_idx_str)
            frame_objects = frame_data.get("objects", {})

            for obj_id_str in frame_objects.keys():
                object_frame_map[obj_id_str].append(frame_idx)

        # Precompute frame ranges for fast temporal overlap check
        object_frame_range = {}
        for obj_id, frame_list in object_frame_map.items():
            if frame_list:
                object_frame_range[obj_id] = (min(frame_list), max(frame_list))

        objects_to_delete = set()
        object_ids = list(objects.keys())
        n_objects = len(object_ids)
        logger.info(f"DuplicateRemoval: checking {n_objects * (n_objects - 1) // 2} pairs from {n_objects} objects")

        for i in range(n_objects):
            for j in range(i + 1, n_objects):
                obj_a = object_ids[i]
                obj_b = object_ids[j]

                if obj_a in objects_to_delete or obj_b in objects_to_delete:
                    continue

                # Skip pairs with no temporal overlap
                range_a = object_frame_range.get(obj_a)
                range_b = object_frame_range.get(obj_b)
                if not range_a or not range_b:
                    continue
                if range_a[1] < range_b[0] or range_b[1] < range_a[0]:
                    continue

                if self._are_duplicates(obj_a, obj_b, object_frame_map, frames):
                    self.duplicate_pairs_found += 1

                    obj_to_delete = self._choose_object_to_delete(
                        obj_a, obj_b, object_frame_map, frames
                    )
                    obj_to_keep = obj_b if obj_to_delete == obj_a else obj_a
                    objects_to_delete.add(obj_to_delete)

                    frames_list = sorted(object_frame_map[obj_to_delete])
                    engine_deleted = self._get_source_engine(obj_to_delete, object_frame_map, frames)
                    engine_kept = self._get_source_engine(obj_to_keep, object_frame_map, frames)
                    self.deletion_details.append(
                        {
                            "deleted_object": obj_to_delete,
                            "kept_object": obj_to_keep,
                            "deleted_engine": engine_deleted,
                            "kept_engine": engine_kept,
                            "frame_start": frames_list[0] if frames_list else None,
                            "frame_end": frames_list[-1] if frames_list else None,
                        }
                    )

        for obj_id in objects_to_delete:
            # Find which object this duplicate should merge into
            detail = next(
                (d for d in self.deletion_details if d["deleted_object"] == obj_id),
                None,
            )
            obj_to_keep = detail["kept_object"] if detail else None

            kept_frames = set(object_frame_map.get(obj_to_keep, [])) if obj_to_keep else set()
            deleted_frames = set(object_frame_map.get(obj_id, []))
            exclusive_frames = deleted_frames - kept_frames

            # Transfer exclusive frames to the kept object
            for frame_idx in exclusive_frames:
                frame_str = str(frame_idx)
                frame_objects = frames.get(frame_str, {}).get("objects", {})
                if obj_id in frame_objects:
                    frame_objects[obj_to_keep] = frame_objects.pop(obj_id)
                    self.frames_merged += 1

            # Delete from shared frames (where both objects exist)
            for frame_idx in (deleted_frames - exclusive_frames):
                frame_str = str(frame_idx)
                frame_objects = frames.get(frame_str, {}).get("objects", {})
                if obj_id in frame_objects:
                    del frame_objects[obj_id]
                    self.frames_modified += 1

            # Delete the duplicate's object entry
            if obj_id in objects:
                del objects[obj_id]
                self.objects_deleted += 1

        for detail in self.deletion_details:
            # Count how many frames were exclusive to the deleted object
            kept_frames = set(object_frame_map.get(detail["kept_object"], []))
            deleted_frames = set(object_frame_map.get(detail["deleted_object"], []))
            merged_count = len(deleted_frames - kept_frames)
            shared_count = len(deleted_frames & kept_frames)
            logger.info(
                f"Removed duplicate {detail['deleted_object']} ({detail['deleted_engine']}) "
                f"-> merged into {detail['kept_object']} ({detail['kept_engine']}): "
                f"{merged_count} frames transferred, {shared_count} shared frames dropped "
                f"(range {detail['frame_start']}-{detail['frame_end']})"
            )

        return openlabel_data

    def _are_duplicates(
        self,
        obj_a: str,
        obj_b: str,
        object_frame_map: Dict[str, List[int]],
        frames: Dict[str, Any],
    ) -> bool:
        """Check if two objects are duplicates based on IOU thresholds.

        Args:
            obj_a: First object ID
            obj_b: Second object ID
            object_frame_map: Mapping of object IDs to frame lists
            frames: Frame data

        Returns:
            True if objects are duplicates based on configured IoU thresholds
        """
        frames_a = set(object_frame_map.get(obj_a, []))
        frames_b = set(object_frame_map.get(obj_b, []))
        shared_frames = frames_a.intersection(frames_b)

        if len(shared_frames) == 0:
            return False

        # Require shared frames to be a significant portion of the shorter object
        shorter_length = min(len(frames_a), len(frames_b))
        shared_ratio = len(shared_frames) / shorter_length if shorter_length > 0 else 0
        if shared_ratio < self.min_shared_ratio:
            engine_a = self._get_source_engine(obj_a, object_frame_map, frames)
            engine_b = self._get_source_engine(obj_b, object_frame_map, frames)
            logger.debug(
                f"DuplicateRemoval: {obj_a} ({engine_a}) vs {obj_b} ({engine_b}): "
                f"shared={len(shared_frames)}/{shorter_length} ({shared_ratio:.0%}) "
                f"-> SKIP (below min_shared_ratio {self.min_shared_ratio})"
            )
            return False

        ious = []
        iomins = []

        for frame_idx in shared_frames:
            frame_str = str(frame_idx)
            frame_data = frames[frame_str]
            frame_objects = frame_data.get("objects", {})

            bbox_a = self._extract_bbox(frame_objects[obj_a])
            bbox_b = self._extract_bbox(frame_objects[obj_b])

            if bbox_a is not None and bbox_b is not None:
                iou = self.iou_calculator.calculate_intersection_over_union(
                    bbox_a, bbox_b
                )
                ious.append(iou)
                iomin = self.iou_calculator.calculate_intersection_over_min(
                    bbox_a, bbox_b
                )
                iomins.append(iomin)

        if len(ious) == 0:
            return False

        avg_iou = sum(ious) / len(ious)
        min_iou = min(ious)
        avg_iomin = sum(iomins) / len(iomins)

        by_iou = avg_iou > self.avg_iou_threshold and min_iou > self.min_iou_threshold
        by_iomin = avg_iomin > self.iomin_threshold
        is_duplicate = by_iou or by_iomin

        engine_a = self._get_source_engine(obj_a, object_frame_map, frames)
        engine_b = self._get_source_engine(obj_b, object_frame_map, frames)
        result_str = "DUPLICATE" if is_duplicate else "KEEP"
        reason = ""
        if is_duplicate:
            reason = " (by IoU)" if by_iou else " (by IoMin)"
        logger.debug(
            f"DuplicateRemoval: {obj_a} ({engine_a}) vs {obj_b} ({engine_b}): "
            f"shared={len(shared_frames)}/{shorter_length} ({shared_ratio:.0%}), "
            f"avg_iou={avg_iou:.2f}, min_iou={min_iou:.2f}, "
            f"avg_iomin={avg_iomin:.2f} -> {result_str}{reason}"
        )

        return is_duplicate

    def _extract_bbox(self, object_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract bounding box from object data and convert to corner points.

        Args:
            object_data: Object data containing rbbox

        Returns:
            4 corner points as numpy array, or None if extraction fails
        """
        try:
            rbbox_val = object_data["object_data"]["rbbox"][0]["val"]
            x, y, w, h, r = rbbox_val

            cos_r = np.cos(r)
            sin_r = np.sin(r)

            hw = w / 2
            hh = h / 2

            corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])

            rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            rotated_corners = corners @ rotation_matrix.T

            oriented_bbox = rotated_corners + np.array([x, y])

            return oriented_bbox.astype(np.float32)

        except (KeyError, IndexError, ValueError) as e:
            logger.debug(f"Failed to extract bbox: {e}")
            return None

    def _get_source_engine(
        self,
        obj_id: str,
        object_frame_map: Dict[str, List[int]],
        frames: Dict[str, Any],
    ) -> str:
        """Get the source engine for an object (from first non-gap frame).

        Args:
            obj_id: Object ID
            object_frame_map: Mapping of object IDs to frame lists
            frames: Frame data

        Returns:
            Source engine name (yolo, oflow, aruco, or unknown)
        """
        return _get_object_source_engine(obj_id, object_frame_map, frames)

    def _choose_object_to_delete(
        self,
        obj_a: str,
        obj_b: str,
        object_frame_map: Dict[str, List[int]],
        frames: Dict[str, Any],
    ) -> str:
        """Choose which object to delete from a duplicate pair.

        Priority order:
        1. Source engine (yolo > aruco > oflow)
        2. Frame count (keep longer sequence)
        3. Confidence (keep higher confidence)

        Args:
            obj_a: First object ID
            obj_b: Second object ID
            object_frame_map: Mapping of object IDs to frame lists
            frames: Frame data

        Returns:
            Object ID to delete
        """
        # 1. Check source engine priority
        engine_a = self._get_source_engine(obj_a, object_frame_map, frames)
        engine_b = self._get_source_engine(obj_b, object_frame_map, frames)
        priority_a = self.ENGINE_PRIORITY.get(engine_a, 0)
        priority_b = self.ENGINE_PRIORITY.get(engine_b, 0)

        if priority_a != priority_b:
            # Delete the lower priority one
            return obj_a if priority_a < priority_b else obj_b

        # 2. Check frame count (keep longer sequence)
        frames_a = len(object_frame_map.get(obj_a, []))
        frames_b = len(object_frame_map.get(obj_b, []))

        if frames_a != frames_b:
            return obj_a if frames_a < frames_b else obj_b

        # 3. Check average confidence
        conf_a = self._calculate_average_confidence(obj_a, object_frame_map, frames)
        conf_b = self._calculate_average_confidence(obj_b, object_frame_map, frames)

        return obj_a if conf_a < conf_b else obj_b

    def _calculate_average_confidence(
        self,
        obj_id: str,
        object_frame_map: Dict[str, List[int]],
        frames: Dict[str, Any],
    ) -> float:
        """Calculate average confidence for an object across all its frames.

        Args:
            obj_id: Object ID
            object_frame_map: Mapping of object IDs to frame lists
            frames: Frame data

        Returns:
            Average confidence value
        """
        confidences = []

        for frame_idx in object_frame_map.get(obj_id, []):
            frame_str = str(frame_idx)
            frame_data = frames[frame_str]
            frame_objects = frame_data.get("objects", {})

            if obj_id in frame_objects:
                try:
                    vec_list = frame_objects[obj_id]["object_data"]["vec"]
                    for vec_item in vec_list:
                        if vec_item.get("name") == "confidence":
                            conf_values = vec_item.get("val", [])
                            if conf_values:
                                confidences.append(conf_values[-1])
                            break
                except (KeyError, IndexError):
                    pass

        return sum(confidences) / len(confidences) if confidences else 0.0

    def get_statistics(self) -> Dict[str, Any]:
        """Get duplicate removal statistics.

        Returns:
            Dictionary with duplicate removal statistics
        """
        return {
            "objects_deleted": self.objects_deleted,
            "duplicate_pairs_found": self.duplicate_pairs_found,
            "frames_modified": self.frames_modified,
            "frames_merged": self.frames_merged,
        }
