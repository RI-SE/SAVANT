"""
conflict_resolution - IoU-based detection conflict resolution

Handles conflicts between different detection engines using Intersection over Union.
"""

import logging
from typing import Any, Dict, List

from ..config import ConflictResolutionConfig, DetectionResult
from ..geometry import BBoxOverlapCalculator

logger = logging.getLogger(__name__)


class DetectionConflictResolver:
    """Handles conflicts between different detection engines using IoU."""

    def __init__(self, config: ConflictResolutionConfig):
        self.config = config
        self.overlap_calc = BBoxOverlapCalculator()
        self.conflicts_resolved = 0
        self.total_conflicts = 0
        self.decision_log: List[Dict[str, Any]] = []

        # Validate IoU threshold
        if not (0.0 <= self.config.iou_threshold <= 1.0):
            raise ValueError(
                f"IoU threshold must be between 0.0 and 1.0, got: {self.config.iou_threshold}"
            )

    def resolve_conflicts(
        self, detection_results: List[DetectionResult], frame_idx: int = None
    ) -> List[DetectionResult]:
        """Resolve conflicts between detection results using IoU with YOLO precedence.

        Args:
            detection_results: List of detection results from all engines
            frame_idx: Optional frame index for logging

        Returns:
            Filtered list with conflicts resolved
        """
        if len(detection_results) <= 1:
            return detection_results

        # Separate results by engine
        yolo_results = [r for r in detection_results if r.source_engine == "yolo"]
        optical_flow_results = [
            r for r in detection_results if r.source_engine == "optical_flow"
        ]
        other_results = [
            r
            for r in detection_results
            if r.source_engine not in ["yolo", "optical_flow"]
        ]

        # Keep all YOLO results (highest precedence)
        final_results = yolo_results.copy()

        # Filter optical flow results that conflict with YOLO
        filtered_optical_flow = self._filter_conflicting_detections_iou(
            primary_detections=yolo_results,
            secondary_detections=optical_flow_results,
            frame_idx=frame_idx,
        )

        final_results.extend(filtered_optical_flow)
        final_results.extend(other_results)

        return final_results

    def _filter_conflicting_detections_iou(
        self,
        primary_detections: List[DetectionResult],
        secondary_detections: List[DetectionResult],
        frame_idx: int = None,
    ) -> List[DetectionResult]:
        """Filter secondary detections that conflict with primary detections using IoU.

        Args:
            primary_detections: High-priority detections (e.g., YOLO)
            secondary_detections: Lower-priority detections (e.g., optical flow)
            frame_idx: Optional frame index for logging

        Returns:
            Filtered secondary detections with IoU conflicts removed
        """
        if not primary_detections:
            return secondary_detections

        filtered_results = []

        for secondary_det in secondary_detections:
            has_conflict = False
            max_iou = 0.0
            conflicting_primary = None

            for primary_det in primary_detections:
                iou = self.overlap_calc.calculate_intersection_over_union(
                    secondary_det.oriented_bbox, primary_det.oriented_bbox
                )

                if iou >= self.config.iou_threshold:
                    has_conflict = True
                    if iou > max_iou:
                        max_iou = iou
                        conflicting_primary = primary_det
                    self.total_conflicts += 1

            if not has_conflict:
                filtered_results.append(secondary_det)
            else:
                self.conflicts_resolved += 1
                secondary_id = secondary_det.object_id or "?"
                primary_id = conflicting_primary.object_id or "?"
                self.decision_log.append(
                    {
                        "stage": "detection",
                        "source": "conflict_resolution",
                        "action": "drop_detection",
                        "object_id": secondary_id,
                        "frame": frame_idx,
                        "reason": f"IoU={max_iou:.2f} with yolo object {primary_id} "
                        f">= threshold {self.config.iou_threshold}",
                        "details": {
                            "losing_engine": "optical_flow",
                            "winning_engine": "yolo",
                            "winning_object_id": primary_id,
                            "iou": max_iou,
                            "iou_threshold": self.config.iou_threshold,
                        },
                    }
                )
                if self.config.enable_logging:
                    frame_str = f"frame {frame_idx}: " if frame_idx is not None else ""
                    logger.info(
                        f"{frame_str}conflict resolved: "
                        f"optical_flow obj {secondary_id} dropped (IoU={max_iou:.2f} with yolo obj {primary_id})"
                    )

        return filtered_results

    def get_decision_log(self) -> List[Dict[str, Any]]:
        """Return structured records of every conflict-resolved detection drop.

        Returns:
            List of decision record dicts, collected regardless of whether
            --verbose-conflicts live logging is enabled.
        """
        return self.decision_log

    def get_conflict_statistics(self) -> dict:
        """Get statistics about resolved conflicts.

        Returns:
            Dictionary with conflict resolution statistics
        """
        return {
            "total_conflicts": self.total_conflicts,
            "conflicts_resolved": self.conflicts_resolved,
            "resolution_rate": (self.conflicts_resolved / max(1, self.total_conflicts))
            * 100,
            "iou_threshold": self.config.iou_threshold,
        }
