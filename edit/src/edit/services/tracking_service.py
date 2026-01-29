"""Object tracking service using OpenCV trackers.

Provides forward and backward tracking of objects across video frames,
with rotation estimation from movement direction.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from savant_common.geometry import (
    BBoxOverlapCalculator,
    bbox_to_corners,
    rotated_to_axis_aligned,
)

logger = logging.getLogger(__name__)


@dataclass
class TrackedFrame:
    """Result of tracking for a single frame."""

    frame_idx: int
    center_x: float
    center_y: float
    width: float
    height: float
    theta: float


class TrackingService:
    """Tracks objects across video frames using OpenCV trackers.

    Supports forward and backward tracking with automatic rotation estimation
    based on movement direction.
    """

    MIN_MOVEMENT_FOR_ROTATION = 5.0  # pixels

    def __init__(self, video_reader, annotation_controller):
        """Initialize tracking service.

        Args:
            video_reader: VideoReader instance for frame access
            annotation_controller: AnnotationController for bbox queries
        """
        self.video_reader = video_reader
        self.annotation_controller = annotation_controller

    def _get_available_trackers(self) -> List[Tuple[str, Any]]:
        """Get list of available tracker factories.

        Returns:
            List of (name, factory_function) tuples
        """
        available = []
        # Try simpler trackers first - they're more reliable
        tracker_types = [
            ("MOSSE", "TrackerMOSSE_create"),
            ("KCF", "TrackerKCF_create"),
            ("CSRT", "TrackerCSRT_create"),
        ]

        for kind, create_func in tracker_types:
            try:
                # Try modern API first
                if hasattr(cv2, create_func):
                    available.append((kind, getattr(cv2, create_func)))
                # Try legacy module
                elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, create_func):
                    available.append((kind, getattr(cv2.legacy, create_func)))
            except Exception:
                continue

        return available

    def create_tracker(self) -> Tuple[Any, str]:
        """Create OpenCV tracker with fallback support.

        Returns:
            Tuple of (tracker instance, tracker name)

        Raises:
            RuntimeError: If no tracker is available
        """
        available = self._get_available_trackers()
        if not available:
            raise RuntimeError(
                "No OpenCV tracker available. Install opencv-contrib-python."
            )

        kind, factory = available[0]
        tracker = factory()
        logger.debug(f"Created {kind} tracker")
        return tracker, kind

    def _try_init_tracker(self, frame: np.ndarray, rect: Tuple[int, int, int, int]) -> Tuple[Optional[Any], str]:
        """Try to initialize a tracker, trying different tracker types if needed.

        Returns:
            Tuple of (tracker, name) or (None, "") if all failed
        """
        # Ensure frame has 3 channels
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Make a copy to ensure it's not a view
        frame = frame.copy()

        available = self._get_available_trackers()
        logger.info(f"Available trackers: {[name for name, _ in available]}")

        for kind, factory in available:
            try:
                tracker = factory()
                logger.debug(f"Trying {kind} with rect={rect}")
                success = tracker.init(frame, rect)
                if success:
                    logger.info(f"{kind} tracker initialized successfully")
                    return tracker, kind
                else:
                    logger.warning(f"{kind} tracker.init() returned False")
            except Exception as e:
                logger.warning(f"{kind} tracker failed with exception: {e}")
                continue

        return None, ""

    def track_forward(
        self,
        start_frame: int,
        bbox_data,
        object_id: str,
        iou_threshold: float = 0.3,
        progress_callback: Optional[callable] = None,
    ) -> List[TrackedFrame]:
        """Track object forward from start_frame.

        Args:
            start_frame: Frame index to start tracking from
            bbox_data: BBoxData with initial bbox parameters
            object_id: ID of object being tracked
            iou_threshold: Stop tracking if IoU with existing bbox exceeds this
            progress_callback: Optional callback(current_frame, total_tracked) for progress updates

        Returns:
            List of TrackedFrame results for successfully tracked frames
        """
        return self._track(
            start_frame=start_frame,
            bbox_data=bbox_data,
            object_id=object_id,
            iou_threshold=iou_threshold,
            direction=1,
            progress_callback=progress_callback,
        )

    def track_backward(
        self,
        start_frame: int,
        bbox_data,
        object_id: str,
        iou_threshold: float = 0.3,
        progress_callback: Optional[callable] = None,
    ) -> List[TrackedFrame]:
        """Track object backward from start_frame.

        Args:
            start_frame: Frame index to start tracking from
            bbox_data: BBoxData with initial bbox parameters
            object_id: ID of object being tracked
            iou_threshold: Stop tracking if IoU with existing bbox exceeds this
            progress_callback: Optional callback(current_frame, total_tracked) for progress updates

        Returns:
            List of TrackedFrame results for successfully tracked frames
        """
        return self._track(
            start_frame=start_frame,
            bbox_data=bbox_data,
            object_id=object_id,
            iou_threshold=iou_threshold,
            direction=-1,
            progress_callback=progress_callback,
        )

    def _track(
        self,
        start_frame: int,
        bbox_data,
        object_id: str,
        iou_threshold: float,
        direction: int,
        progress_callback: Optional[callable] = None,
    ) -> List[TrackedFrame]:
        """Core tracking loop.

        Args:
            start_frame: Starting frame index
            bbox_data: Initial bbox (BBoxData-like object with center_x, center_y, etc.)
            object_id: Object ID to track
            iou_threshold: IoU threshold for stopping
            direction: 1 for forward, -1 for backward

        Returns:
            List of TrackedFrame for each successfully tracked frame
        """
        results: List[TrackedFrame] = []

        # Extract bbox parameters
        center_x = bbox_data.center_x
        center_y = bbox_data.center_y
        width = bbox_data.width
        height = bbox_data.height
        theta = bbox_data.theta

        # Get starting frame first to know frame dimensions
        try:
            frame = self.video_reader.get_frame(start_frame)
        except Exception as e:
            logger.error(f"Failed to get start frame {start_frame}: {e}")
            return results

        if frame is None or frame.size == 0:
            logger.error(f"Got empty frame at {start_frame}")
            return results

        frame_h, frame_w = frame.shape[:2]
        logger.debug(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")

        # Convert rotated bbox to axis-aligned for tracker initialization
        init_x, init_y, init_w, init_h = rotated_to_axis_aligned(
            center_x, center_y, width, height, theta
        )

        # Clamp bbox to frame boundaries
        init_x = max(0, init_x)
        init_y = max(0, init_y)
        # Ensure bbox doesn't extend beyond frame
        if init_x + init_w > frame_w:
            init_w = frame_w - init_x
        if init_y + init_h > frame_h:
            init_h = frame_h - init_y

        # Validate bbox dimensions
        if init_w < 1 or init_h < 1:
            logger.warning(
                f"Invalid bbox dimensions after clamping: w={init_w}, h={init_h}"
            )
            return results

        # Ensure native Python ints for OpenCV
        ix, iy, iw, ih = int(init_x), int(init_y), int(init_w), int(init_h)
        init_rect = (ix, iy, iw, ih)
        logger.info(
            f"Tracker init: frame={start_frame}, rect={init_rect}, "
            f"frame_size=({frame_w}x{frame_h}), "
            f"original bbox: cx={center_x:.1f}, cy={center_y:.1f}, "
            f"w={width:.1f}, h={height:.1f}, theta={theta:.3f}"
        )

        # Ensure frame is contiguous and in correct format
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)

        # Try to initialize a tracker (tries multiple tracker types)
        tracker, tracker_name = self._try_init_tracker(frame, init_rect)

        if tracker is None:
            logger.warning(
                f"All trackers failed to initialize with rect={init_rect}, "
                f"frame_size=({frame_w}x{frame_h}), frame_dtype={frame.dtype}"
            )
            return results

        # Track state
        prev_cx, prev_cy = center_x, center_y
        prev_theta = theta
        frame_count = self.video_reader.project_state.video_metadata.frame_count

        current_frame = start_frame + direction
        while 0 <= current_frame < frame_count:
            # Get frame
            try:
                frame = self.video_reader.get_frame(current_frame)
            except Exception as e:
                logger.debug(f"Failed to get frame {current_frame}: {e}")
                break

            # Update tracker
            ok, tracked_bbox = tracker.update(frame)
            if not ok:
                logger.debug(f"Tracker lost object at frame {current_frame}")
                break

            # Extract new center from tracked axis-aligned bbox
            tx, ty, tw, th = tracked_bbox
            new_cx = tx + tw / 2
            new_cy = ty + th / 2

            # Check if bbox is still within frame bounds
            if (new_cx - width / 2 < 0 or new_cx + width / 2 > frame_w or
                new_cy - height / 2 < 0 or new_cy + height / 2 > frame_h):
                logger.info(
                    f"Tracking stopped at frame {current_frame}: "
                    f"bbox out of bounds (cx={new_cx:.1f}, cy={new_cy:.1f})"
                )
                break

            # Estimate rotation from movement direction
            dx = new_cx - prev_cx
            dy = new_cy - prev_cy
            distance = math.hypot(dx, dy)

            if distance >= self.MIN_MOVEMENT_FOR_ROTATION:
                movement_angle = math.atan2(dy, dx)
                # Adjust for aspect ratio: if height > width, object is "tall"
                # and its forward direction is perpendicular to movement
                if height > width:
                    new_theta = movement_angle + math.pi / 2
                else:
                    new_theta = movement_angle
                # Normalize to [0, 2pi)
                new_theta = new_theta % (2 * math.pi)
            else:
                # Keep previous rotation for small movements
                new_theta = prev_theta

            # Check overlap with existing bboxes in this frame
            tracked_corners = bbox_to_corners(new_cx, new_cy, width, height, new_theta)
            overlaps = self._check_overlap(
                current_frame, tracked_corners, iou_threshold
            )

            if overlaps:
                logger.debug(
                    f"Tracking stopped at frame {current_frame} due to overlap"
                )
                break

            # Add to results
            results.append(
                TrackedFrame(
                    frame_idx=current_frame,
                    center_x=new_cx,
                    center_y=new_cy,
                    width=width,
                    height=height,
                    theta=new_theta,
                )
            )

            # Call progress callback to keep UI responsive
            # If callback returns True, cancel tracking
            if progress_callback is not None:
                try:
                    if progress_callback(current_frame, len(results)):
                        logger.info("Tracking cancelled by user")
                        break
                except Exception:
                    pass

            prev_cx, prev_cy = new_cx, new_cy
            prev_theta = new_theta
            current_frame += direction

        logger.info(
            f"Tracking completed: {len(results)} frames tracked "
            f"({'forward' if direction > 0 else 'backward'})"
        )
        return results

    def _check_overlap(
        self,
        frame_idx: int,
        tracked_corners: np.ndarray,
        iou_threshold: float,
    ) -> bool:
        """Check if tracked bbox overlaps with existing annotations.

        Args:
            frame_idx: Frame to check
            tracked_corners: Corner points of tracked bbox
            iou_threshold: IoU threshold for overlap detection

        Returns:
            True if overlap detected above threshold (stops tracking to avoid overwriting)
        """
        try:
            active_objects = self.annotation_controller.get_active_objects(frame_idx)
        except Exception as e:
            logger.debug(f"Failed to get active objects for frame {frame_idx}: {e}")
            return False

        if not active_objects:
            return False

        logger.debug(f"Frame {frame_idx}: checking {len(active_objects)} objects")

        for obj in active_objects:
            obj_id = obj.get("id")  # Note: get_active_objects returns "id", not "object_id"

            # Get the existing bbox
            try:
                existing_bbox = self.annotation_controller.get_bbox(frame_idx, obj_id)
                if existing_bbox is None:
                    logger.debug(f"Frame {frame_idx}: no bbox for {obj_id}")
                    continue
            except Exception as e:
                logger.debug(f"Failed to get bbox for {obj_id} at frame {frame_idx}: {e}")
                continue

            # RotatedBBox uses x_center, y_center, rotation (not center_x, center_y, theta)
            existing_corners = bbox_to_corners(
                existing_bbox.x_center,
                existing_bbox.y_center,
                existing_bbox.width,
                existing_bbox.height,
                existing_bbox.rotation,
            )

            iou = BBoxOverlapCalculator.calculate_intersection_over_union(
                tracked_corners, existing_corners
            )

            if iou > 0:
                logger.info(
                    f"Frame {frame_idx}: IoU with {obj_id} = {iou:.3f} "
                    f"(threshold={iou_threshold})"
                )

            if iou > iou_threshold:
                logger.info(
                    f"Tracking stopped at frame {frame_idx}: "
                    f"overlap with object {obj_id} (IoU={iou:.3f})"
                )
                return True

        return False
