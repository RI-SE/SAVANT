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
    # If the tracked center moves less than this distance (px) over a full
    # position history window, the tracker has latched onto static background.
    # Window of 10 frames + 5px threshold tolerates slow-moving objects
    # (e.g. a car braking at a junction) while still catching a truly frozen
    # tracker that has latched onto snow or road texture (0px displacement).
    STATIONARY_THRESHOLD = 5.0  # pixels
    STATIONARY_WINDOW = 10      # frames

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
            ("MIL", "TrackerMIL_create"),
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
                # OpenCV 4.11+ returns None (void); older versions return bool
                if success is None or success:
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
        stop_frame: Optional[int] = None,
        skip_object_ids: Optional[set] = None,
    ) -> List[TrackedFrame]:
        """Track object forward from start_frame.

        Args:
            start_frame: Frame index to start tracking from
            bbox_data: BBoxData with initial bbox parameters
            object_id: ID of object being tracked
            iou_threshold: Stop tracking if IoU with existing bbox exceeds this
            progress_callback: Optional callback(current_frame, total_tracked) for progress updates
            stop_frame: Optional frame index to stop tracking at (inclusive)
            skip_object_ids: Optional set of object IDs to ignore during overlap check

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
            stop_frame=stop_frame,
            skip_object_ids=skip_object_ids,
        )

    def track_backward(
        self,
        start_frame: int,
        bbox_data,
        object_id: str,
        iou_threshold: float = 0.3,
        progress_callback: Optional[callable] = None,
        stop_frame: Optional[int] = None,
        skip_object_ids: Optional[set] = None,
    ) -> List[TrackedFrame]:
        """Track object backward from start_frame.

        Args:
            start_frame: Frame index to start tracking from
            bbox_data: BBoxData with initial bbox parameters
            object_id: ID of object being tracked
            iou_threshold: Stop tracking if IoU with existing bbox exceeds this
            progress_callback: Optional callback(current_frame, total_tracked) for progress updates
            stop_frame: Optional frame index to stop tracking at (inclusive)
            skip_object_ids: Optional set of object IDs to ignore during overlap check

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
            stop_frame=stop_frame,
            skip_object_ids=skip_object_ids,
        )

    def _track(
        self,
        start_frame: int,
        bbox_data,
        object_id: str,
        iou_threshold: float,
        direction: int,
        progress_callback: Optional[callable] = None,
        stop_frame: Optional[int] = None,
        skip_object_ids: Optional[set] = None,
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
        init_area = iw * ih  # used later to detect tracker clamping at frame edge
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
        prev_frame = frame
        frame_count = self.video_reader.project_state.video_metadata.frame_count

        # Rolling position history for velocity-based exit prediction.
        # When the object is about to exit the frame its predicted next
        # position will be outside — we stop before the tracker can latch
        # onto background features inside the frame.
        from collections import deque
        _pos_history: deque = deque(maxlen=self.STATIONARY_WINDOW)
        _pos_history.append((center_x, center_y))

        current_frame = start_frame + direction
        while 0 <= current_frame < frame_count:
            # Stop at the boundary frame (inclusive) when re-tracking a range
            if stop_frame is not None:
                if direction == 1 and current_frame > stop_frame:
                    break
                if direction == -1 and current_frame < stop_frame:
                    break
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

            # Detect tracker clamping at the frame edge.
            # OpenCV trackers clip their output rect to frame boundaries, so when
            # an object exits the frame the reported rect stays fully inside and
            # the overlap check below would see 100% overlap and never fire.
            # Instead, check whether the tracked area has shrunk significantly
            # compared to the initial bbox: clamping at any edge causes the
            # reported width or height (or both) to drop, halving the area.
            if init_area > 0 and tw * th < 0.5 * init_area:
                logger.info(
                    f"Tracking stopped at frame {current_frame}: "
                    f"tracked area ({tw * th:.0f}) is less than half the initial "
                    f"area ({init_area}); object likely exited the frame."
                )
                break

            # Stop if the tracked box has mostly left the frame.
            # Measure the fraction of the axis-aligned tracked rect that still
            # overlaps the frame — this handles rotated/large objects correctly.
            overlap_w = min(tx + tw, frame_w) - max(tx, 0)
            overlap_h = min(ty + th, frame_h) - max(ty, 0)
            overlap_area = max(0.0, overlap_w) * max(0.0, overlap_h)
            tracked_area = tw * th
            if tracked_area > 0 and overlap_area / tracked_area < 0.5:
                logger.info(
                    f"Tracking stopped at frame {current_frame}: "
                    f"bbox mostly outside frame ({overlap_area/tracked_area:.1%} overlap)"
                )
                break

            # Estimate rotation via sparse optical flow over the bbox crop,
            # falling back to movement-direction heuristic if flow fails.
            new_theta = self._estimate_rotation_from_flow(
                prev_frame, frame, prev_cx, prev_cy, width, height, prev_theta,
                direction=direction,
            )
            if new_theta is None:
                dx = (new_cx - prev_cx) * direction
                dy = (new_cy - prev_cy) * direction
                if math.hypot(dx, dy) >= self.MIN_MOVEMENT_FOR_ROTATION:
                    movement_angle = math.atan2(dy, dx)
                    if height > width:
                        new_theta = (movement_angle + math.pi / 2) % (2 * math.pi)
                    else:
                        new_theta = movement_angle % (2 * math.pi)
                else:
                    new_theta = prev_theta

            # Check overlap with existing bboxes in this frame
            tracked_corners = bbox_to_corners(new_cx, new_cy, width, height, new_theta)
            overlaps = self._check_overlap(
                current_frame, tracked_corners, iou_threshold, skip_object_ids=skip_object_ids
            )

            if overlaps:
                logger.debug(
                    f"Tracking stopped at frame {current_frame} due to overlap"
                )
                break

            # Velocity-based exit prediction: if we have enough history,
            # compute rolling average velocity and project one step ahead.
            # Stop (after recording the current frame) when the predicted
            # center exits the frame — this fires before the tracker can
            # latch onto in-frame background after the object has left.
            _pos_history.append((new_cx, new_cy))
            _will_exit = False
            if len(_pos_history) >= 2:
                pts = list(_pos_history)
                avg_dx = sum(pts[i][0] - pts[i-1][0] for i in range(1, len(pts))) / (len(pts) - 1)
                avg_dy = sum(pts[i][1] - pts[i-1][1] for i in range(1, len(pts))) / (len(pts) - 1)
                pred_cx = new_cx + avg_dx
                pred_cy = new_cy + avg_dy
                if pred_cx < 0 or pred_cx >= frame_w or pred_cy < 0 or pred_cy >= frame_h:
                    logger.info(
                        f"Tracking stopped at frame {current_frame}: "
                        f"predicted exit at ({pred_cx:.1f}, {pred_cy:.1f}) "
                        f"velocity=({avg_dx:.1f}, {avg_dy:.1f})"
                    )
                    _will_exit = True

            # Stationarity check: when the history is full, measure total
            # displacement from oldest to newest point.  If the tracker
            # hasn't moved at all it has latched onto static background
            # (e.g. snow, road texture) rather than the object.
            if len(_pos_history) == _pos_history.maxlen:
                oldest_x, oldest_y = _pos_history[0]
                displacement = math.hypot(new_cx - oldest_x, new_cy - oldest_y)
                if displacement < self.STATIONARY_THRESHOLD:
                    logger.info(
                        f"Tracking stopped at frame {current_frame}: "
                        f"tracker appears stationary (displacement={displacement:.2f}px "
                        f"over {_pos_history.maxlen} frames); likely latched onto background."
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

            # Stop after recording the current edge frame if exit is predicted
            if _will_exit:
                break

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
            prev_frame = frame
            current_frame += direction

        logger.info(
            f"Tracking completed: {len(results)} frames tracked "
            f"({'forward' if direction > 0 else 'backward'})"
        )
        return results

    def _estimate_rotation_from_flow(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        cx: float,
        cy: float,
        width: float,
        height: float,
        prev_theta: float,
        padding: float = 0.2,
        direction: int = 1,
    ) -> Optional[float]:
        """Estimate rotation by running sparse optical flow on the bbox crop.

        The crop region is scaled relative to the bbox size, so it works for
        both small and large objects.

        When tracking backward (direction=-1) the affine delta is negated so
        the rotation accumulates in the correct direction.

        Returns the new absolute theta (radians), or None if estimation failed.
        """
        fh, fw = prev_frame.shape[:2]

        # Crop with relative padding around the bbox
        half_w = width / 2 * (1 + padding)
        half_h = height / 2 * (1 + padding)
        x1 = int(max(0, cx - half_w))
        y1 = int(max(0, cy - half_h))
        x2 = int(min(fw, cx + half_w))
        y2 = int(min(fh, cy + half_h))

        if x2 - x1 < 4 or y2 - y1 < 4:
            return None

        # Convert crops to grayscale
        prev_crop = prev_frame[y1:y2, x1:x2]
        curr_crop = curr_frame[y1:y2, x1:x2]
        if prev_crop.ndim == 3:
            prev_gray = cv2.cvtColor(prev_crop, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_crop, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_crop
            curr_gray = curr_crop

        # Detect good features in the previous crop
        max_corners = 40
        quality = 0.01
        min_dist = max(3.0, min(width, height) * 0.05)  # relative to bbox size
        pts = cv2.goodFeaturesToTrack(
            prev_gray, maxCorners=max_corners, qualityLevel=quality,
            minDistance=min_dist,
        )
        if pts is None or len(pts) < 4:
            return None

        # Track them with Lucas-Kanade
        pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None)
        if pts_curr is None:
            return None

        good_prev = pts[status.ravel() == 1]
        good_curr = pts_curr[status.ravel() == 1]
        if len(good_prev) < 2:
            return None

        # Fit a partial affine (translation + rotation + uniform scale)
        M, inliers = cv2.estimateAffinePartial2D(good_prev, good_curr, method=cv2.RANSAC)
        if M is None or inliers is None or inliers.sum() < 2:
            return None

        # Extract rotation from the 2×2 part of the affine matrix.
        # Negate when tracking backward so the delta applies in the correct direction.
        delta_theta = math.atan2(M[1, 0], M[0, 0]) * direction
        # Clamp to a realistic per-frame rotation limit (~10°).
        # Optical flow on low-contrast backgrounds (e.g. dark car on asphalt)
        # often picks up road texture rather than the vehicle, producing large
        # spurious deltas that accumulate over many frames.
        max_delta = math.radians(10.0)
        delta_theta = max(-max_delta, min(max_delta, delta_theta))
        new_theta = (prev_theta + delta_theta) % (2 * math.pi)
        return new_theta

    def _check_overlap(
        self,
        frame_idx: int,
        tracked_corners: np.ndarray,
        iou_threshold: float,
        skip_object_ids: Optional[set] = None,
    ) -> bool:
        """Check if tracked bbox overlaps with existing annotations.

        Args:
            frame_idx: Frame to check
            tracked_corners: Corner points of tracked bbox
            iou_threshold: IoU threshold for overlap detection
            skip_object_ids: Optional set of object IDs to ignore (e.g. the object being re-tracked)

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

            # Skip specified objects (e.g. the object being re-tracked)
            if skip_object_ids and obj_id in skip_object_ids:
                continue

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
