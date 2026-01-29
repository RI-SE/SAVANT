"""
outputvideo - Output video rendering from postprocessed OpenLabel data

Handles rendering of annotated video from final postprocessed OpenLabel data,
with support for detecting and highlighting postprocessing modifications.
Includes optional optical flow debug visualization (magnitude heatmap, motion mask).
"""

import logging
from typing import Dict, List, Optional

import cv2
import numpy as np

from .config import Constants, DetectionResult, MarkitConfig, OpticalFlowParams
from .processing import FrameAnnotator, OpticalFlowEngine

logger = logging.getLogger(__name__)


def _xywhr_to_bbox_points(
    cx: float, cy: float, w: float, h: float, r: float
) -> np.ndarray:
    """Convert xywhr to oriented bbox corner points.

    Args:
        cx: Center x coordinate
        cy: Center y coordinate
        w: Width
        h: Height
        r: Rotation angle in radians

    Returns:
        Numpy array of bbox corner points
    """
    # Convert radians to degrees for cv2
    rect = ((float(cx), float(cy)), (float(w), float(h)), float(np.degrees(r)))
    bbox_points = cv2.boxPoints(rect)
    return bbox_points.astype(np.int32)


def _class_name_to_id(class_name: str, class_map: Dict[int, str]) -> int:
    """Map class name back to ID using reverse lookup.

    Args:
        class_name: Class name string
        class_map: Class ID to name mapping

    Returns:
        Class ID (0 if not found)
    """
    for class_id, name in class_map.items():
        if name == class_name:
            return class_id
    return 0  # Default


def _draw_raw_yolo_boxes(
    frame: np.ndarray, frame_idx: int, debug_data: Dict
) -> np.ndarray:
    """Draw original YOLO boxes (before OpenLabel conversion) in red for debugging.

    Args:
        frame: Input frame
        frame_idx: Current frame index
        debug_data: Separate debug data structure with raw YOLO xywhr

    Returns:
        Frame with raw YOLO boxes drawn in red
    """
    annotated_frame = frame.copy()
    red = (0, 0, 255)  # Red color for raw YOLO boxes (distinct from all other colors)

    # Get debug data for this frame
    frame_debug = debug_data.get(frame_idx, {})

    for obj_id, obj_debug_data in frame_debug.items():
        try:
            raw_xywhr = obj_debug_data.get("raw_xywhr")

            if raw_xywhr and len(raw_xywhr) >= 5:
                cx, cy, w, h, r = raw_xywhr[:5]

                # Draw raw YOLO box using original dimensions and angle
                bbox_points = _xywhr_to_bbox_points(cx, cy, w, h, r)
                cv2.drawContours(annotated_frame, [bbox_points], 0, red, 2)

                # Draw center point
                center = (int(cx), int(cy))
                cv2.circle(annotated_frame, center, 3, red, -1)

                # Add label "RAW YOLO"
                label = f"RAW:{obj_id}"
                label_pos = (int(cx) - 30, int(cy) - 10)
                cv2.putText(
                    annotated_frame,
                    label,
                    label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    red,
                    1,
                    cv2.LINE_AA,
                )

        except Exception as e:
            logger.debug(f"Error drawing raw YOLO box for object {obj_id}: {e}")

    return annotated_frame


def _openlabel_to_detections(
    frame_data: Dict, objects_data: Dict, class_map: Dict[int, str]
) -> List[DetectionResult]:
    """Convert OpenLabel frame data back to DetectionResult objects for annotation.

    Args:
        frame_data: Frame data from OpenLabel
        objects_data: Objects metadata from OpenLabel
        class_map: Class ID to name mapping

    Returns:
        List of DetectionResult objects
    """
    import re

    detection_results = []

    for obj_id, obj_frame_data in frame_data.get("objects", {}).items():
        try:
            # Extract rbbox data
            rbbox_list = obj_frame_data.get("object_data", {}).get("rbbox", [])
            if not rbbox_list:
                continue

            xywhr = rbbox_list[0].get("val", [])
            if len(xywhr) < 5:
                continue

            center_x, center_y, width, height, rotation = xywhr[:5]

            # Extract metadata
            vec_data = obj_frame_data.get("object_data", {}).get("vec", [])
            confidence = 1.0
            source_engine = "yolo"
            housekeeping_tags = []

            for vec in vec_data:
                if vec.get("name") == "confidence":
                    conf_vals = vec.get("val", [1.0])
                    # Use last confidence (detector confidence, not housekeeping)
                    confidence = conf_vals[-1] if conf_vals else 1.0
                elif vec.get("name") == "annotator":
                    annotators = vec.get("val", [""])

                    # Extract housekeeping tags from markit_housekeeping(...) entry
                    for ann in annotators:
                        match = re.match(r"markit_housekeeping\(([^)]*)\)", ann)
                        if match and match.group(1):
                            housekeeping_tags = [t.strip() for t in match.group(1).split(",")]

                    # Determine source engine from detector annotator (not housekeeping)
                    # Search from last to first for a detector entry
                    for ann in reversed(annotators):
                        if "markit_housekeeping" in ann:
                            continue  # Skip housekeeping entries
                        if "yolo" in ann:
                            source_engine = "yolo"
                            break
                        elif "optical_flow" in ann or "oflow" in ann:
                            source_engine = "optical_flow"
                            break
                        elif "aruco" in ann:
                            source_engine = "aruco"
                            break
                    else:
                        # No detector found - likely gap-filled frame
                        # Use "gap" as source if gap tag present, else default
                        if "gap" in housekeeping_tags:
                            source_engine = "gap"
                        else:
                            source_engine = "unknown"

            # Get class from objects data
            obj_meta = objects_data.get(obj_id, {})
            class_name = obj_meta.get("type", "unknown")
            class_id = _class_name_to_id(class_name, class_map)

            # Reconstruct oriented bbox from xywhr
            oriented_bbox = _xywhr_to_bbox_points(
                center_x, center_y, width, height, rotation
            )

            # Create DetectionResult
            detection = DetectionResult(
                class_id=class_id,
                confidence=confidence,
                oriented_bbox=oriented_bbox,
                center=np.array([center_x, center_y]),
                angle=rotation,
                source_engine=source_engine,
                object_id=int(obj_id),
            )
            # Attach housekeeping tags for label rendering
            detection.housekeeping_tags = housekeeping_tags
            detection_results.append(detection)

        except Exception as e:
            logger.error(f"Error converting OpenLabel object {obj_id}: {e}")

    return detection_results


def draw_optical_flow_debug(
    frame: np.ndarray,
    debug_data: Dict[str, np.ndarray],
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay optical flow visualization on frame.

    Draws a magnitude heatmap showing motion intensity (blue=no motion, red=high motion).

    Args:
        frame: BGR image
        debug_data: Dict with 'magnitude', 'motion_mask', 'flow' arrays
        alpha: Blend factor for overlay (0=original only, 1=heatmap only)

    Returns:
        Annotated frame with flow visualization
    """
    if debug_data is None:
        return frame

    magnitude = debug_data.get("magnitude")
    if magnitude is None:
        return frame

    # Normalize magnitude to 0-255 for colormap
    mag_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    mag_uint8 = mag_normalized.astype(np.uint8)

    # Apply JET colormap: blue (low) -> green -> red (high)
    heatmap = cv2.applyColorMap(mag_uint8, cv2.COLORMAP_JET)

    # Resize heatmap to match frame size if needed (when processing_scale < 1.0)
    if heatmap.shape[:2] != frame.shape[:2]:
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Blend heatmap with original frame
    result = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)

    return result


def render_output_video(
    config: MarkitConfig, openlabel_data: Dict, debug_data: Dict = None
) -> None:
    """Render annotated video from final postprocessed OpenLabel data.

    Args:
        config: Application configuration
        openlabel_data: Final OpenLabel data with postprocessing applied
        debug_data: Optional separate debug data structure (for verbose mode)
    """
    if not config.output_video_path:
        return

    if debug_data is None:
        debug_data = {}

    logger.info("Rendering output video from postprocessed data...")
    if config.verbose:
        logger.info(
            "Verbose mode: Drawing YOLO boxes (red) and OpenLabel boxes (green/colors)"
        )

    # Check if optical flow debug visualization is enabled
    flow_debug_enabled = config.optical_flow_params.debug_visualization
    flow_engine: Optional[OpticalFlowEngine] = None

    if flow_debug_enabled:
        logger.info("Optical flow debug visualization enabled (magnitude heatmap)")
        # Create a dedicated optical flow engine for visualization
        flow_engine = OpticalFlowEngine(config.optical_flow_params)

    # Open input video
    cap = cv2.VideoCapture(config.video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video for rendering: {config.video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Setup video writer - try H.264 first for better compression, fall back to mp4v
    # H.264 requires libx264; if unavailable, mp4v (MPEG-4 Part 2) is used
    codec_chain = ["avc1", Constants.MP4V_FOURCC]
    out = None
    used_codec = None

    for codec in codec_chain:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(
            config.output_video_path, fourcc, fps, (frame_width, frame_height)
        )
        if out.isOpened():
            used_codec = codec
            break
        out.release()

    if used_codec is None or not out.isOpened():
        logger.error("Failed to open video writer with any available codec")
        cap.release()
        return

    if used_codec == "avc1":
        logger.info("Using H.264 codec for output video")
    elif used_codec != "avc1":
        logger.debug(f"H.264 not available, using '{used_codec}' codec")

    frame_idx = 0
    frames_data = openlabel_data.get("openlabel", {}).get("frames", {})
    objects_data = openlabel_data.get("openlabel", {}).get("objects", {})

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            annotated_frame = frame.copy()

            # Draw optical flow debug visualization (heatmap) if enabled
            if flow_engine is not None:
                # Process frame to get optical flow debug data
                flow_engine.process_frame(frame)
                flow_debug_data = flow_engine.get_debug_visualization()
                if flow_debug_data is not None:
                    annotated_frame = draw_optical_flow_debug(
                        annotated_frame, flow_debug_data, alpha=0.5
                    )

            # Get detections for this frame from OpenLabel data
            frame_str = str(frame_idx)
            if frame_str in frames_data:
                # If verbose, draw original YOLO boxes first (in red)
                if config.verbose:
                    annotated_frame = _draw_raw_yolo_boxes(
                        annotated_frame, frame_idx, debug_data
                    )

                # Then draw OpenLabel boxes on top (via standard annotator)
                detection_results = _openlabel_to_detections(
                    frames_data[frame_str], objects_data, config.class_map
                )
                annotated_frame = FrameAnnotator.annotate_frame(
                    annotated_frame, detection_results, config.class_map
                )

            out.write(annotated_frame)

            frame_idx += 1
            if frame_idx % 100 == 0:
                logger.info(f"Rendered {frame_idx} frames...")

    finally:
        cap.release()
        out.release()
        if flow_engine is not None:
            flow_engine.cleanup()

    logger.info(f"Output video rendered: {config.output_video_path}")
