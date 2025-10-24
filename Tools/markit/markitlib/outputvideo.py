"""
outputvideo - Output video rendering from postprocessed OpenLabel data

Handles rendering of annotated video from final postprocessed OpenLabel data,
with support for detecting and highlighting postprocessing modifications.
"""

import logging
from typing import Dict, List

import cv2
import numpy as np

from .config import Constants, DetectionResult, MarkitConfig
from .processing import FrameAnnotator

logger = logging.getLogger(__name__)


def _xywhr_to_bbox_points(cx: float, cy: float, w: float, h: float, r: float) -> np.ndarray:
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


def _openlabel_to_detections(frame_data: Dict, objects_data: Dict, class_map: Dict[int, str]) -> List[DetectionResult]:
    """Convert OpenLabel frame data back to DetectionResult objects for annotation.

    Args:
        frame_data: Frame data from OpenLabel
        objects_data: Objects metadata from OpenLabel
        class_map: Class ID to name mapping

    Returns:
        List of DetectionResult objects
    """
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

            # Check for postprocessing markers and extract metadata
            has_gap = False
            has_rot = False

            for vec in vec_data:
                if vec.get("name") == "confidence":
                    conf_vals = vec.get("val", [1.0])
                    confidence = conf_vals[-1] if conf_vals else 1.0
                elif vec.get("name") == "annotator":
                    annotators = vec.get("val", [""])
                    # Check for specific postprocessing types
                    for ann in annotators:
                        if "gap" in ann:
                            has_gap = True
                        elif "rot" in ann:
                            has_rot = True

                    # Determine source engine (gap takes priority over rot)
                    if has_gap:
                        source_engine = "postprocessed_gap"
                    elif has_rot:
                        source_engine = "postprocessed_rot"
                    else:
                        # Use the last/most recent annotator for engine detection
                        annotator = annotators[-1] if annotators else ""
                        if "yolo" in annotator:
                            source_engine = "yolo"
                        elif "oflow" in annotator:
                            source_engine = "optical_flow"
                        elif "aruco" in annotator:
                            source_engine = "aruco"

            # Get class from objects data
            obj_meta = objects_data.get(obj_id, {})
            class_name = obj_meta.get("type", "unknown")
            class_id = _class_name_to_id(class_name, class_map)

            # Reconstruct oriented bbox from xywhr
            oriented_bbox = _xywhr_to_bbox_points(center_x, center_y, width, height, rotation)

            # Create DetectionResult
            detection = DetectionResult(
                class_id=class_id,
                confidence=confidence,
                oriented_bbox=oriented_bbox,
                center=np.array([center_x, center_y]),
                angle=rotation,
                source_engine=source_engine,
                object_id=int(obj_id)
            )
            detection_results.append(detection)

        except Exception as e:
            logger.error(f"Error converting OpenLabel object {obj_id}: {e}")

    return detection_results


def render_output_video(config: MarkitConfig, openlabel_data: Dict) -> None:
    """Render annotated video from final postprocessed OpenLabel data.

    Args:
        config: Application configuration
        openlabel_data: Final OpenLabel data with postprocessing applied
    """
    if not config.output_video_path:
        return

    logger.info("Rendering output video from postprocessed data...")

    # Open input video
    cap = cv2.VideoCapture(config.video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video for rendering: {config.video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*Constants.MP4V_FOURCC)
    out = cv2.VideoWriter(config.output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_idx = 0
    frames_data = openlabel_data.get("openlabel", {}).get("frames", {})
    objects_data = openlabel_data.get("openlabel", {}).get("objects", {})

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Get detections for this frame from OpenLabel data
            frame_str = str(frame_idx)
            if frame_str in frames_data:
                detection_results = _openlabel_to_detections(
                    frames_data[frame_str],
                    objects_data,
                    config.class_map
                )
                annotated_frame = FrameAnnotator.annotate_frame(frame, detection_results)
                out.write(annotated_frame)
            else:
                out.write(frame)  # No detections, write original frame

            frame_idx += 1
            if frame_idx % 100 == 0:
                logger.info(f"Rendered {frame_idx} frames...")

    finally:
        cap.release()
        out.release()

    logger.info(f"Output video rendered: {config.output_video_path}")
