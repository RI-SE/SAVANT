# edit/frontend/utils/frame_sync.py
from __future__ import annotations

from edit.frontend.types import BBoxData, ConfidenceFlagMap, Relationship


def _update_overlay_from_model(main_window):
    """Fetch boxes for current frame and update overlay + sidebar."""
    current_frame_index = main_window.video_controller.current_index()
    try:
        # Retrieve FrameBBox objects from backend
        frame_bounding_boxes = (
            main_window.project_state_controller.boxes_with_ids_for_frame(
                current_frame_index
            )
        )

        frame_bounding_boxes_frontend_data = [
            BBoxData(
                object_id=fbbox.object_id,
                object_type=fbbox.object_type,
                center_x=fbbox.bbox.cx,
                center_y=fbbox.bbox.cy,
                width=fbbox.bbox.width,
                height=fbbox.bbox.height,
                theta=fbbox.bbox.theta,
            )
            for fbbox in frame_bounding_boxes
        ]

        # Update overlay dimensions and set bounding boxes
        video_width, video_height = (
            main_window.project_state_controller.get_video_size()
        )
        main_window.overlay.set_frame_size(video_width, video_height)
        main_window.overlay.set_rotated_boxes(frame_bounding_boxes_frontend_data)
        # Retreive relationships from backend.
        frame_relationships = main_window.annotation_controller.get_frame_relationships(
            current_frame_index
        )
        main_window.overlay.set_relationships(
            list(
                map(
                    lambda relationship: Relationship(**relationship),
                    frame_relationships,
                )
            )
        )

        frame_issues_map = main_window.state.confidence_issues()
        frame_issues = frame_issues_map.get(current_frame_index, [])
        flags: ConfidenceFlagMap = {}
        for issue in frame_issues:
            object_id = getattr(issue, "object_id", None)
            severity = getattr(issue, "severity", None)
            if not object_id or severity not in ("warning", "error"):
                continue
            if severity == "error":
                flags[object_id] = "error"
            elif severity == "warning" and object_id not in flags:
                flags[object_id] = "warning"
        main_window.overlay.set_confidence_flags(flags)

        # Refresh sidebar with active objects
        active_objects = main_window.annotation_controller.get_active_objects(
            current_frame_index
        )
        main_window.sidebar.refresh_active_objects(active_objects, flags)
        main_window.sidebar._refresh_active_frame_tags(current_frame_index)
        main_window.sidebar.refresh_confidence_issue_list(current_frame_index)

    except Exception:
        main_window.overlay.set_rotated_boxes([])
        raise
