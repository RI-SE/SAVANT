# savant_app/frontend/utils/playback.py
from PyQt6.QtCore import Qt, QTimer

from savant_app.frontend.types import BBoxData, BBoxDimensionData

from .render import show_frame


def wire(main_window):
    if not hasattr(main_window, "_play_timer"):
        main_window._play_timer = QTimer(main_window)
    _reset_timer_connection(main_window)
    main_window._is_playing = False

    playback_controls = main_window.playback_controls

    if hasattr(playback_controls, "play_clicked"):
        _safe_connect(playback_controls.play_clicked, lambda: _toggle_play(main_window))

    if hasattr(playback_controls, "pause_clicked"):
        _safe_connect(playback_controls.pause_clicked, lambda: _stop(main_window))

    if hasattr(playback_controls, "next_frame_clicked"):
        _safe_connect(
            playback_controls.next_frame_clicked,
            lambda: _step_once(main_window, direction=+1),
        )

    if hasattr(playback_controls, "prev_frame_clicked"):
        _safe_connect(
            playback_controls.prev_frame_clicked,
            lambda: _step_once(main_window, direction=-1),
        )

    if hasattr(playback_controls, "skip_backward_clicked"):
        _safe_connect(
            playback_controls.skip_backward_clicked, lambda n: _skip(main_window, -n)
        )

    if hasattr(playback_controls, "skip_forward_clicked"):
        _safe_connect(
            playback_controls.skip_forward_clicked, lambda n: _skip(main_window, +n)
        )

    if hasattr(main_window.sidebar, "highlight_selected_object"):
        main_window.sidebar.highlight_selected_object.connect(
            lambda object_id: _display_annotation_info(
                main_window, playback_controls, object_id
            )
        )

    if hasattr(main_window.overlay, "bounding_box_selected"):
        main_window.overlay.bounding_box_selected.connect(
            lambda object_id: _display_annotation_info(
                main_window, playback_controls, object_id
            )
        )

    # Connect the new live-update signal from the overlay
    if hasattr(main_window.overlay, "boxModified"):
        _safe_connect(
            main_window.overlay.boxModified,
            lambda bbox_data: _update_live_annotation_info(
                main_window.playback_controls, bbox_data
            ),
        )


def _safe_connect(signal, slot):
    try:
        signal.connect(slot, Qt.ConnectionType.UniqueConnection)
    except TypeError:
        pass


def _reset_timer_connection(main_window):
    try:
        main_window._play_timer.timeout.disconnect()
    except TypeError:
        pass
    main_window._play_timer.timeout.connect(
        lambda: _tick(main_window), Qt.ConnectionType.UniqueConnection
    )


def _toggle_play(main_window):
    if main_window._is_playing:
        _stop(main_window)
    else:
        _start(main_window)


def _start(main_window):
    _reset_timer_connection(main_window)

    total = main_window.project_state_controller.get_frame_count()
    idx = main_window.video_controller.current_index()
    if idx < 0 or (total and idx >= total - 1):
        pixmap, j = main_window.video_controller.jump_to_frame(0)
        show_frame(main_window, pixmap, j)

    fps = main_window.project_state_controller.get_fps()
    if not fps or fps <= 0:
        fps = 25

    main_window._play_timer.start(max(1, int(1000 / int(fps))))
    main_window._is_playing = True
    if hasattr(main_window.playback_controls, "set_playing"):
        main_window.playback_controls.set_playing(True)


def _stop(main_window):
    if main_window._play_timer.isActive():
        main_window._play_timer.stop()
    main_window._is_playing = False
    if hasattr(main_window.playback_controls, "set_playing"):
        main_window.playback_controls.set_playing(False)


def _tick(main_window):
    try:
        pixmap, idx = main_window.video_controller.next_frame()
        if pixmap is None or idx is None:
            _stop(main_window)
            return
        show_frame(main_window, pixmap, idx)
    except Exception:
        _stop(main_window)
        raise  # re-raise for global handler


def _step_once(main_window, direction: int):
    try:
        if direction >= 0:
            pixmap, idx = main_window.video_controller.next_frame()
        else:
            pixmap, idx = main_window.video_controller.previous_frame()

        show_frame(main_window, pixmap, idx)
    except Exception:
        _stop(main_window)
        raise


def _skip(main_window, n: int):
    try:
        pixmap, idx = main_window.video_controller.skip_frames(n)
        show_frame(main_window, pixmap, idx)
    except Exception:
        _stop(main_window)
        raise


def _update_live_annotation_info(playback_controls, bbox_data: BBoxData):
    """
    Updates the playback controls info label with live data from overlay drag.
    """
    if bbox_data is None:
        playback_controls.clear_annotation_info()
        return

    # Adapt BBoxData (from overlay) to BBoxDimensionData (for playback_controls)
    dimension_data = BBoxDimensionData(
        x_center=bbox_data.center_x,
        y_center=bbox_data.center_y,
        width=bbox_data.width,
        height=bbox_data.height,
        rotation=bbox_data.theta,
    )
    playback_controls.display_annotation_info(dimension_data)


def _display_annotation_info(main_window, playback_controls, object_id: str):
    if object_id is None:
        playback_controls.clear_annotation_info()
        return
    frame_key = int(main_window.video_controller.current_index())
    bbox_dimensions = main_window.annotation_controller.get_bbox(frame_key, object_id)
    playback_controls.display_annotation_info(bbox_dimensions)
