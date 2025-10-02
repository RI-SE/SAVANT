# savant_app/frontend/utils/playback.py
from PyQt6.QtCore import QTimer, Qt
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
        _safe_connect(playback_controls.next_frame_clicked,
                      lambda: _step_once(main_window, direction=+1))

    if hasattr(playback_controls, "prev_frame_clicked"):
        _safe_connect(playback_controls.prev_frame_clicked,
                      lambda: _step_once(main_window, direction=-1))

    if hasattr(playback_controls, "skip_backward_clicked"):
        _safe_connect(playback_controls.skip_backward_clicked, lambda n: _skip(main_window, -n))

    if hasattr(playback_controls, "skip_forward_clicked"):
        _safe_connect(playback_controls.skip_forward_clicked, lambda n: _skip(main_window, +n))


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
        lambda: _tick(main_window), Qt.ConnectionType.UniqueConnection)


def _toggle_play(main_window):
    if main_window._is_playing:
        _stop(main_window)
    else:
        _start(main_window)


def _start(main_window):
    _reset_timer_connection(main_window)

    try:
        total = main_window.video_controller.total_frames()
        idx = main_window.video_controller.current_index()
        if idx < 0 or (total and idx >= total - 1):
            pixmap, j = main_window.video_controller.jump_to_frame(0)
            show_frame(main_window, pixmap, j)
    except Exception:
        pass

    fps = 0
    try:
        fps = main_window.video_controller.fps()
    except Exception:
        pass
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
            raise StopIteration
        show_frame(main_window, pixmap, idx)
    except StopIteration:
        _stop(main_window)
    except Exception:
        _stop(main_window)


def _step_once(main_window, direction: int):
    try:
        if direction >= 0:
            pixmap, idx = main_window.video_controller.next_frame()
        else:
            pixmap, idx = main_window.video_controller.previous_frame()

        show_frame(main_window, pixmap, idx)
    except Exception:
        _stop(main_window)


def _skip(main_window, n: int):
    try:
        pixmap, idx = main_window.video_controller.skip_frames(n)
        show_frame(main_window, pixmap, idx)
    except Exception:
        _stop(main_window)
