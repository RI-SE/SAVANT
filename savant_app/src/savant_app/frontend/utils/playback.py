# savant_app/frontend/utils/playback.py
from PyQt6.QtCore import QTimer, Qt
from .render import show_frame


def wire(mw):
    if not hasattr(mw, "_play_timer"):
        mw._play_timer = QTimer(mw)
    _reset_timer_connection(mw)
    mw._is_playing = False

    pc = mw.playback_controls

    if hasattr(pc, "play_clicked"):
        _safe_connect(pc.play_clicked, lambda: _toggle_play(mw))

    if hasattr(pc, "pause_clicked"):
        _safe_connect(pc.pause_clicked, lambda: _stop(mw))

    if hasattr(pc, "next_frame_clicked"):
        _safe_connect(pc.next_frame_clicked, lambda: _step_once(mw, direction=+1))

    if hasattr(pc, "prev_frame_clicked"):
        _safe_connect(pc.prev_frame_clicked, lambda: _step_once(mw, direction=-1))

    if hasattr(pc, "skip_backward_clicked"):
        _safe_connect(pc.skip_backward_clicked, lambda n: _skip(mw, -n))

    if hasattr(pc, "skip_forward_clicked"):
        _safe_connect(pc.skip_forward_clicked, lambda n: _skip(mw, +n))


def _safe_connect(signal, slot):
    try:
        signal.connect(slot, Qt.ConnectionType.UniqueConnection)
    except TypeError:
        pass


def _reset_timer_connection(mw):
    try:
        mw._play_timer.timeout.disconnect()
    except TypeError:
        pass
    mw._play_timer.timeout.connect(lambda: _tick(mw), Qt.ConnectionType.UniqueConnection)


def _toggle_play(mw):
    if mw._is_playing:
        _stop(mw)
    else:
        _start(mw)


def _start(mw):
    _reset_timer_connection(mw)

    try:
        total = mw.video_controller.total_frames()
        idx = mw.video_controller.current_index()
        if idx < 0 or (total and idx >= total - 1):
            pixmap, j = mw.video_controller.jump_to_frame(0)
            show_frame(mw, pixmap, j)
    except Exception:
        pass

    fps = 0
    try:
        fps = mw.video_controller.fps()
    except Exception:
        pass
    if not fps or fps <= 0:
        fps = 25

    mw._play_timer.start(max(1, int(1000 / int(fps))))
    mw._is_playing = True
    if hasattr(mw.playback_controls, "set_playing"):
        mw.playback_controls.set_playing(True)


def _stop(mw):
    if mw._play_timer.isActive():
        mw._play_timer.stop()
    mw._is_playing = False
    if hasattr(mw.playback_controls, "set_playing"):
        mw.playback_controls.set_playing(False)


def _tick(mw):
    try:
        pixmap, idx = mw.video_controller.next_frame()
        if pixmap is None or idx is None:
            raise StopIteration
        show_frame(mw, pixmap, idx)
    except StopIteration:
        _stop(mw)
    except Exception:
        _stop(mw)


def _step_once(mw, direction: int):
    try:
        if direction >= 0:
            pixmap, idx = mw.video_controller.next_frame()
        else:
            pixmap, idx = mw.video_controller.previous_frame()

        show_frame(mw, pixmap, idx)
    except Exception:
        _stop(mw)


def _skip(mw, n: int):
    try:
        pixmap, idx = mw.video_controller.skip_frames(n)
        show_frame(mw, pixmap, idx)
    except Exception:
        _stop(mw)
