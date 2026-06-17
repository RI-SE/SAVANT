from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut

from edit.frontend.utils.settings_store import get_bbox_zoom_padding, get_zoom_rate
from edit.frontend.widgets.video_display import CANVAS_PADDING_FRACTION


def wire(main_window, initial: float | None = None):

    def _clamp(zoom_value: float) -> float:
        return max(0.05, min(zoom_value, 20.0))

    def _apply_zoom(zoom_value: float, anchor_position=None):
        main_window._zoom = _clamp(zoom_value)
        if hasattr(main_window.video_widget, "set_zoom"):
            if anchor_position is not None:
                main_window.video_widget.set_zoom(main_window._zoom, anchor_position)
            else:
                main_window.video_widget.set_zoom(main_window._zoom)
        if hasattr(main_window.overlay, "set_zoom"):
            main_window.overlay.set_zoom(main_window._zoom)
        if hasattr(main_window.overlay, "update"):
            main_window.overlay.update()

    def zoom_in(anchor_position=None):
        _apply_zoom(main_window._zoom * 1.1, anchor_position)

    def zoom_out(anchor_position=None):
        _apply_zoom(main_window._zoom / 1.1, anchor_position)

    def zoom_fit():
        target = getattr(main_window, "_default_zoom", None) or 1.0
        _apply_zoom(target)
        main_window.video_widget.set_pan(0.0, 0.0)

    def _set_default_zoom(value: float, *, apply: bool = False):
        main_window._default_zoom = _clamp(value)
        if apply:
            _apply_zoom(main_window._default_zoom)

    default_zoom = initial if initial is not None else get_zoom_rate()
    if default_zoom <= 0:
        default_zoom = 1.0
    main_window._zoom = default_zoom
    _set_default_zoom(default_zoom, apply=True)

    def _wheel_zoom(event):
        mods = event.modifiers()
        if mods & (
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier
        ):
            delta = event.angleDelta().y()
            cursor_position = event.position()
            if delta > 0:
                zoom_in(cursor_position)
            elif delta < 0:
                zoom_out(cursor_position)
            event.accept()
        else:
            event.ignore()

    def _zoom_to_rect(rect):
        """Zoom and pan so the given display-space rectangle fills the viewport."""
        vw = main_window.video_widget
        viewport_w = vw.width()
        viewport_h = vw.height()
        if rect.width() <= 0 or rect.height() <= 0:
            return

        # The rect is in widget coordinates at the current zoom level.
        # Calculate how much to scale up to fill the viewport.
        zoom_factor = min(viewport_w / rect.width(), viewport_h / rect.height())
        new_zoom = _clamp(main_window._zoom * zoom_factor)

        # Convert the rectangle's center from display coords to image coords
        # at the OLD zoom level, then compute the pan needed at the NEW zoom.
        base_scale = vw._fit_scale()
        old_scale = base_scale * main_window._zoom
        if old_scale <= 0:
            return

        draw_rect = vw._draw_rect()
        # Image-space coordinates of the rectangle center
        img_cx = (rect.center().x() - draw_rect.left()) / old_scale
        img_cy = (rect.center().y() - draw_rect.top()) / old_scale

        new_scale = base_scale * new_zoom
        new_draw_w = vw._pixmap.width() * new_scale
        new_draw_h = vw._pixmap.height() * new_scale
        centered_off_x = (viewport_w - new_draw_w) / 2
        centered_off_y = (viewport_h - new_draw_h) / 2

        # We want img_cx/img_cy to appear at the viewport center
        target_disp_x = viewport_w / 2
        target_disp_y = viewport_h / 2
        pan_x = target_disp_x - (centered_off_x + img_cx * new_scale)
        pan_y = target_disp_y - (centered_off_y + img_cy * new_scale)

        _apply_zoom(new_zoom)
        vw.set_pan(pan_x, pan_y)

    def _toggle_zoom_rect():
        vw = main_window.video_widget
        if vw._zoom_rect_mode:
            vw.cancel_zoom_rect_mode()
        else:
            vw.start_zoom_rect_mode()

    if hasattr(main_window.video_widget, "setMouseTracking"):
        main_window.video_widget.setMouseTracking(True)
    main_window.video_widget.wheelEvent = _wheel_zoom

    # Connect zoom-rect signal
    main_window.video_widget.zoom_rect_selected.connect(_zoom_to_rect)

    QShortcut(
        QKeySequence(QKeySequence.StandardKey.ZoomIn), main_window, activated=zoom_in
    )
    QShortcut(
        QKeySequence(QKeySequence.StandardKey.ZoomOut), main_window, activated=zoom_out
    )
    QShortcut(QKeySequence("Ctrl+0"), main_window, activated=zoom_fit)
    QShortcut(QKeySequence("Z"), main_window, activated=_toggle_zoom_rect)

    # Wire reset view button if present
    if hasattr(main_window, "playback_controls") and hasattr(
        main_window.playback_controls, "reset_view_clicked"
    ):
        main_window.playback_controls.reset_view_clicked.connect(zoom_fit)

    def zoom_to_bbox(bbox):
        """Zoom and pan so a BBoxData (video coords) fills the viewport with padding."""
        vw = main_window.video_widget
        if not vw._pixmap or vw._pixmap.isNull():
            return
        viewport_w = vw.width()
        viewport_h = vw.height()
        if bbox.width <= 0 or bbox.height <= 0:
            return

        PADDING = get_bbox_zoom_padding()

        base_scale = vw._fit_scale()
        padded_w = bbox.width * PADDING
        padded_h = bbox.height * PADDING

        new_zoom = _clamp(min(
            viewport_w / (padded_w * base_scale),
            viewport_h / (padded_h * base_scale),
        ))

        new_scale = base_scale * new_zoom
        new_draw_w = vw._pixmap.width() * new_scale
        new_draw_h = vw._pixmap.height() * new_scale
        centered_off_x = (viewport_w - new_draw_w) / 2
        centered_off_y = (viewport_h - new_draw_h) / 2

        # Desired pan: center the bbox in the viewport.
        pan_x = viewport_w / 2 - (centered_off_x + bbox.center_x * new_scale)
        pan_y = viewport_h / 2 - (centered_off_y + bbox.center_y * new_scale)

        # Clamp pan to what video_display._clamp_pan() will allow, so we
        # can then check if the bbox is still fully visible after clamping.
        pad_x = new_draw_w * CANVAS_PADDING_FRACTION
        pad_y = new_draw_h * CANVAS_PADDING_FRACTION
        max_pan_x = max(0.0, (new_draw_w - viewport_w) / 2) + pad_x
        max_pan_y = max(0.0, (new_draw_h - viewport_h) / 2) + pad_y
        pan_x = max(-max_pan_x, min(pan_x, max_pan_x))
        pan_y = max(-max_pan_y, min(pan_y, max_pan_y))

        # After clamping, verify the bbox edges are visible; if not, nudge the
        # pan to the nearest position that keeps the bbox fully in view.
        half_w = bbox.width * new_scale / 2
        half_h = bbox.height * new_scale / 2
        bbox_screen_cx = centered_off_x + bbox.center_x * new_scale + pan_x
        bbox_screen_cy = centered_off_y + bbox.center_y * new_scale + pan_y

        # Push pan so bbox left/right/top/bottom edges are all inside viewport
        if bbox_screen_cx - half_w < 0:
            pan_x += half_w - bbox_screen_cx
        elif bbox_screen_cx + half_w > viewport_w:
            pan_x -= (bbox_screen_cx + half_w - viewport_w)
        if bbox_screen_cy - half_h < 0:
            pan_y += half_h - bbox_screen_cy
        elif bbox_screen_cy + half_h > viewport_h:
            pan_y -= (bbox_screen_cy + half_h - viewport_h)

        # Re-clamp after nudging (can't go past image edge)
        pan_x = max(-max_pan_x, min(pan_x, max_pan_x))
        pan_y = max(-max_pan_y, min(pan_y, max_pan_y))

        _apply_zoom(new_zoom)
        vw.set_pan(pan_x, pan_y)

    def pan_to_selected_bbox():
        """Pan so the currently selected bbox is centred in the viewport (zoom unchanged).

        Uses the overlay's own _compute_transform() so the computed pan is guaranteed
        to match exactly how the overlay renders the selected box — regardless of any
        difference between the video pixmap size and the overlay's frame_size.
        """
        vw = main_window.video_widget
        overlay = getattr(main_window, "overlay", None)
        if overlay is None:
            return
        bbox = overlay._get_selected_bbox()
        if bbox is None:
            return

        # _compute_transform returns (scale, off_x, off_y, base_scale)
        scale, off_x, off_y, _ = overlay._compute_transform()
        if scale <= 0:
            return

        display_w = overlay.width()
        display_h = overlay.height()

        # Current screen-space position of the bbox centre (before any pan change).
        current_screen_x = off_x + bbox.center_x * scale
        current_screen_y = off_y + bbox.center_y * scale

        # Delta needed to shift the bbox centre to the viewport centre.
        dx = display_w / 2 - current_screen_x
        dy = display_h / 2 - current_screen_y

        # New pan = current overlay pan + required delta.
        new_pan_x = overlay._pan_x + dx
        new_pan_y = overlay._pan_y + dy

        # Use exact pan (no clamping) so edge bboxes reach the true centre.
        vw.set_pan_exact(new_pan_x, new_pan_y)

    main_window.zoom_in = zoom_in
    main_window.zoom_out = zoom_out
    main_window.zoom_fit = zoom_fit
    main_window.zoom_to_bbox = zoom_to_bbox
    main_window.pan_to_selected_bbox = pan_to_selected_bbox
    main_window.set_default_zoom = lambda value, *, apply=False: _set_default_zoom(
        value, apply=apply
    )
