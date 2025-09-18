from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QPen, QColor, QPolygonF, QBrush
from PyQt6.QtCore import Qt, QPointF, pyqtSignal, QRectF
from typing import List, Tuple
import math


class Overlay(QWidget):
    """
    Draws rotated bounding boxes over the video. Boxes are expected as
    (center_x_vid, center_y_vid, width_vid, height_vid, theta_radians)
    in ORIGINAL VIDEO PIXELS (not scaled).

    Rotation is applied manually via cos/sin so it will ALWAYS be visible
    if theta != 0, regardless of painter transformations.
    """

    boxMoved = pyqtSignal(int, float, float)
    boxResized = pyqtSignal(int, float, float, float, float)
    boxRotated = pyqtSignal(int, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self._frame_size: Tuple[int, int] = (0, 0)
        self._zoom: float = 1.0
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0
        self._boxes: List[Tuple[float, float, float, float, float]] = []

        # This flag flips the sign if needed (Rotation of boxes).
        self._theta_is_clockwise: bool = True

        # Debug helpers (Shows the center and axes of the boxes)
        self._show_centers: bool = True
        self._show_axes: bool = True
        self._pen_center = QPen(QColor(255, 0, 0), 1)
        self._pen_axis = QPen(QColor(0, 200, 255), 1)

        # TODO set box colour from settings (Can be per annotator)
        # Box colour
        self._pen_box = QPen(QColor(0, 255, 0), 2)
        self._pen_box_selected = QPen(QColor(255, 255, 0))
        self._pen_box_selected.setWidth(3)

        # Interaction state
        self._interactive: bool = True
        self._selected_idx: int | None = None
        self._drag_mode: str | None = None
        self._press_pos_disp: QPointF | None = None
        self._orig_box: Tuple[float, float, float, float, float] | None = None
        self._hit_tol_px: float = 14.0
        self._handle_draw_px: float = 14.0

        # Hover feedback
        self._hover_idx: int | None = None
        self._hover_mode: str | None = None
        self._pen_hover_edge = QPen(QColor(0, 200, 255))
        self._pen_hover_edge.setWidth(5)

        # Rotation handle
        self._rotate_handle_offset_px = 24
        self._pen_rotate_handle = QPen(QColor(0, 200, 255))
        self._pen_rotate_handle.setWidth(2)
        self._brush_rotate_handle = QBrush(QColor(0, 200, 255))
        self._press_angle = 0.0
        self._orig_theta = 0.0

    def set_frame_size(self, width_vid: int, height_vid: int):
        self._frame_size = (width_vid, height_vid)
        self.update()

    def set_zoom(self, z: float):
        self._zoom = max(0.05, min(z, 20.0))
        self.update()

    def set_pan(self, pan_x: float, pan_y: float):
        self._pan_x, self._pan_y = pan_x, pan_y
        self.update()

    def set_rotated_boxes(self, boxes: List[Tuple[float, float, float, float, float]]):
        """Boxes in VIDEO coords:
        (center_x_vid, center_y_vid, width_vid, height_vid, theta_radians).
        """
        self._boxes = boxes or []
        self.update()

    def set_theta_clockwise(self, clockwise: bool):
        """If your dataset'sin_a theta increases clockwise, keep True (default)."""
        self._theta_is_clockwise = clockwise
        self.update()

    def show_centers(self, enabled: bool):
        self._show_centers = enabled
        self.update()

    def show_axes(self, enabled: bool):
        self._show_axes = enabled
        self.update()

    def set_interactive(self, enabled: bool) -> None:
        """Enable/disable selection and drag/resize in the overlay."""
        self._interactive = bool(enabled)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, not enabled)

    def selected_index(self) -> int | None:
        """Return selected overlay index or None if nothing selected."""
        return self._selected_idx

    def _display_to_video(self, x_disp: float, y_disp: float) -> QPointF:
        scale, off_x, off_y, _ = self._compute_transform()
        if scale == 0:
            return QPointF(0.0, 0.0)
        return QPointF((x_disp - off_x) / scale, (y_disp - off_y) / scale)

    def mousePressEvent(self, ev):
        if ev.button() in (Qt.MouseButton.MiddleButton,) or (
            ev.button() == Qt.MouseButton.LeftButton
            and (ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
        ):
            ev.ignore()
            return

        if not self._interactive or ev.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(ev)

        idx, mode = self._hit_test(ev.position())

        if idx is None:
            self._selected_idx = None
            self._drag_mode = None
            self._hover_idx, self._hover_mode = None, None
            self.update()
            ev.ignore()
            return

        # Hit → select & start drag
        self._selected_idx = idx
        self._drag_mode = mode
        self._press_pos_disp = ev.position()
        self._orig_box = self._boxes[idx]

        if mode == "R":
            cx0, cy0, _, _, theta0 = self._orig_box
            scale, off_x, off_y, _ = self._compute_transform()
            px = (self._press_pos_disp.x() - off_x) / scale
            py = (self._press_pos_disp.y() - off_y) / scale
            self._press_angle = math.atan2(py - cy0, px - cx0)
            self._orig_theta = theta0

        self.update()
        ev.accept()

    def mouseMoveEvent(self, ev):
        if not self._interactive:
            return super().mouseMoveEvent(ev)

        if self._drag_mode is None:
            idx, mode = self._hit_test(ev.position())
            if idx != self._hover_idx or mode != self._hover_mode:
                self._hover_idx, self._hover_mode = idx, mode
                self.update()

            if mode in ("E", "W"):
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            elif mode in ("N", "S"):
                self.setCursor(Qt.CursorShape.SizeVerCursor)
            elif mode in ("move", "R"):
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                self.unsetCursor()

            ev.ignore()
            return

        # Dragging a selected box/handle
        if self._selected_idx is None or self._orig_box is None:
            return super().mouseMoveEvent(ev)

        cur_disp = ev.position()
        dx_disp = cur_disp.x() - self._press_pos_disp.x()
        dy_disp = cur_disp.y() - self._press_pos_disp.y()

        scale, off_x, off_y, _ = self._compute_transform()
        if scale <= 0:
            return
        dx_vid = dx_disp / scale
        dy_vid = dy_disp / scale

        cx0, cy0, w0, h0, theta = self._orig_box
        cx, cy, w, h = cx0, cy0, w0, h0

        if self._drag_mode == "move":
            cx = cx0 + dx_vid
            cy = cy0 + dy_vid

        elif self._drag_mode in ("E", "W", "N", "S"):
            angle_rad = -theta if self._theta_is_clockwise else theta
            ct = math.cos(angle_rad)
            st = math.sin(angle_rad)

            ex_x, ex_y = ct, st
            ey_x, ey_y = -st, ct

            # Project mouse delta (video coords) onto each local axis
            d_along_x = dx_vid * ex_x + dy_vid * ex_y
            d_along_y = dx_vid * ey_x + dy_vid * ey_y

            if self._drag_mode == "E":
                d = d_along_x
                w = max(1e-6, w0 + d)
                cx = cx0 + 0.5 * d * ex_x
                cy = cy0 + 0.5 * d * ex_y

            elif self._drag_mode == "W":
                d = d_along_x
                w = max(1e-6, w0 - d)
                cx = cx0 + 0.5 * d * ex_x
                cy = cy0 + 0.5 * d * ex_y

            elif self._drag_mode == "S":
                d = d_along_y
                h = max(1e-6, h0 + d)
                cx = cx0 + 0.5 * d * ey_x
                cy = cy0 + 0.5 * d * ey_y

            elif self._drag_mode == "N":
                d = d_along_y
                h = max(1e-6, h0 - d)
                cx = cx0 + 0.5 * d * ey_x
                cy = cy0 + 0.5 * d * ey_y

        elif self._drag_mode == "R":
            mx = (ev.position().x() - off_x) / scale
            my = (ev.position().y() - off_y) / scale
            cur_angle = math.atan2(my - cy0, mx - cx0)
            d_theta = cur_angle - self._press_angle
            if d_theta > math.pi:
                d_theta -= 2 * math.pi
            if d_theta <= -math.pi:
                d_theta += 2 * math.pi
            if self._theta_is_clockwise:
                theta = self._orig_theta - d_theta
            else:
                theta = self._orig_theta + d_theta

            cx, cy, w, h = cx0, cy0, w0, h0
            self._boxes[self._selected_idx] = (cx, cy, w, h, theta)
            self.update()
            ev.accept()
            return

        # live preview
        self._boxes[self._selected_idx] = (cx, cy, w, h, theta)
        self.update()
        ev.accept()

    def mouseReleaseEvent(self, ev):
        if not self._interactive or ev.button() != Qt.MouseButton.LeftButton:
            return super().mouseReleaseEvent(ev)

        if self._drag_mode is None:
            ev.ignore()
            return

        if self._selected_idx is not None:
            cx, cy, w, h, theta = self._boxes[self._selected_idx]
            if self._drag_mode == "move":
                self.boxMoved.emit(self._selected_idx, cx, cy)
            elif self._drag_mode == "R":
                self.boxRotated.emit(self._selected_idx, theta)
            else:
                self.boxResized.emit(self._selected_idx, cx, cy, w, h)

        # clear drag/hover state
        self._hover_idx, self._hover_mode = None, None
        self._drag_mode = None
        self._press_pos_disp = None
        self._orig_box = None
        ev.accept()

    def wheelEvent(self, ev):
        ev.ignore()

    # ---------------- hit-testing ----------------

    def _hit_test(self, pos_disp):
        """
        Return (box_index, mode) where mode in {"move","N","S","E","W","R"}; or (None,None).
        DISPLAY-space hit-test that matches paintEvent exactly, including _theta_is_clockwise.
        """
        if not getattr(self, "_boxes", None):
            return None, None

        scale, off_x, off_y, _ = self._compute_transform()
        tol = getattr(self, "_hit_tol_px", 14.0)
        s = max(tol, getattr(self, "_handle_draw_px", tol))
        half = s * 0.5

        x = pos_disp.x()
        y = pos_disp.y()

        for idx, (cx_vid, cy_vid, w_vid, h_vid, theta) in enumerate(self._boxes):
            cx_disp = off_x + cx_vid * scale
            cy_disp = off_y + cy_vid * scale
            box_w_disp = w_vid * scale
            box_h_disp = h_vid * scale

            angle_rad = -theta if self._theta_is_clockwise else theta
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

            half_w = box_w_disp * 0.5
            half_h = box_h_disp * 0.5

            # corners in DISPLAY coords (same local order as paintEvent)
            corners = []
            for lx, ly in [(-half_w, -half_h), (half_w, -half_h),
                           (half_w, half_h), (-half_w, half_h)]:
                rx = lx * cos_a - ly * sin_a
                ry = lx * sin_a + ly * cos_a
                corners.append(QPointF(cx_disp + rx, cy_disp + ry))

            TL, TR, BR, BL = corners

            # midpoints (DISPLAY)
            mid_top = QPointF((TL.x()+TR.x())*0.5, (TL.y()+TR.y())*0.5)
            mid_right = QPointF((TR.x()+BR.x())*0.5, (TR.y()+BR.y())*0.5)
            mid_bottom = QPointF((BL.x()+BR.x())*0.5, (BL.y()+BR.y())*0.5)
            mid_left = QPointF((TL.x()+BL.x())*0.5, (TL.y()+BL.y())*0.5)

            # side handle hits (square around midpoint)
            def in_handle(pt: QPointF) -> bool:
                return (pt.x()-half <= x <= pt.x()+half) and (pt.y()-half <= y <= pt.y()+half)

            if in_handle(mid_top):
                return idx, "N"
            if in_handle(mid_right):
                return idx, "E"
            if in_handle(mid_bottom):
                return idx, "S"
            if in_handle(mid_left):
                return idx, "W"

            # rotate handle (ALWAYS OUTSIDE top edge) — same outward normal logic as paintEvent
            tx, ty = (TR.x() - TL.x()), (TR.y() - TL.y())
            edge_len = math.hypot(tx, ty) or 1.0
            nx, ny = (-ty / edge_len, tx / edge_len)

            # flip normal if it points toward the center → ensure "outside"
            vec_cx, vec_cy = (cx_disp - mid_top.x()), (cy_disp - mid_top.y())
            if nx * vec_cx + ny * vec_cy > 0:
                nx, ny = -nx, -ny

            handle_offset = getattr(self, "_rotate_handle_offset_px", 24)
            rot_cx = mid_top.x() + nx * handle_offset
            rot_cy = mid_top.y() + ny * handle_offset
            rot_half = getattr(self, "_handle_draw_px", tol) * 0.5
            if (rot_cx - rot_half <= x <= rot_cx + rot_half) and (
                    rot_cy - rot_half <= y <= rot_cy + rot_half):
                return idx, "R"

            # inside test (point-in-quad) → MOVE
            def tri_area(a, b, c):
                return ((b.x()-a.x())*(c.y()-a.y()) - (b.y()-a.y())*(c.x()-a.x()))
            P = QPointF(x, y)
            a1 = tri_area(TL, TR, P)
            a2 = tri_area(TR, BR, P)
            a3 = tri_area(BR, BL, P)
            a4 = tri_area(BL, TL, P)
            if (a1 >= 0 and a2 >= 0 and a3 >= 0 and a4 >= 0) or (
                    a1 <= 0 and a2 <= 0 and a3 <= 0 and a4 <= 0):
                return idx, "move"

        return None, None

    def _compute_transform(self) -> tuple[float, float, float, float]:
        """
        Returns (scale, off_x, off_y, base) where:
          base  = min(viewW/frameW, viewH/frameH)
          scale = base * self._zoom
          off_x/off_y = letterbox offsets + pan
        """
        frame_width, frame_height = self._frame_size
        display_width, display_height = self.width(), self.height()
        if (
            frame_width <= 0
            or frame_height <= 0
            or display_width <= 0
            or display_height <= 0
        ):
            return (1.0, 0.0, 0.0, 1.0)

        base = min(display_width / frame_width, display_height / frame_height)
        scale = base * self._zoom
        draw_w, draw_h = frame_width * scale, frame_height * scale
        off_x = (display_width - draw_w) / 2 + self._pan_x
        off_y = (display_height - draw_h) / 2 + self._pan_y
        return (scale, off_x, off_y, base)

    def paintEvent(self, _):
        if not self._boxes or self._frame_size == (0, 0):
            return

        scale, offset_x, offset_y, _ = self._compute_transform()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        for idx, (center_x_vid, center_y_vid, width_vid, height_vid, theta) in enumerate(
                self._boxes):
            center_x_disp = offset_x + center_x_vid * scale
            center_y_disp = offset_y + center_y_vid * scale
            box_width_disp = width_vid * scale
            box_height_disp = height_vid * scale

            angle_rad = -theta if self._theta_is_clockwise else theta
            cos_angle, sin_angle = math.cos(angle_rad), math.sin(angle_rad)

            half_w = box_width_disp / 2.0
            half_h = box_height_disp / 2.0

            corners = []
            for local_offset_x, local_offset_y in [
                (-half_w, -half_h), (half_w, -half_h),
                (half_w, half_h), (-half_w, half_h)
            ]:
                rotated_offset_x = local_offset_x * cos_angle - local_offset_y * sin_angle
                rotated_offset_y = local_offset_x * sin_angle + local_offset_y * cos_angle
                corner_x_disp = center_x_disp + rotated_offset_x
                corner_y_disp = center_y_disp + rotated_offset_y
                corners.append(QPointF(corner_x_disp, corner_y_disp))

            polygon = QPolygonF(corners)

            # outline (selected highlighted)
            if self._selected_idx == idx:
                painter.setPen(self._pen_box_selected)
            else:
                painter.setPen(self._pen_box)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPolygon(polygon)

            TL, TR, BR, BL = corners[0], corners[1], corners[2], corners[3]

            # draw side handles when selected
            if self._selected_idx == idx:
                s = self._handle_draw_px
                half = s / 2.0
                mid_top = QPointF((TL.x()+TR.x())*0.5, (TL.y()+TR.y())*0.5)
                mid_right = QPointF((TR.x()+BR.x())*0.5, (TR.y()+BR.y())*0.5)
                mid_bottom = QPointF((BL.x()+BR.x())*0.5, (BL.y()+BR.y())*0.5)
                mid_left = QPointF((TL.x()+BL.x())*0.5, (TL.y()+BL.y())*0.5)

                painter.setPen(QPen(QColor(30, 30, 30)))
                painter.setBrush(QColor(255, 255, 255))
                for pt in (mid_top, mid_right, mid_bottom, mid_left):
                    rectf = QRectF(float(pt.x() - half), float(pt.y() - half), float(s), float(s))
                    painter.drawRect(rectf)

                # rotate handle above top edge (always outside) + connector
                tx, ty = (TR.x() - TL.x()), (TR.y() - TL.y())
                edge_len = math.hypot(tx, ty) or 1.0
                nx, ny = (-ty / edge_len, tx / edge_len)

                vec_cx, vec_cy = (center_x_disp - mid_top.x()), (center_y_disp - mid_top.y())
                if nx * vec_cx + ny * vec_cy > 0:
                    nx, ny = -nx, -ny

                handle_offset = self._rotate_handle_offset_px
                rot_cx = mid_top.x() + nx * handle_offset
                rot_cy = mid_top.y() + ny * handle_offset

                # connector line
                pen_line = QPen(QColor(0, 200, 255))
                pen_line.setWidth(2)
                pen_line.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen_line)
                painter.drawLine(mid_top, QPointF(rot_cx, rot_cy))

                # handle circle
                r = self._handle_draw_px * 0.5
                painter.setPen(self._pen_rotate_handle)
                painter.setBrush(self._brush_rotate_handle)
                painter.drawEllipse(QRectF(rot_cx - r, rot_cy - r, 2*r, 2*r))

            # hover edge indicator (cyan, fat)
            if idx == self._hover_idx and self._hover_mode in ("N", "S", "E", "W"):
                painter.setPen(self._pen_hover_edge)
                if self._hover_mode == "N":
                    painter.drawLine(TL, TR)
                elif self._hover_mode == "S":
                    painter.drawLine(BL, BR)
                elif self._hover_mode == "E":
                    painter.drawLine(TR, BR)
                elif self._hover_mode == "W":
                    painter.drawLine(TL, BL)

            # centers / axes (optional debug)
            if self._show_centers:
                painter.setPen(self._pen_center)
                painter.drawLine(QPointF(center_x_disp - 4, center_y_disp),
                                 QPointF(center_x_disp + 4, center_y_disp))
                painter.drawLine(QPointF(center_x_disp, center_y_disp - 4),
                                 QPointF(center_x_disp, center_y_disp + 4))

            if self._show_axes:
                painter.setPen(self._pen_axis)
                axis_x = center_x_disp + half_w * cos_angle
                axis_y = center_y_disp + half_h * sin_angle
                painter.drawLine(QPointF(center_x_disp, center_y_disp),
                                 QPointF(axis_x, axis_y))

        painter.end()
