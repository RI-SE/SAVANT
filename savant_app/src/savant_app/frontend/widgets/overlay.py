from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QPen, QColor, QPolygonF, QBrush
from PyQt6.QtCore import Qt, QPointF, pyqtSignal, QRectF
from typing import List, Tuple
import math
from savant_app.frontend.types import BBoxData
from savant_app.frontend.widgets.cascade_dropdown import CascadeDropdown
from savant_app.frontend.widgets.cascade_button import CascadeButton


class Overlay(QWidget):
    """
    Draws rotated bounding boxes over the video. Boxes are expected as
    BBox instances in VIDEO coords: (center_x, center_y, width, height, theta)
    in ORIGINAL VIDEO PIXELS (not scaled).

    Rotation is applied manually via cos/sin so it will ALWAYS be visible
    if theta != 0, regardless of painter transformations.
    """

    # TODO: Refactor these to use custom dataclass types!
    boxMoved = pyqtSignal(str, float, float)  # (object_id, x, y)
    boxResized = pyqtSignal(
        str, float, float, float, float, float
    )  # (object_id, x, y, w, h, rotate_angle)
    boxRotated = pyqtSignal(
        str, float, float, float
    )  # (object_id, width, height, rotation)
    bounding_box_selected = pyqtSignal(str)  # (object_id)
    deletePressed = pyqtSignal()
    cascadeApplyAll = pyqtSignal(
        str, object, object, object
    )  # (object_id, width, height, theta). Object lets us pass optional floats
    #cascadeApplyRotationAll = pyqtSignal(
    #    str, float, float, float
    #)  # (object_id, width, height, theta)
    cascadeApplyFrameRange = pyqtSignal(
        str, object, object, object 
    )  # (object_id, width, height, theta). Object lets us pass optional floats
    #cascadeApplyRotationFrameRange = pyqtSignal(
    #    str, float, float, float
    #)  # (object_id, width, height, theta)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self._frame_size: Tuple[int, int] = (0, 0)
        self._zoom: float = 1.0
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0
        self._boxes: List[BBoxData] = []

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
        self._orig_box: BBoxData | None = None
        self._hit_tol_px: float = 14.0
        self._handle_draw_px: float = 14.0

        # Hover feedback
        self.setMouseTracking(True)
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
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Cascade dropdown
        self.cascade_dropdown = CascadeDropdown(self)
        self.cascade_dropdown.applySizeToAll.connect(self._on_cascade_size_to_all)
        self.cascade_dropdown.applyRotationToAll.connect(self._on_cascade_rotation_to_all)
        self.cascade_dropdown.applySizeToFrameRange.connect(
            self._on_cascade_size_to_frame_range
        )
        self.cascade_dropdown.applyRotationToFrameRange.connect(
            self._on_cascade_rotation_to_frame_range
        )
        self.cascade_dropdown.cancelled.connect(self._on_cascade_cancel)

        # Cascade button
        self.cascade_button = CascadeButton(self)
        self.cascade_button.clicked.connect(self._on_cascade_button_clicked)

    def set_frame_size(self, width_vid: int, height_vid: int):
        self._frame_size = (width_vid, height_vid)
        self.update()

    def set_zoom(self, z: float):
        self._zoom = max(0.05, min(z, 20.0))
        self.update()

    def set_pan(self, pan_x: float, pan_y: float):
        self._pan_x, self._pan_y = pan_x, pan_y
        self.update()

    def set_rotated_boxes(self, boxes: List[BBoxData]):
        """Set boxes as BBox instances in VIDEO coords."""
        self._boxes = []
        if boxes is not None:
            for box in boxes:
                if isinstance(box, BBoxData):
                    self._boxes.append(box)
        self.update()

    def set_theta_clockwise(self, clockwise: bool):
        """If your dataset's theta increases clockwise, keep True (default)."""
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

    def selected_object_id(self) -> str | None:
        """Return selected overlay object ID or None if nothing selected."""
        if self._selected_idx is not None and 0 <= self._selected_idx < len(
            self._boxes
        ):
            return self._boxes[self._selected_idx].object_id
        return None

    def clear_selection(self) -> None:
        """Clear any current selection and hover/drag state."""
        self._selected_idx = None
        self._drag_mode = None
        self._hover_idx = None
        self._hover_mode = None
        self._press_pos_disp = None
        self._orig_box = None
        # Hide the cascade dropdown and button when selection is cleared
        if hasattr(self, "cascade_dropdown"):
            self.cascade_dropdown.hide()
        if hasattr(self, "cascade_button"):
            self.cascade_button.hide()
        self.update()

    def _display_to_video(self, x_disp: float, y_disp: float) -> QPointF:
        scale, off_x, off_y, _ = self._compute_transform()
        if scale == 0:
            return QPointF(0.0, 0.0)
        return QPointF((x_disp - off_x) / scale, (y_disp - off_y) / scale)

    def mousePressEvent(self, ev):
        self.setFocus()
        if ev.button() in (Qt.MouseButton.MiddleButton,) or (
            ev.button() == Qt.MouseButton.LeftButton
            and (ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
        ):
            ev.ignore()
            return

        if not self._interactive or ev.button() != Qt.MouseButton.LeftButton:
            # If we have cascade components visible, hide them when clicking outside
            if (
                hasattr(self, "cascade_dropdown") and self.cascade_dropdown.isVisible()
            ) or (hasattr(self, "cascade_button") and self.cascade_button.isVisible()):
                self.clear_selection()
            return super().mousePressEvent(ev)

        idx, mode = self._hit_test(ev.position())

        if idx is None:
            self._selected_idx = None
            self._drag_mode = None
            self._hover_idx, self._hover_mode = None, None
            # Clear selection in the sidebar too
            self.bounding_box_selected.emit(self.selected_object_id())
            # Hide cascade components when clicking away from any box
            self.clear_selection()
            self.update()
            ev.ignore()
            return

        # Hit → select & start drag
        self._selected_idx = idx
        self._drag_mode = mode
        self._press_pos_disp = ev.position()
        self._orig_box = self._boxes[idx]

        # Clicked on a box (or its handle), so trigger
        # the highlight of the corresponding object in the sidebar.
        self.bounding_box_selected.emit(self.selected_object_id())

        # Handle cascade button option
        self.show_cascade_option()

        if mode == "R":
            cx0 = self._orig_box.center_x
            cy0 = self._orig_box.center_y
            scale, off_x, off_y, _ = self._compute_transform()
            px = (self._press_pos_disp.x() - off_x) / scale
            py = (self._press_pos_disp.y() - off_y) / scale
            self._press_angle = math.atan2(py - cy0, px - cx0)
            self._orig_theta = self._orig_box.theta

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

            if mode in ("E", "W", "N", "S"):
                theta = self._boxes[idx].theta
                ax_x, ax_y = self._axis_from_mode(theta, mode)
                self.setCursor(self._cursor_for_axis(ax_x, ax_y))
            elif mode == "move":
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            elif mode == "R":
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.unsetCursor()

            ev.accept()
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

        # Get current bbox properties
        cx0 = self._orig_box.center_x
        cy0 = self._orig_box.center_y
        w0 = self._orig_box.width
        h0 = self._orig_box.height
        theta = self._orig_box.theta
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

        # Create new BBox for live preview
        current_box = self._boxes[self._selected_idx]
        object_id = "unknown"
        if isinstance(current_box, BBoxData):
            object_id = current_box.object_id

        new_bbox = BBoxData(
            object_id=object_id,
            object_type=current_box.object_type,
            center_x=cx,
            center_y=cy,
            width=w,
            height=h,
            theta=theta,
        )
        self._boxes[self._selected_idx] = new_bbox
        self.update()
        ev.accept()

    def mouseReleaseEvent(self, ev):
        if not self._interactive or ev.button() != Qt.MouseButton.LeftButton:
            return super().mouseReleaseEvent(ev)

        if self._drag_mode is None:
            ev.ignore()
            return

        selected_bbox = self._get_selected_bbox()
        if selected_bbox is not None:
            if self._drag_mode == "move":
                self.boxMoved.emit(
                    selected_bbox.object_id,
                    selected_bbox.center_x,
                    selected_bbox.center_y,
                )
            elif self._drag_mode == "R":
                self.boxRotated.emit(
                    selected_bbox.object_id,
                    selected_bbox.width,
                    selected_bbox.height,
                    selected_bbox.theta,
                )
            else:
                self.boxResized.emit(
                    selected_bbox.object_id,
                    selected_bbox.center_x,
                    selected_bbox.center_y,
                    selected_bbox.width,
                    selected_bbox.height,
                    selected_bbox.theta,
                )

        # clear drag/hover state
        self._hover_idx, self._hover_mode = None, None
        self._drag_mode = None
        self._press_pos_disp = None
        self._orig_box = None
        ev.accept()

    def wheelEvent(self, ev):
        ev.ignore()

    def _axis_from_mode(self, theta: float, mode: str) -> tuple[float, float]:
        """Return unit axis (ax_x, ax_y) in DISPLAY coords for the edge being resized."""
        angle_rad = -theta if self._theta_is_clockwise else theta
        if mode in ("E", "W"):
            return math.cos(angle_rad), math.sin(angle_rad)
        else:
            return -math.sin(angle_rad), math.cos(angle_rad)

    def _cursor_for_axis(self, ax_x: float, ax_y: float) -> Qt.CursorShape:
        """
        Map axis angle to the nearest standard cursor:
        0/180° → SizeHor, 90° → SizeVer, 45° → FDiag, 135° → BDiag
        """
        deg = abs(math.degrees(math.atan2(ax_y, ax_x))) % 180.0
        if deg < 22.5 or deg >= 157.5:
            return Qt.CursorShape.SizeHorCursor
        elif deg < 67.5:
            return Qt.CursorShape.SizeFDiagCursor
        elif deg < 112.5:
            return Qt.CursorShape.SizeVerCursor
        else:
            return Qt.CursorShape.SizeBDiagCursor

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

        for idx, bbox in enumerate(self._boxes):
            # Handle both BBox and tuple formats during transition
            if isinstance(bbox, BBoxData):
                cx_vid = bbox.center_x
                cy_vid = bbox.center_y
                w_vid = bbox.width
                h_vid = bbox.height
                theta = bbox.theta
            elif isinstance(bbox, tuple) and len(bbox) == 5:
                cx_vid, cy_vid, w_vid, h_vid, theta = bbox
            else:
                continue

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
            for lx, ly in [
                (-half_w, -half_h),
                (half_w, -half_h),
                (half_w, half_h),
                (-half_w, half_h),
            ]:
                rx = lx * cos_a - ly * sin_a
                ry = lx * sin_a + ly * cos_a
                corners.append(QPointF(cx_disp + rx, cy_disp + ry))

            TL, TR, BR, BL = corners

            # midpoints (DISPLAY)
            mid_top = QPointF((TL.x() + TR.x()) * 0.5, (TL.y() + TR.y()) * 0.5)
            mid_right = QPointF((TR.x() + BR.x()) * 0.5, (TR.y() + BR.y()) * 0.5)
            mid_bottom = QPointF((BL.x() + BR.x()) * 0.5, (BL.y() + BR.y()) * 0.5)
            mid_left = QPointF((TL.x() + BL.x()) * 0.5, (TL.y() + BL.y()) * 0.5)

            # side handle hits (square around midpoint)
            def in_handle(pt: QPointF) -> bool:
                return (pt.x() - half <= x <= pt.x() + half) and (
                    pt.y() - half <= y <= pt.y() + half
                )

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
                rot_cy - rot_half <= y <= rot_cy + rot_half
            ):
                return idx, "R"

            # inside test (point-in-quad) → MOVE
            def tri_area(a, b, c):
                return (b.x() - a.x()) * (c.y() - a.y()) - (b.y() - a.y()) * (
                    c.x() - a.x()
                )

            P = QPointF(x, y)
            a1 = tri_area(TL, TR, P)
            a2 = tri_area(TR, BR, P)
            a3 = tri_area(BR, BL, P)
            a4 = tri_area(BL, TL, P)
            if (a1 >= 0 and a2 >= 0 and a3 >= 0 and a4 >= 0) or (
                a1 <= 0 and a2 <= 0 and a3 <= 0 and a4 <= 0
            ):
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

    def _get_box_idx_from_obj_id(self, object_id: str) -> int | None:
        """Return the index of the box with the given object_id, or None if not found."""
        for idx, box in enumerate(self._boxes):
            if box.object_id == object_id:
                return idx
        return None

    def select_box_by_obj_id(self, object_id: str | None):
        """Externally select a bounding box by object_id (or None to clear selection)."""
        if object_id is None:
            self.clear_selection()
            return

        self._selected_idx = self._get_box_idx_from_obj_id(object_id)

        self._drag_mode = None
        self._hover_idx = None
        self._hover_mode = None
        self._press_pos_disp = None
        self._orig_box = None
        self.update()

    def select_box(self, idx: int | None):
        """Externally select a bounding box by index (or None to clear selection)."""
        if idx is not None and (0 <= idx < len(self._boxes)):
            self._selected_idx = idx
        else:
            self._selected_idx = None

        self._drag_mode = None
        self._hover_idx = None
        self._hover_mode = None
        self._press_pos_disp = None
        self._orig_box = None
        self.update()

    def paintEvent(self, _):
        if not self._boxes or self._frame_size == (0, 0):
            return

        scale, offset_x, offset_y, _ = self._compute_transform()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        for idx, bbox in enumerate(self._boxes):
            if isinstance(bbox, BBoxData):
                center_x = bbox.center_x
                center_y = bbox.center_y
                box_width = bbox.width
                box_height = bbox.height
                theta_val = bbox.theta
                object_id = bbox.object_id
                object_type = bbox.object_type
            elif isinstance(bbox, tuple) and len(bbox) == 5:
                center_x, center_y, box_width, box_height, theta_val = bbox
                object_id = "unknown"
                object_type = "unknown"
            else:
                continue

            center_x_disp = offset_x + center_x * scale
            center_y_disp = offset_y + center_y * scale
            box_width_disp = box_width * scale
            box_height_disp = box_height * scale

            angle_rad = -theta_val if self._theta_is_clockwise else theta_val
            cos_angle, sin_angle = math.cos(angle_rad), math.sin(angle_rad)
            half_w = box_width_disp / 2.0
            half_h = box_height_disp / 2.0

            # Compute corners
            corners = []
            for lx, ly in [
                (-half_w, -half_h),
                (half_w, -half_h),
                (half_w, half_h),
                (-half_w, half_h),
            ]:
                rx = lx * cos_angle - ly * sin_angle
                ry = lx * sin_angle + ly * cos_angle
                corners.append(QPointF(center_x_disp + rx, center_y_disp + ry))
            polygon = QPolygonF(corners)

            # Draw box
            painter.setPen(
                self._pen_box_selected if self._selected_idx == idx else self._pen_box
            )
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPolygon(polygon)
            TL, TR, BR, BL = corners[0], corners[1], corners[2], corners[3]

            # --- Draw rotated text with semi-transparent background ---
            mid_top = QPointF((TL.x() + TR.x()) / 2, (TL.y() + TR.y()) / 2)
            tx, ty = TR.x() - TL.x(), TR.y() - TL.y()
            edge_angle = math.degrees(math.atan2(ty, tx))
            text_margin = 10
            nx, ny = -ty / math.hypot(tx, ty), tx / math.hypot(tx, ty)
            vec_cx, vec_cy = center_x_disp - mid_top.x(), center_y_disp - mid_top.y()
            if nx * vec_cx + ny * vec_cy > 0:
                nx, ny = -nx, -ny
            text_x = mid_top.x() + nx * text_margin
            text_y = mid_top.y() + ny * text_margin
            text_str = f"{object_id} ({object_type})"

            painter.save()
            painter.translate(text_x, text_y)
            painter.rotate(edge_angle)

            # Measure text
            metrics = painter.fontMetrics()
            text_width = metrics.horizontalAdvance(text_str)
            text_height = metrics.height()
            padding = 2

            # Draw semi-transparent black background
            painter.setBrush(QColor(0, 0, 0, 160))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(
                -padding,
                -metrics.ascent() - padding,
                text_width + 2 * padding,
                text_height + 2 * padding,
            )

            # Draw text
            painter.setPen(QColor(255, 255, 0))  # bright yellow
            painter.drawText(0, 0, text_str)
            painter.restore()

            # --- Draw side handles if selected ---
            if self._selected_idx == idx:
                s = self._handle_draw_px
                half = s / 2.0
                mid_right = QPointF((TR.x() + BR.x()) / 2, (TR.y() + BR.y()) / 2)
                mid_bottom = QPointF((BL.x() + BR.x()) / 2, (BL.y() + BR.y()) / 2)
                mid_left = QPointF((TL.x() + BL.x()) / 2, (TL.y() + BL.y()) / 2)
                painter.setPen(QPen(QColor(30, 30, 30)))
                painter.setBrush(QColor(255, 255, 255))
                for pt in (mid_top, mid_right, mid_bottom, mid_left):
                    painter.drawRect(QRectF(pt.x() - half, pt.y() - half, s, s))

                # Rotate handle
                edge_len = math.hypot(tx, ty) or 1.0
                nx, ny = -ty / edge_len, tx / edge_len
                if nx * vec_cx + ny * vec_cy > 0:
                    nx, ny = -nx, -ny
                handle_offset = self._rotate_handle_offset_px
                rot_cx = mid_top.x() + nx * handle_offset
                rot_cy = mid_top.y() + ny * handle_offset
                pen_line = QPen(QColor(0, 200, 255))
                pen_line.setWidth(2)
                pen_line.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen_line)
                painter.drawLine(mid_top, QPointF(rot_cx, rot_cy))
                r = self._handle_draw_px / 2
                painter.setPen(self._pen_rotate_handle)
                painter.setBrush(self._brush_rotate_handle)
                painter.drawEllipse(QRectF(rot_cx - r, rot_cy - r, 2 * r, 2 * r))

            # --- Hover edges ---
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

            # --- Centers / axes ---
            if self._show_centers:
                painter.setPen(self._pen_center)
                painter.drawLine(
                    QPointF(center_x_disp - 4, center_y_disp),
                    QPointF(center_x_disp + 4, center_y_disp),
                )
                painter.drawLine(
                    QPointF(center_x_disp, center_y_disp - 4),
                    QPointF(center_x_disp, center_y_disp + 4),
                )
            if self._show_axes:
                painter.setPen(self._pen_axis)
                axis_x = center_x_disp + half_w * cos_angle
                axis_y = center_y_disp + half_h * sin_angle
                painter.drawLine(
                    QPointF(center_x_disp, center_y_disp), QPointF(axis_x, axis_y)
                )

        painter.end()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Delete and not (
            e.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self.deletePressed.emit()
            e.accept()
            return
        super().keyPressEvent(e)

    def _on_cascade_size_to_all(self):
        """Handle cascade apply to all frames."""
        selected_bbox = self._get_selected_bbox()

        # Emit a custom signal for cascade operations
        self.cascadeApplyAll.emit(
            selected_bbox.object_id,
            selected_bbox.width,
            selected_bbox.height,
            None # Pass none for the rotation
        )


    def _on_cascade_rotation_to_all(self):
        """Handle cascade apply to all frames."""
        selected_bbox = self._get_selected_bbox()

        # Emit a custom signal for cascade operations
        self.cascadeApplyAll.emit(
            selected_bbox.object_id,
            None, # Pass none for the width
            None, # Pass none for the height
            selected_bbox.theta,
        )

    def _on_cascade_size_to_frame_range(self):
        """Handle cascade apply to next X frames."""
        selected_bbox = self._get_selected_bbox()

        # Emit a custom signal for cascade operations
        self.cascadeApplyFrameRange.emit(
            selected_bbox.object_id,
            selected_bbox.width,
            selected_bbox.height,
            None,
        )

    def _on_cascade_rotation_to_frame_range(self):
        """Handle cascade apply to next X frames."""
        selected_bbox = self._get_selected_bbox()

        # Emit a custom signal for cascade operations
        self.cascadeApplyFrameRange.emit(
            selected_bbox.object_id,
            None,
            None,
            selected_bbox.theta,
        )

    def _on_cascade_cancel(self):
        """Handle cascade cancel."""
        # Just hide the dropdown, no action needed
        pass

    def _on_cascade_button_clicked(self):
        """Handle cascade button click - show the dropdown with options."""
        # Position dropdown near the cascade button
        button_pos = self.cascade_button.pos()
        dropdown_x = button_pos.x()
        dropdown_y = button_pos.y() + self.cascade_button.height()

        # Show the dropdown
        self.cascade_dropdown.show_at_position(dropdown_x, dropdown_y)

    def show_cascade_option(self):
        """Show the cascade button near the resized/rotated annotation."""
        # Get the position of the selected annotation
        bbox = self._get_selected_bbox()

        # Convert video coordinates to display coordinates
        scale, offset_x, offset_y, _ = self._compute_transform()
        center_x_disp = offset_x + bbox.center_x * scale
        center_y_disp = offset_y + bbox.center_y * scale

        # Position cascade button near the annotation (top-right corner)
        button_x = center_x_disp + (bbox.width * scale / 2) + 10
        button_y = center_y_disp - (bbox.height * scale / 2) - 10

        # Show the cascade button
        self.cascade_button.show_at_position(button_x, button_y)

    def _get_selected_bbox(self):
        """Return the currently selected bounding box, or None if none is selected."""
        if self._selected_idx is not None and 0 <= self._selected_idx < len(
            self._boxes
        ):
            return self._boxes[self._selected_idx]
        return None
