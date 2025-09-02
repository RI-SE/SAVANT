from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QPen, QColor, QPolygonF
from PyQt6.QtCore import Qt, QPointF
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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
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

        for center_x_vid, center_y_vid, width_vid, height_vid, theta in self._boxes:
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
                (-half_w, -half_h),
                (half_w, -half_h),
                (half_w, half_h),
                (-half_w, half_h),
            ]:

                rotated_offset_x = (
                    local_offset_x * cos_angle - local_offset_y * sin_angle
                )
                rotated_offset_y = (
                    local_offset_x * sin_angle + local_offset_y * cos_angle
                )

                corner_x_disp = center_x_disp + rotated_offset_x
                corner_y_disp = center_y_disp + rotated_offset_y
                corners.append(QPointF(corner_x_disp, corner_y_disp))

            painter.setPen(self._pen_box)
            painter.drawPolygon(QPolygonF(corners))

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
