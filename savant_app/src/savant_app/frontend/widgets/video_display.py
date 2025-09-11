# frontend/widgets/video_display.py
from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QPainter, QMouseEvent, QCursor, QPixmap, QPen
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal
from typing import Optional
from savant_app.frontend.states.annotation_state import AnnotationState


class VideoDisplay(QLabel):
    pan_changed = pyqtSignal(float, float)
    bbox_drawn = pyqtSignal(AnnotationState)

    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: black; border: 1px solid #444;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pixmap = None
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._dragging = False
        self._drag_start_pos = QPointF()
        self._pan_start = QPointF()

        # Drawing state
        self.drawing = False
        self.start_point = QPointF()
        self.end_point = QPointF()
        self.current_annotation_state: Optional[AnnotationState] = None

        self._pixmap = None
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._dragging = False
        self._drag_start_pos = QPointF()
        self._pan_start = QPointF()

    def start_drawing_mode(self, annotation_state: AnnotationState):
        """Enable bounding box drawing mode for specific object type."""
        self.current_annotation_state = annotation_state 
        self.setCursor(Qt.CursorShape.CrossCursor)


    def mousePressEvent(self, e: QMouseEvent):
        """Unified mouse press handler for both drawing and panning."""
        if self.current_annotation_state and e.button() == Qt.MouseButton.LeftButton:
            self._handle_drawing_press(e)
        elif (
            not self.drawing
            and self._zoom > 1.0
            and e.button() == Qt.MouseButton.LeftButton
        ):
            self._handle_panning_press(e)
        else:
            super().mousePressEvent(e)

    def mouseMoveEvent(self, e: QMouseEvent):
        """Unified mouse move handler for both drawing and panning."""
        if self.drawing:
            self._handle_drawing_move(e)
        elif self._dragging:
            self._handle_panning_move(e)
        else:
            super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e: QMouseEvent):
        """Unified mouse release handler for both drawing and panning."""
        if self.drawing and e.button() == Qt.MouseButton.LeftButton:
            self._handle_drawing_release(e)
        elif self._dragging and e.button() == Qt.MouseButton.LeftButton:
            self._handle_panning_release(e)
        else:
            super().mouseReleaseEvent(e)

    def _handle_drawing_press(self, e: QMouseEvent):
        """Handle drawing mode mouse press."""
        self.drawing = True
        self.start_point = e.position()
        self.end_point = e.position()
        self.update()

    def _handle_drawing_move(self, e: QMouseEvent):
        """Handle drawing mode mouse move."""
        if self.drawing:
            self.end_point = e.position()
            self.update()

    def _handle_drawing_release(self, e: QMouseEvent):
        """Handle drawing mode mouse release."""
        self.drawing = False
        self.setCursor(Qt.CursorShape.ArrowCursor)

        if self._pixmap is None or self._pixmap.isNull():
            # Fallback to current behavior if no pixmap available
            scale_factor = 1.0  # noqa: F841
        else:
            base_scale = self._fit_scale()
            scale_factor = base_scale * self._zoom  # noqa: F841

        rect = self.contentsRect()  # noqa: F841
        target_rect = self._draw_rect()

        # Calculate scaling factors for coordinate conversion
        width_ratio = (
            self._pixmap.width() / target_rect.width() if target_rect.width() > 0 else 1
        )
        height_ratio = (
            self._pixmap.height() / target_rect.height()
            if target_rect.height() > 0
            else 1
        )

        # Convert display coordinates to absolute pixel positions
        # Adjust for image position within widget
        x1_abs = (self.start_point.x() - target_rect.left()) * width_ratio
        y1_abs = (self.start_point.y() - target_rect.top()) * height_ratio
        x2_abs = (self.end_point.x() - target_rect.left()) * width_ratio
        y2_abs = (self.end_point.y() - target_rect.top()) * height_ratio

        # Convert to center/width/height format (YOLO OBB format)
        center_x = (x1_abs + x2_abs) / 2
        center_y = (y1_abs + y2_abs) / 2
        width = abs(x2_abs - x1_abs)
        height = abs(y2_abs - y1_abs)

        # Rotation is currently 0 as per specification
        rotation = 0.0

        self.current_annotation_state.coordinates = (center_x, center_y, width, height, rotation)

        self.bbox_drawn.emit(
            self.current_annotation_state
        )
        self.current_annotation_state = None
        self.update()

    def _handle_panning_press(self, e: QMouseEvent):
        """Handle panning mode mouse press."""
        self._dragging = True
        self._drag_start_pos = e.position()
        self._pan_start = QPointF(self._pan)
        self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
        e.accept()

    def _handle_panning_move(self, e: QMouseEvent):
        """Handle panning mode mouse move."""
        if self._dragging:
            delta = e.position() - self._drag_start_pos
            self._pan = QPointF(
                self._pan_start.x() + delta.x(), self._pan_start.y() + delta.y()
            )
            self._clamp_pan()
            self.update()
            self.pan_changed.emit(self._pan.x(), self._pan.y())
            e.accept()

    def _handle_panning_release(self, e: QMouseEvent):
        """Handle panning mode mouse release."""
        if self._dragging and e.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            e.accept()

    def set_zoom(self, zoom: float) -> None:
        self._zoom = max(0.05, min(zoom, 20.0))
        self._clamp_pan()
        self.update()
        self.pan_changed.emit(self._pan.x(), self._pan.y())

    def zoom(self) -> float:
        return self._zoom

    def set_pan(self, pan_x: float, pan_y: float) -> None:
        self._pan = QPointF(pan_x, pan_y)
        self._clamp_pan()
        self.update()
        self.pan_changed.emit(self._pan.x(), self._pan.y())

    def show_frame(self, pixmap: QPixmap) -> None:
        """Show a new frame in the video display."""
        if pixmap and not pixmap.isNull():
            self._pixmap = pixmap
            self._clamp_pan()
            self.update()
            self.pan_changed.emit(self._pan.x(), self._pan.y())

    def refresh_frame(self) -> None:
        """Refresh the current frame in the video display"""
        if self._pixmap and not self._pixmap.isNull():
            self._clamp_pan()
            self.update()
            self.pan_changed.emit(self._pan.x(), self._pan.y())

    def _fit_scale(self) -> float:
        if not self._pixmap or self._pixmap.isNull():
            return 1.0
        fw, fh = self._pixmap.width(), self._pixmap.height()
        vw, vh = self.width(), self.height()
        return min(vw / fw, vh / fh)

    def _draw_rect(self) -> QRectF:
        if not self._pixmap or self._pixmap.isNull():
            return QRectF()
        fw, fh = self._pixmap.width(), self._pixmap.height()
        vw, vh = self.width(), self.height()
        base = self._fit_scale()
        scale = base * self._zoom
        draw_w, draw_h = fw * scale, fh * scale
        off_x = (vw - draw_w) / 2 + self._pan.x()
        off_y = (vh - draw_h) / 2 + self._pan.y()
        return QRectF(off_x, off_y, draw_w, draw_h)

    def _clamp_pan(self) -> None:
        """Keep image from drifting too far off-screen."""
        if not self._pixmap or self._pixmap.isNull():
            self._pan = QPointF(0, 0)
            return
        fw, fh = self._pixmap.width(), self._pixmap.height()
        vw, vh = self.width(), self.height()
        base = self._fit_scale()
        scale = base * self._zoom
        draw_w, draw_h = fw * scale, fh * scale

        max_x = max(0.0, (draw_w - vw) / 2)
        max_y = max(0.0, (draw_h - vh) / 2)
        x = max(-max_x, min(self._pan.x(), max_x))
        y = max(-max_y, min(self._pan.y(), max_y))
        self._pan = QPointF(x, y)

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.GlobalColor.black)

        if self._pixmap and not self._pixmap.isNull():
            target = self._draw_rect()
            source = QRectF(0, 0, self._pixmap.width(), self._pixmap.height())
            p.drawPixmap(target, self._pixmap, source)

        # Draw bounding box if in drawing mode
        if self.drawing:
            p.setPen(QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.SolidLine))
            rect = QRectF(self.start_point, self.end_point)
            p.drawRect(rect)

        p.end()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._clamp_pan()
        self.pan_changed.emit(self._pan.x(), self._pan.y())
