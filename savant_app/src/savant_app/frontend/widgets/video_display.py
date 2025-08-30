from PyQt6.QtWidgets import QLabel, QSizePolicy
from PyQt6.QtGui import QPixmap, QPainter, QPen
from PyQt6.QtCore import Qt, QPointF, pyqtSignal, QRectF


class VideoDisplay(QLabel):
    bbox_drawn = pyqtSignal(dict)  # Emits {"coordinates": (x1,y1,x2,y2)}

    def __init__(self):
        super().__init__()
        self.setPixmap(QPixmap())
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background-color: black; border: 1px solid #444;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setScaledContents(True)
        
        # Drawing state
        self.drawing = False
        self.start_point = QPointF()
        self.end_point = QPointF()
        self.current_object_type = ""

    def show_frame(self, pixmap: QPixmap) -> None:
        if pixmap and not pixmap.isNull():
            self.setPixmap(pixmap)

    def start_drawing_mode(self, object_type: str):
        """Enable bounding box drawing mode for specific object type."""
        self.current_object_type = object_type
        self.setCursor(Qt.CursorShape.CrossCursor)

    def mousePressEvent(self, event):
        if self.current_object_type and event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.start_point = event.position()
            self.end_point = event.position()
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_point = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drawing:
            self.drawing = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            # Emit normalized coordinates (0-1 relative to video size)
            rect = self.contentsRect()
            x1 = min(max(self.start_point.x() / rect.width(), 0.0), 1.0)
            y1 = min(max(self.start_point.y() / rect.height(), 0.0), 1.0)
            x2 = min(max(self.end_point.x() / rect.width(), 0.0), 1.0)
            y2 = min(max(self.end_point.y() / rect.height(), 0.0), 1.0)
            
            self.bbox_drawn.emit({
                "type": self.current_object_type,
                "coordinates": (x1, y1, x2, y2)
            })
            self.current_object_type = ""
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.drawing:
            painter = QPainter(self)
            painter.setPen(QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.SolidLine))
            rect = QRectF(self.start_point, self.end_point)
            painter.drawRect(rect)
