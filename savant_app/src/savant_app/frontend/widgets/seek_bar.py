# frontend/widgets/seek_bar.py
from PyQt6.QtWidgets import (
    QWidget,
    QSlider,
    QHBoxLayout,
    QLabel,
    QStyle,
    QStyleOptionSlider,
)
from PyQt6.QtGui import QPainter, QColor
from PyQt6.QtCore import Qt, pyqtSignal, QRectF
from savant_app.frontend.theme.constants import (
    SEEK_BAR_MARKER_THICKNESS,
    SEEK_BAR_WARNING_MARKER_COLOR,
    SEEK_BAR_ERROR_MARKER_COLOR,
)


class SeekSlider(QSlider):
    """QSlider that jumps directly to the mouse click position."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._warning_frames: list[int] = []
        self._error_frames: list[int] = []
        self._show_warnings: bool = True
        self._show_errors: bool = True

    def mousePressEvent(self, event):
        """
        Override the default mouse press behavior.

        By default, clicking on the slider groove moves by the pageStep (e.g. 10).
        This override maps the click position directly to the corresponding slider
        value, so clicking anywhere instantly seeks to that frame.

        Args:
            event (QMouseEvent): Mouse click event.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            option = QStyleOptionSlider()
            self.initStyleOption(option)
            groove = self.style().subControlRect(
                QStyle.ComplexControl.CC_Slider,
                option,
                QStyle.SubControl.SC_SliderGroove,
                self,
            )
            handle = self.style().subControlRect(
                QStyle.ComplexControl.CC_Slider,
                option,
                QStyle.SubControl.SC_SliderHandle,
                self,
            )
            if self.orientation() == Qt.Orientation.Horizontal:
                click_x = event.position().x()
                handle_w = handle.width()
                available = max(0, groove.width() - handle_w)
                pos = int(click_x - groove.x() - handle_w / 2)
                pos = max(0, min(pos, available))
                value = QStyle.sliderValueFromPosition(
                    self.minimum(),
                    self.maximum(),
                    pos,
                    available if available > 0 else 1,
                )
                self.setValue(value)
                event.accept()
                return
            else:
                click_y = event.position().y()
                handle_h = handle.height()
                available = max(0, groove.height() - handle_h)
                pos = int(click_y - groove.y() - handle_h / 2)
                pos = max(0, min(pos, available))
                value = QStyle.sliderValueFromPosition(
                    self.minimum(),
                    self.maximum(),
                    pos,
                    available if available > 0 else 1,
                )
                self.setValue(value)
                event.accept()
                return
        super().mousePressEvent(event)

    def set_warning_frames(self, frames):
        self._warning_frames = self._normalize_frames(frames)
        self.update()

    def set_error_frames(self, frames):
        self._error_frames = self._normalize_frames(frames)
        self.update()

    def warning_frames(self) -> list[int]:
        return list(self._warning_frames)

    def error_frames(self) -> list[int]:
        return list(self._error_frames)

    def set_show_warnings(self, show: bool) -> None:
        if self._show_warnings != bool(show):
            self._show_warnings = bool(show)
            self.update()

    def set_show_errors(self, show: bool) -> None:
        if self._show_errors != bool(show):
            self._show_errors = bool(show)
            self.update()

    def show_warnings(self) -> bool:
        return bool(self._show_warnings)

    def show_errors(self) -> bool:
        return bool(self._show_errors)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.maximum() < self.minimum():
            return

        option = QStyleOptionSlider()
        self.initStyleOption(option)
        groove = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderGroove,
            self,
        )
        if groove.width() <= 0 or groove.height() <= 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        span = max(1, self.maximum() - self.minimum())
        marker_width = max(1, SEEK_BAR_MARKER_THICKNESS)

        handle = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderHandle,
            self,
        )

        def draw_markers(frames: list[int], color: QColor):
            if not frames:
                return
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            for frame_idx in frames:
                if frame_idx < self.minimum() or frame_idx > self.maximum():
                    continue
                if self.orientation() == Qt.Orientation.Horizontal:
                    available = groove.width() - handle.width()
                    if available > 0:
                        slider_pos = QStyle.sliderPositionFromValue(
                            self.minimum(), self.maximum(), frame_idx, available
                        )
                        position_x = groove.left() + handle.width() / 2 + slider_pos
                    else:
                        ratio = (frame_idx - self.minimum()) / span
                        position_x = groove.left() + ratio * groove.width()
                    rect = QRectF(
                        position_x - marker_width / 2,
                        groove.top(),
                        marker_width,
                        groove.height(),
                    )
                else:
                    available = groove.height() - handle.height()
                    if available > 0:
                        slider_pos = QStyle.sliderPositionFromValue(
                            self.minimum(), self.maximum(), frame_idx, available
                        )
                        position_y = groove.top() + handle.height() / 2 + slider_pos
                    else:
                        ratio = (frame_idx - self.minimum()) / span
                        position_y = groove.top() + ratio * groove.height()
                    rect = QRectF(
                        groove.left(),
                        position_y - marker_width / 2,
                        groove.width(),
                        marker_width,
                    )
                painter.fillRect(rect, color)

        if self._show_warnings:
            draw_markers(self._warning_frames, SEEK_BAR_WARNING_MARKER_COLOR)
        if self._show_errors:
            draw_markers(self._error_frames, SEEK_BAR_ERROR_MARKER_COLOR)
        painter.end()

    def _normalize_frames(self, frames) -> list[int]:
        if frames is None:
            return []
        minimum = self.minimum()
        maximum = self.maximum()
        if maximum < minimum:
            return []
        normalized = sorted(
            {
                int(frame)
                for frame in frames
                if isinstance(frame, (int, float)) and minimum <= int(frame) <= maximum
            }
        )
        return normalized


class SeekBar(QWidget):
    """
    Frame seek bar composed of a SeekSlider and a label.

    Emits:
        frame_changed (int): Emitted whenever the slider position changes,
                             carrying the new frame index.
    """

    frame_changed = pyqtSignal(int)

    def __init__(self, frame_count: int = 0):
        """
        Initialize the seek bar.

        Args:
            frame_count (int, optional): Total number of frames in the video.
                                         Used to set the slider range. Defaults to 0.
        """
        super().__init__()

        self.slider = SeekSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(0, frame_count - 1))
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        self.slider.setTracking(False)

        self.label = QLabel("0 / 0")

        layout = QHBoxLayout(self)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.label)

        self.slider.valueChanged.connect(self._on_value_changed)
        self._warning_frames_raw: list[int] = []
        self._error_frames_raw: list[int] = []
        self._warning_frames: list[int] = []
        self._error_frames: list[int] = []
        self._show_warnings: bool = True
        self._show_errors: bool = True

    def _on_value_changed(self, value: int):
        """
        Slot called when the slider value changes.

        Updates the label with the new index and emits frame_changed.

        Args:
            value (int): New frame index selected by the user.
        """
        self.label.setText(f"{value} / {self.slider.maximum()}")
        self.frame_changed.emit(value)

    def update_range(self, frame_count: int):
        """
        Update the seek bar range when a new video is loaded.

        Resets the slider to 0 and adjusts the label.

        Args:
            frame_count (int): Total frames in the new video.
        """
        self.slider.blockSignals(True)
        self.slider.setMaximum(max(0, frame_count - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.label.setText(f"0 / {frame_count}")
        self._apply_marker_frames()

    def set_position(self, index: int):
        """
        Set the current position of the slider programmatically.

        Typically called while playing video or when stepping frames,
        to keep the slider synced with the actual playback state.

        Args:
            index (int): Frame index to move the slider to.
        """
        self.slider.blockSignals(True)
        self.slider.setValue(index)
        self.slider.blockSignals(False)
        self.label.setText(f"{index} / {self.slider.maximum()}")

    def set_index(self, value: int, emit_signal: bool = True):
        """
        Set the current frame on the slider. Optionally emit frame_changed.
        Use this for programmatic frame updates (playback, arrow keys).
        """
        self.slider.setValue(int(value))
        if emit_signal:
            self.frame_changed.emit(int(value))

    def set_confidence_markers(
        self, warning_frames: list[int] | None, error_frames: list[int] | None
    ):
        self.set_warning_frames(warning_frames or [])
        self.set_error_frames(error_frames or [])

    def set_warning_frames(self, frames: list[int]):
        self._warning_frames_raw = self._normalize_source_frames(frames)
        self._apply_marker_frames()

    def set_error_frames(self, frames: list[int]):
        self._error_frames_raw = self._normalize_source_frames(frames)
        self._apply_marker_frames()

    def warning_frames(self) -> list[int]:
        return list(self._warning_frames)

    def error_frames(self) -> list[int]:
        return list(self._error_frames)

    def set_warning_visibility(self, show: bool):
        self._show_warnings = bool(show)
        self.slider.set_show_warnings(show)

    def set_error_visibility(self, show: bool):
        self._show_errors = bool(show)
        self.slider.set_show_errors(show)

    def warning_visibility(self) -> bool:
        return bool(self._show_warnings)

    def error_visibility(self) -> bool:
        return bool(self._show_errors)

    def _apply_marker_frames(self):
        filtered_warnings = self._filter_frames_for_range(self._warning_frames_raw)
        filtered_errors = self._filter_frames_for_range(self._error_frames_raw)
        self.slider.set_warning_frames(filtered_warnings)
        self.slider.set_error_frames(filtered_errors)
        self._warning_frames = self.slider.warning_frames()
        self._error_frames = self.slider.error_frames()

    def _filter_frames_for_range(self, frames: list[int]) -> list[int]:
        if not frames:
            return []
        minimum = self.slider.minimum()
        maximum = self.slider.maximum()
        if maximum < minimum:
            return []
        return [frame for frame in frames if minimum <= frame <= maximum]

    def _normalize_source_frames(self, frames) -> list[int]:
        if frames is None:
            return []
        return sorted(
            {
                int(frame)
                for frame in frames
                if isinstance(frame, (int, float)) and int(frame) >= 0
            }
        )
