# frontend/widgets/seek_bar.py
from PyQt6.QtWidgets import QWidget, QSlider, QHBoxLayout, QLabel, QStyle, QStyleOptionSlider
from PyQt6.QtCore import Qt, pyqtSignal


class SeekSlider(QSlider):
    """QSlider that jumps directly to the mouse click position."""

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
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)
            groove = self.style().subControlRect(QStyle.ComplexControl.CC_Slider, opt,
                                                 QStyle.SubControl.SC_SliderGroove, self)
            handle = self.style().subControlRect(QStyle.ComplexControl.CC_Slider, opt,
                                                 QStyle.SubControl.SC_SliderHandle, self)
            if self.orientation() == Qt.Orientation.Horizontal:
                groove_w = groove.width()
                click_x = event.position().x()
                handle_w = handle.width()
                pos = int(click_x - groove.x() - handle_w / 2)
                pos = max(0, min(pos, groove_w - 1))
                value = QStyle.sliderValueFromPosition(self.minimum(),
                                                       self.maximum(), pos, groove_w - 1)
                self.setValue(value)
                event.accept()
                return
            else:
                groove_h = groove.height()
                click_y = event.position().y()
                handle_h = handle.height()
                pos = int(click_y - groove.y() - handle_h / 2)
                pos = max(0, min(pos, groove_h - 1))
                value = QStyle.sliderValueFromPosition(self.minimum(),
                                                       self.maximum(), pos, groove_h - 1)
                self.setValue(value)
                event.accept()
                return
        super().mousePressEvent(event)


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
