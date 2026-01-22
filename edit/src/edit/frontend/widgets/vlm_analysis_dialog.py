"""Dialog for viewing VLM analysis data (contexts and tags)."""

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QStyle,
    QVBoxLayout,
    QWidget,
)


class VLMAnalysisDialog(QDialog):
    """Modal dialog displaying VLM contexts and tags."""

    def __init__(
        self,
        contexts: dict[str, dict[str, Any]] | None,
        tags: dict[str, dict[str, Any]] | None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("VLM Analysis")
        self.setModal(True)
        self.resize(550, 450)

        self._contexts = contexts or {}
        self._tags = tags or {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        self.setLayout(main_layout)

        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(12)
        content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._add_tags_section(content_layout)
        self._add_contexts_section(content_layout)

        content_layout.addStretch()
        scroll.setWidget(content)
        main_layout.addWidget(scroll, stretch=1)

        # OK button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        ok_button = QPushButton("OK")
        ok_button.setFixedWidth(100)
        ok_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
        )
        ok_button.clicked.connect(self.accept)
        button_layout.addWidget(ok_button)
        main_layout.addLayout(button_layout)

    def _add_section_header(self, layout: QVBoxLayout, text: str) -> None:
        header = QLabel(text)
        font = QFont()
        font.setBold(True)
        font.setPointSize(11)
        header.setFont(font)
        layout.addWidget(header)

    def _add_tags_section(self, layout: QVBoxLayout) -> None:
        self._add_section_header(layout, "Video-Level Tags")
        if not self._tags:
            layout.addWidget(self._no_data_label("No VLM tags available"))
            return

        for tag_id, tag in self._tags.items():
            tag_type = tag.get("type", "Unknown")
            tag_data = tag.get("tag_data", {})
            layout.addWidget(
                self._create_data_widget(tag_type.replace("Tag", ""), tag_data)
            )

    def _add_contexts_section(self, layout: QVBoxLayout) -> None:
        self._add_section_header(layout, "Frame-Bound Contexts")
        if not self._contexts:
            layout.addWidget(self._no_data_label("No VLM contexts available"))
            return

        for ctx_id, ctx in self._contexts.items():
            ctx_type = ctx.get("type", "Unknown")
            intervals = ctx.get("frame_intervals", [])
            ctx_data = ctx.get("context_data", {})

            # Add frame interval info to display
            interval_str = ", ".join(
                f"{iv.get('frame_start', 0)}-{iv.get('frame_end', 0)}"
                for iv in intervals
            )
            layout.addWidget(
                self._create_data_widget(
                    f"{ctx_type.replace('Context', '')} (frames {interval_str})",
                    ctx_data,
                )
            )

    def _no_data_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("color: #888; font-style: italic; margin-left: 8px;")
        return label

    def _create_data_widget(self, title: str, data: dict) -> QWidget:
        widget = QWidget()
        widget.setStyleSheet(
            "background-color: rgba(128,128,128,0.1); border-radius: 4px;"
        )
        vlayout = QVBoxLayout(widget)
        vlayout.setContentsMargins(10, 6, 10, 6)
        vlayout.setSpacing(2)

        title_label = QLabel(f"<b>{title}</b>")
        vlayout.addWidget(title_label)

        for item in data.get("text", []):
            name = item.get("name", "")
            val = item.get("val") or ""
            vlayout.addWidget(QLabel(f"  {name}: {val}"))

        for item in data.get("num", []):
            val = item.get("val", 0)
            val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
            vlayout.addWidget(QLabel(f"  {item.get('name', '')}: {val_str}"))

        for item in data.get("boolean", []):
            val = "Yes" if item.get("val", False) else "No"
            vlayout.addWidget(QLabel(f"  {item.get('name', '')}: {val}"))

        return widget
