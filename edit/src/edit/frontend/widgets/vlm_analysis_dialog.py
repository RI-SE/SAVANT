"""Dialog for viewing and editing VLM analysis data (contexts and tags)."""

from copy import deepcopy
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStyle,
    QVBoxLayout,
    QWidget,
)


class VLMAnalysisDialog(QDialog):
    """Modal dialog displaying VLM contexts and tags with editing support."""

    def __init__(
        self,
        contexts: dict[str, dict[str, Any]] | None,
        tags: dict[str, dict[str, Any]] | None,
        parent=None,
        *,
        frontend_state=None,
        undo_manager=None,
        undo_context=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("VLM Analysis")
        self.setModal(True)
        self.resize(550, 500)

        self._contexts = contexts or {}
        self._tags = tags or {}
        self._frontend_state = frontend_state
        self._undo_manager = undo_manager
        self._undo_context = undo_context

        # Track editing state for each tag
        self._tag_editors: dict[str, dict] = {}
        self._tag_widgets: dict[str, QWidget] = {}

        self._setup_ui()

    def _can_edit(self) -> bool:
        """Check if editing is supported (all required components available)."""
        return all([
            self._frontend_state is not None,
            self._undo_manager is not None,
            self._undo_context is not None,
            self._undo_context.vlm_gateway is not None if self._undo_context else False,
        ])

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        self.setLayout(main_layout)

        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        content = QWidget()
        self._content_layout = QVBoxLayout(content)
        self._content_layout.setSpacing(12)
        self._content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._add_tags_section(self._content_layout)
        self._add_contexts_section(self._content_layout)

        self._content_layout.addStretch()
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
            widget = self._create_tag_widget(
                tag_id, tag_type.replace("Tag", ""), tag_data
            )
            self._tag_widgets[tag_id] = widget
            layout.addWidget(widget)

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

    def _create_tag_widget(
        self, tag_id: str, title: str, data: dict
    ) -> QWidget:
        """Create a widget for a tag that supports editing."""
        widget = QWidget()
        widget.setStyleSheet(
            "background-color: rgba(128,128,128,0.1); border-radius: 4px;"
        )
        vlayout = QVBoxLayout(widget)
        vlayout.setContentsMargins(10, 6, 10, 6)
        vlayout.setSpacing(2)

        # Title row with edit button
        title_row = QHBoxLayout()
        title_label = QLabel(f"<b>{title}</b>")
        title_row.addWidget(title_label)
        title_row.addStretch()

        if self._can_edit():
            edit_btn = QPushButton("Edit")
            edit_btn.setFixedWidth(60)
            edit_btn.clicked.connect(
                lambda _, tid=tag_id: self._start_editing_tag(tid)
            )
            title_row.addWidget(edit_btn)

        vlayout.addLayout(title_row)

        # Data display (will be replaced during editing)
        data_container = QWidget()
        data_layout = QVBoxLayout(data_container)
        data_layout.setContentsMargins(0, 0, 0, 0)
        data_layout.setSpacing(2)

        self._populate_data_display(data_layout, data)

        vlayout.addWidget(data_container)

        # Store references for editing
        self._tag_editors[tag_id] = {
            "widget": widget,
            "data_container": data_container,
            "data": data,
            "is_editing": False,
        }

        return widget

    def _populate_data_display(self, layout: QVBoxLayout, data: dict) -> None:
        """Populate a layout with read-only data display."""
        for item in data.get("text", []):
            name = item.get("name", "")
            val = item.get("val") or ""
            layout.addWidget(QLabel(f"  {name}: {val}"))

        for item in data.get("num", []):
            val = item.get("val", 0)
            val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
            layout.addWidget(QLabel(f"  {item.get('name', '')}: {val_str}"))

        for item in data.get("boolean", []):
            val = "Yes" if item.get("val", False) else "No"
            layout.addWidget(QLabel(f"  {item.get('name', '')}: {val}"))

        for item in data.get("vec", []):
            name = item.get("name", "")
            vals = item.get("val", [])
            val_str = ", ".join(self._format_vec_value(v) for v in vals)
            layout.addWidget(QLabel(f"  {name}: {val_str}"))

    def _start_editing_tag(self, tag_id: str) -> None:
        """Switch a tag widget to editing mode."""
        if tag_id not in self._tag_editors:
            return

        editor_info = self._tag_editors[tag_id]
        if editor_info["is_editing"]:
            return

        editor_info["is_editing"] = True
        data = editor_info["data"]
        data_container = editor_info["data_container"]

        # Clear current display
        old_layout = data_container.layout()
        while old_layout.count():
            item = old_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Create editors for editable fields
        editors = {}

        # Text fields (except annotator which is auto-managed)
        for item in data.get("text", []):
            name = item.get("name", "")
            if name == "annotator":
                continue  # Skip annotator - it's managed by the system
            val = item.get("val") or ""

            row = QHBoxLayout()
            row.addWidget(QLabel(f"  {name}:"))
            editor = QLineEdit(str(val))
            editor.setMinimumWidth(150)
            editors[("text", name)] = editor
            row.addWidget(editor)
            row.addStretch()

            row_widget = QWidget()
            row_widget.setLayout(row)
            old_layout.addWidget(row_widget)

        # Num fields (except confidence which is auto-managed)
        for item in data.get("num", []):
            name = item.get("name", "")
            if name == "confidence":
                continue  # Skip confidence - it's managed by the system
            val = item.get("val", 0)

            row = QHBoxLayout()
            row.addWidget(QLabel(f"  {name}:"))
            editor = QLineEdit(str(val))
            editor.setMinimumWidth(80)
            editors[("num", name)] = editor
            row.addWidget(editor)
            row.addStretch()

            row_widget = QWidget()
            row_widget.setLayout(row)
            old_layout.addWidget(row_widget)

        # Boolean fields
        for item in data.get("boolean", []):
            name = item.get("name", "")
            val = item.get("val", False)

            row = QHBoxLayout()
            row.addWidget(QLabel(f"  {name}:"))
            editor = QCheckBox()
            editor.setChecked(bool(val))
            editors[("boolean", name)] = editor
            row.addWidget(editor)
            row.addStretch()

            row_widget = QWidget()
            row_widget.setLayout(row)
            old_layout.addWidget(row_widget)

        # Vec fields (read-only display)
        for item in data.get("vec", []):
            name = item.get("name", "")
            vals = item.get("val", [])
            val_str = ", ".join(self._format_vec_value(v) for v in vals)
            old_layout.addWidget(QLabel(f"  {name}: {val_str} (auto-managed)"))

        # Save/Cancel buttons
        button_row = QHBoxLayout()
        button_row.addStretch()

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(lambda: self._save_tag_edit(tag_id, editors))
        button_row.addWidget(save_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(lambda: self._cancel_tag_edit(tag_id))
        button_row.addWidget(cancel_btn)

        button_widget = QWidget()
        button_widget.setLayout(button_row)
        old_layout.addWidget(button_widget)

    def _save_tag_edit(self, tag_id: str, editors: dict) -> None:
        """Save the edited tag data with annotator prepending."""
        if tag_id not in self._tag_editors:
            return

        # Get current annotator
        annotator = self._frontend_state.require_current_annotator()
        if not annotator:
            QMessageBox.warning(
                self,
                "Annotator Required",
                "Please set an annotator name to edit VLM tags.",
            )
            return

        editor_info = self._tag_editors[tag_id]
        old_data = editor_info["data"]
        new_data = deepcopy(old_data)

        # Update text fields
        for item in new_data.get("text", []):
            name = item.get("name", "")
            if name == "annotator":
                continue
            key = ("text", name)
            if key in editors:
                item["val"] = editors[key].text()

        # Update num fields
        for item in new_data.get("num", []):
            name = item.get("name", "")
            if name == "confidence":
                continue
            key = ("num", name)
            if key in editors:
                try:
                    item["val"] = float(editors[key].text())
                except ValueError:
                    pass  # Keep old value if parse fails

        # Update boolean fields
        for item in new_data.get("boolean", []):
            name = item.get("name", "")
            key = ("boolean", name)
            if key in editors:
                item["val"] = editors[key].isChecked()

        # Prepend annotator and confidence to vec fields
        for item in new_data.get("vec", []):
            name = item.get("name", "")
            if name == "annotator":
                if isinstance(item.get("val"), list):
                    item["val"].insert(0, annotator)
                else:
                    item["val"] = [annotator]
            elif name == "confidence":
                if isinstance(item.get("val"), list):
                    item["val"].insert(0, 1.0)  # Human confidence is always 1.0
                else:
                    item["val"] = [1.0]

        # Create and execute undo command
        from edit.frontend.utils.undo.commands import UpdateVLMTagCommand
        from edit.frontend.utils.undo.snapshots import VLMTagSnapshot

        before = VLMTagSnapshot(tag_id=tag_id, tag_data=deepcopy(old_data))
        after = VLMTagSnapshot(tag_id=tag_id, tag_data=new_data)

        command = UpdateVLMTagCommand(
            tag_id=tag_id,
            before=before,
            after=after,
        )
        self._undo_manager.execute(command, self._undo_context)

        # Update local state and refresh display
        editor_info["data"] = new_data
        self._tags[tag_id]["tag_data"] = new_data
        editor_info["is_editing"] = False
        self._refresh_tag_display(tag_id)

    def _cancel_tag_edit(self, tag_id: str) -> None:
        """Cancel editing and restore the read-only display."""
        if tag_id not in self._tag_editors:
            return

        editor_info = self._tag_editors[tag_id]
        editor_info["is_editing"] = False
        self._refresh_tag_display(tag_id)

    def _refresh_tag_display(self, tag_id: str) -> None:
        """Refresh the display of a tag after editing."""
        if tag_id not in self._tag_editors:
            return

        editor_info = self._tag_editors[tag_id]
        data = editor_info["data"]
        data_container = editor_info["data_container"]

        # Clear current display
        old_layout = data_container.layout()
        while old_layout.count():
            item = old_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Repopulate with read-only display
        self._populate_data_display(old_layout, data)

    def _create_data_widget(self, title: str, data: dict) -> QWidget:
        """Create a read-only widget for context data (contexts are not editable)."""
        widget = QWidget()
        widget.setStyleSheet(
            "background-color: rgba(128,128,128,0.1); border-radius: 4px;"
        )
        vlayout = QVBoxLayout(widget)
        vlayout.setContentsMargins(10, 6, 10, 6)
        vlayout.setSpacing(2)

        title_label = QLabel(f"<b>{title}</b>")
        vlayout.addWidget(title_label)

        self._populate_data_display(vlayout, data)

        return widget

    def _format_vec_value(self, val: Any) -> str:
        """Format a single value in a vec list for display."""
        if isinstance(val, float):
            return f"{val:.2f}"
        return str(val)
