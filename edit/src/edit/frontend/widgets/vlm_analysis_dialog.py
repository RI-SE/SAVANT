"""Dialog for viewing and editing VLM analysis data (contexts and tags)."""

from copy import deepcopy
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QStyle,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# Fields that support rationales, organized by tag type
# Format: tag_type -> list of field names
RATIONALE_FIELDS = {
    "Weather": [
        "precipitation",
        "precipitation_intensity",
        "particulates",
        "time_of_day",
        "sun_position",
        "cloud_cover",
    ],
    "Traffic": [
        "density",
        "flow",
        "temporary_structures",
    ],
    "Road": [
        "drivable_area_type",
        "surface_type",
        "surface_condition",
        "surface_quality",
    ],
}


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
        self.resize(600, 550)

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
                    tag_type=ctx_type.replace("Context", ""),
                )
            )

    def _no_data_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("color: #888; font-style: italic; margin-left: 8px;")
        return label

    def _get_rationale_fields_for_tag(self, tag_type: str) -> list[str]:
        """Get the list of fields that support rationales for a tag type."""
        return RATIONALE_FIELDS.get(tag_type, [])

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

        self._populate_data_display(data_layout, data, tag_type=title)

        vlayout.addWidget(data_container)

        # Store references for editing
        self._tag_editors[tag_id] = {
            "widget": widget,
            "data_container": data_container,
            "data": data,
            "tag_type": title,
            "is_editing": False,
        }

        return widget

    def _get_field_value(self, data: dict, field_name: str) -> Any:
        """Get a field value from tag data by searching text, num, boolean lists."""
        for item in data.get("text", []):
            if item.get("name") == field_name:
                return item.get("val")
        for item in data.get("num", []):
            if item.get("name") == field_name:
                return item.get("val")
        for item in data.get("boolean", []):
            if item.get("name") == field_name:
                return item.get("val")
        return None

    def _populate_data_display(
        self, layout: QVBoxLayout, data: dict, tag_type: str = ""
    ) -> None:
        """Populate a layout with read-only data display, grouping fields with rationales."""
        rationale_fields = self._get_rationale_fields_for_tag(tag_type)

        # Track which fields we've displayed (to avoid duplicates)
        displayed_fields = set()

        # First, display fields that support rationales with their grouped display
        for field_name in rationale_fields:
            field_val = self._get_field_value(data, field_name)
            if field_val is None:
                continue

            displayed_fields.add(field_name)
            displayed_fields.add(f"{field_name}_rationale")
            displayed_fields.add(f"{field_name}_rationale_rating")
            displayed_fields.add(f"{field_name}_human_rationale")

            # Create grouped display for this field
            group_widget = self._create_field_group_display(
                field_name, field_val, data
            )
            layout.addWidget(group_widget)

        # Then display remaining fields that don't have rationale support
        for item in data.get("text", []):
            name = item.get("name", "")
            if name in displayed_fields or name.endswith("_rationale") or name.endswith("_rating"):
                continue
            val = item.get("val") or ""
            layout.addWidget(QLabel(f"  {name}: {val}"))

        for item in data.get("num", []):
            name = item.get("name", "")
            if name in displayed_fields:
                continue
            val = item.get("val", 0)
            val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
            layout.addWidget(QLabel(f"  {name}: {val_str}"))

        for item in data.get("boolean", []):
            name = item.get("name", "")
            if name in displayed_fields:
                continue
            val = "Yes" if item.get("val", False) else "No"
            layout.addWidget(QLabel(f"  {name}: {val}"))

        for item in data.get("vec", []):
            name = item.get("name", "")
            if name in displayed_fields:
                continue
            vals = item.get("val", [])
            val_str = ", ".join(self._format_vec_value(v) for v in vals)
            layout.addWidget(QLabel(f"  {name}: {val_str}"))

    def _create_field_group_display(
        self, field_name: str, field_val: Any, data: dict
    ) -> QWidget:
        """Create a grouped display for a field with its rationale."""
        widget = QWidget()
        vlayout = QVBoxLayout(widget)
        vlayout.setContentsMargins(4, 4, 4, 4)
        vlayout.setSpacing(2)

        # Field value
        vlayout.addWidget(QLabel(f"  <b>{field_name}</b>: {field_val}"))

        # VLM rationale
        vlm_rationale = self._get_field_value(data, f"{field_name}_rationale")
        if vlm_rationale:
            rationale_label = QLabel(f"    VLM: {vlm_rationale}")
            rationale_label.setStyleSheet("font-style: italic; color: #666; font-size: 11px;")
            rationale_label.setWordWrap(True)
            vlayout.addWidget(rationale_label)

            # Rating display
            rating = self._get_field_value(data, f"{field_name}_rationale_rating")
            if rating == "good":
                rating_text = "    Rating: 👍 Good"
            elif rating == "bad":
                rating_text = "    Rating: 👎 Bad"
            else:
                rating_text = "    Rating: (unrated)"
            rating_label = QLabel(rating_text)
            rating_label.setStyleSheet("color: #888; font-size: 10px;")
            vlayout.addWidget(rating_label)
        else:
            vlayout.addWidget(QLabel("    VLM: —"))

        # Human rationale
        human_rationale = self._get_field_value(data, f"{field_name}_human_rationale")
        if human_rationale:
            human_label = QLabel(f"    Human: {human_rationale}")
            human_label.setStyleSheet("font-style: italic; color: #484; font-size: 11px;")
            human_label.setWordWrap(True)
            vlayout.addWidget(human_label)
        else:
            vlayout.addWidget(QLabel("    Human: —"))

        return widget

    def _start_editing_tag(self, tag_id: str) -> None:
        """Switch a tag widget to editing mode."""
        if tag_id not in self._tag_editors:
            return

        editor_info = self._tag_editors[tag_id]
        if editor_info["is_editing"]:
            return

        editor_info["is_editing"] = True
        data = editor_info["data"]
        tag_type = editor_info["tag_type"]
        data_container = editor_info["data_container"]

        # Clear current display
        old_layout = data_container.layout()
        while old_layout.count():
            item = old_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Create editors for editable fields
        editors = {}
        rationale_fields = self._get_rationale_fields_for_tag(tag_type)

        # Track which fields we've displayed
        displayed_fields = set()

        # First, display fields that support rationales with editing UI
        for field_name in rationale_fields:
            field_val = self._get_field_value(data, field_name)
            if field_val is None:
                continue

            displayed_fields.add(field_name)
            displayed_fields.add(f"{field_name}_rationale")
            displayed_fields.add(f"{field_name}_rationale_rating")
            displayed_fields.add(f"{field_name}_human_rationale")

            # Create edit group for this field
            group_widget, field_editors = self._create_field_group_edit(
                field_name, field_val, data
            )
            editors.update(field_editors)
            old_layout.addWidget(group_widget)

        # Text fields (except rationale-related and annotator)
        for item in data.get("text", []):
            name = item.get("name", "")
            if name in displayed_fields or name == "annotator":
                continue
            if name.endswith("_rationale") or name.endswith("_rating"):
                continue
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
            if name == "confidence" or name in displayed_fields:
                continue
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
            if name in displayed_fields:
                continue
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
            if name in displayed_fields:
                continue
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

    def _create_field_group_edit(
        self, field_name: str, field_val: Any, data: dict
    ) -> tuple[QWidget, dict]:
        """Create an editing group for a field with rating and human rationale."""
        widget = QWidget()
        widget.setStyleSheet("background-color: rgba(100,100,150,0.1); border-radius: 4px; margin: 2px;")
        vlayout = QVBoxLayout(widget)
        vlayout.setContentsMargins(6, 4, 6, 4)
        vlayout.setSpacing(4)

        editors = {}

        # Field value (editable)
        value_row = QHBoxLayout()
        value_row.addWidget(QLabel(f"  <b>{field_name}</b>:"))
        value_editor = QLineEdit(str(field_val))
        value_editor.setMinimumWidth(150)
        editors[("text", field_name)] = value_editor
        value_row.addWidget(value_editor)
        value_row.addStretch()
        value_widget = QWidget()
        value_widget.setLayout(value_row)
        vlayout.addWidget(value_widget)

        # VLM rationale (read-only) with rating buttons
        vlm_rationale = self._get_field_value(data, f"{field_name}_rationale")
        if vlm_rationale:
            rationale_label = QLabel(f"    VLM: {vlm_rationale}")
            rationale_label.setStyleSheet("font-style: italic; color: #666; font-size: 11px;")
            rationale_label.setWordWrap(True)
            vlayout.addWidget(rationale_label)

            # Rating buttons
            rating_row = QHBoxLayout()
            rating_row.addWidget(QLabel("    Rate VLM:"))

            current_rating = self._get_field_value(data, f"{field_name}_rationale_rating") or "unrated"

            good_btn = QRadioButton("👍 Good")
            bad_btn = QRadioButton("👎 Bad")
            unrated_btn = QRadioButton("Unrated")

            # Set current rating
            if current_rating == "good":
                good_btn.setChecked(True)
            elif current_rating == "bad":
                bad_btn.setChecked(True)
            else:
                unrated_btn.setChecked(True)

            # Group the buttons
            rating_group = QButtonGroup(widget)
            rating_group.addButton(good_btn, 1)
            rating_group.addButton(bad_btn, 2)
            rating_group.addButton(unrated_btn, 0)

            rating_row.addWidget(good_btn)
            rating_row.addWidget(bad_btn)
            rating_row.addWidget(unrated_btn)
            rating_row.addStretch()

            rating_widget = QWidget()
            rating_widget.setLayout(rating_row)
            vlayout.addWidget(rating_widget)

            editors[("rating", f"{field_name}_rationale_rating")] = rating_group
        else:
            vlayout.addWidget(QLabel("    VLM: — (no VLM rationale)"))

        # Human rationale (editable)
        human_row = QVBoxLayout()
        human_row.addWidget(QLabel("    Human rationale:"))

        current_human = self._get_field_value(data, f"{field_name}_human_rationale") or ""
        human_editor = QTextEdit()
        human_editor.setPlainText(current_human)
        human_editor.setMaximumHeight(60)
        human_editor.setPlaceholderText("Add your rationale here...")
        editors[("human_rationale", f"{field_name}_human_rationale")] = human_editor
        human_row.addWidget(human_editor)

        human_widget = QWidget()
        human_widget.setLayout(human_row)
        vlayout.addWidget(human_widget)

        return widget, editors

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

        # Helper to find or create a text field
        def ensure_text_field(name: str) -> dict:
            for item in new_data.get("text", []):
                if item.get("name") == name:
                    return item
            # Create new field
            if "text" not in new_data:
                new_data["text"] = []
            new_item = {"name": name, "val": ""}
            new_data["text"].append(new_item)
            return new_item

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

        # Handle rating fields
        for key, editor in editors.items():
            if key[0] == "rating":
                field_name = key[1]  # e.g., "precipitation_rationale_rating"
                rating_group = editor
                checked_id = rating_group.checkedId()
                if checked_id == 1:
                    rating_val = "good"
                elif checked_id == 2:
                    rating_val = "bad"
                else:
                    rating_val = "unrated"

                # Find or create the rating field
                rating_field = ensure_text_field(field_name)
                rating_field["val"] = rating_val

        # Handle human rationale fields
        for key, editor in editors.items():
            if key[0] == "human_rationale":
                field_name = key[1]  # e.g., "precipitation_human_rationale"
                human_text = editor.toPlainText().strip()

                if human_text:
                    # Find or create the human rationale field
                    human_field = ensure_text_field(field_name)
                    human_field["val"] = human_text
                else:
                    # Remove empty human rationale field if it exists
                    new_data["text"] = [
                        item for item in new_data.get("text", [])
                        if item.get("name") != field_name
                    ]

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
        tag_type = editor_info["tag_type"]
        data_container = editor_info["data_container"]

        # Clear current display
        old_layout = data_container.layout()
        while old_layout.count():
            item = old_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Repopulate with read-only display
        self._populate_data_display(old_layout, data, tag_type=tag_type)

    def _create_data_widget(
        self, title: str, data: dict, tag_type: str = ""
    ) -> QWidget:
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

        self._populate_data_display(vlayout, data, tag_type=tag_type)

        return widget

    def _format_vec_value(self, val: Any) -> str:
        """Format a single value in a vec list for display."""
        if isinstance(val, float):
            return f"{val:.2f}"
        return str(val)
