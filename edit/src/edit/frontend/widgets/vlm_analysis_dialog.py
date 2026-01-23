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
    QSpinBox,
    QStyle,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# Fields that support rationales, organized by tag type
RATIONALE_FIELDS = {
    "Weather": [
        "precipitation", "precipitation_intensity", "particulates",
        "time_of_day", "sun_position", "cloud_cover",
    ],
    "Traffic": [
        "density", "flow", "temporary_structures",
        "pedestrians_present", "cyclists_present", "special_vehicles_present",
    ],
    "Road": [
        "drivable_area_type", "surface_type", "surface_condition", "surface_quality",
    ],
}


class VLMAnalysisDialog(QDialog):
    """Modal dialog displaying VLM contexts and tags with per-field editing."""

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
        self.resize(650, 600)

        self._contexts = contexts or {}
        self._tags = tags or {}
        self._frontend_state = frontend_state
        self._undo_manager = undo_manager
        self._undo_context = undo_context

        # Track editing state for each field
        self._field_widgets: dict[str, dict] = {}
        self._tag_containers: dict[str, QWidget] = {}

        self._setup_ui()

    def _can_edit(self) -> bool:
        """Check if editing is supported."""
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
            self._tag_containers[tag_id] = widget
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

            interval_str = ", ".join(
                f"{iv.get('frame_start', 0)}-{iv.get('frame_end', 0)}"
                for iv in intervals
            )
            layout.addWidget(
                self._create_data_widget(
                    f"{ctx_type.replace('Context', '')} (frames {interval_str})",
                    ctx_data,
                    tag_type=ctx_type.replace("Context", ""),
                    editable=False,
                )
            )

    def _no_data_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("color: #888; font-style: italic; margin-left: 8px;")
        return label

    def _get_rationale_fields_for_tag(self, tag_type: str) -> list[str]:
        """Get fields that support rationales for a tag type."""
        return RATIONALE_FIELDS.get(tag_type, [])

    def _get_field_value(self, data: dict, field_name: str) -> Any:
        """Get a field value from tag data."""
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

    def _get_vec_value(self, data: dict, field_name: str) -> list | None:
        """Get a vec value from tag data."""
        for item in data.get("vec", []):
            if item.get("name") == field_name:
                return item.get("val")
        return None

    def _create_tag_widget(self, tag_id: str, title: str, data: dict) -> QWidget:
        """Create a widget for a tag with per-field inline editing."""
        widget = QWidget()
        widget.setStyleSheet(
            "background-color: rgba(128,128,128,0.1); border-radius: 4px;"
        )
        vlayout = QVBoxLayout(widget)
        vlayout.setContentsMargins(10, 6, 10, 6)
        vlayout.setSpacing(4)

        # Title
        title_label = QLabel(f"<b>{title}</b>")
        vlayout.addWidget(title_label)

        # Create field rows
        self._populate_tag_fields(vlayout, tag_id, data, tag_type=title)

        return widget

    def _populate_tag_fields(
        self, layout: QVBoxLayout, tag_id: str, data: dict, tag_type: str = ""
    ) -> None:
        """Populate a layout with per-field displays/editors."""
        rationale_fields = self._get_rationale_fields_for_tag(tag_type)
        displayed_fields = set()

        # Fields with rationale support - display grouped
        for field_name in rationale_fields:
            field_val = self._get_field_value(data, field_name)
            if field_val is None:
                continue

            displayed_fields.add(field_name)
            displayed_fields.add(f"{field_name}_rationale")
            displayed_fields.add(f"{field_name}_rationale_rating")
            displayed_fields.add(f"{field_name}_human_rationale")

            field_widget = self._create_field_row(
                tag_id, field_name, field_val, data, has_rationale=True, tag_type=tag_type
            )
            layout.addWidget(field_widget)

        # Text fields without rationale
        for item in data.get("text", []):
            name = item.get("name", "")
            if name in displayed_fields or name.endswith("_rationale") or name.endswith("_rating"):
                continue
            val = item.get("val") or ""
            field_widget = self._create_field_row(
                tag_id, name, val, data, has_rationale=False, field_type="text", tag_type=tag_type
            )
            layout.addWidget(field_widget)
            displayed_fields.add(name)

        # Num fields
        for item in data.get("num", []):
            name = item.get("name", "")
            if name in displayed_fields:
                continue
            val = item.get("val", 0)
            field_widget = self._create_field_row(
                tag_id, name, val, data, has_rationale=False, field_type="num", tag_type=tag_type
            )
            layout.addWidget(field_widget)
            displayed_fields.add(name)

        # Boolean fields
        for item in data.get("boolean", []):
            name = item.get("name", "")
            if name in displayed_fields:
                continue
            val = item.get("val", False)
            field_widget = self._create_field_row(
                tag_id, name, val, data, has_rationale=False, field_type="boolean", tag_type=tag_type
            )
            layout.addWidget(field_widget)
            displayed_fields.add(name)

    def _create_field_row(
        self,
        tag_id: str,
        field_name: str,
        field_val: Any,
        data: dict,
        has_rationale: bool = False,
        field_type: str = "text",
        tag_type: str = "",
    ) -> QWidget:
        """Create a row for a single field with inline editing on click."""
        widget = QWidget()
        if has_rationale:
            widget.setStyleSheet("background-color: rgba(100,100,150,0.08); border-radius: 3px; margin: 1px;")
        vlayout = QVBoxLayout(widget)
        vlayout.setContentsMargins(4, 2, 4, 2)
        vlayout.setSpacing(2)

        # Get confidence and annotator for this field
        conf_list = self._get_vec_value(data, f"{field_name}_confidence")
        ann_list = self._get_vec_value(data, f"{field_name}_annotator")
        confidence = conf_list[0] if conf_list else None
        annotator = ann_list[0] if ann_list else None

        # Field value row (clickable to edit)
        value_row = QHBoxLayout()
        value_row.setSpacing(4)

        # Format display value
        if isinstance(field_val, bool):
            display_val = "Yes" if field_val else "No"
        elif isinstance(field_val, float):
            display_val = f"{field_val:.2f}"
        else:
            display_val = str(field_val)

        # Value label (clickable)
        value_label = QLabel(f"  <b>{field_name}</b>: {display_val}")
        value_label.setCursor(Qt.CursorShape.PointingHandCursor)
        if self._can_edit():
            value_label.mousePressEvent = lambda e, tid=tag_id, fn=field_name, fv=field_val, ft=field_type, d=data, tt=tag_type: (
                self._start_field_edit(tid, fn, fv, ft, d, tt)
            )
        value_row.addWidget(value_label)

        # Confidence/annotator info
        if confidence is not None:
            conf_pct = int(confidence * 100)
            info_label = QLabel(f"({conf_pct}%)")
            info_label.setStyleSheet("color: #666; font-size: 10px;")
            value_row.addWidget(info_label)

        value_row.addStretch()
        value_container = QWidget()
        value_container.setLayout(value_row)
        vlayout.addWidget(value_container)

        # Store reference for potential editing
        field_key = f"{tag_id}:{field_name}"
        self._field_widgets[field_key] = {
            "widget": widget,
            "value_label": value_label,
            "tag_id": tag_id,
            "field_name": field_name,
            "field_type": field_type,
            "data": data,
            "tag_type": tag_type,
        }

        # Rationale section (if applicable)
        if has_rationale:
            vlm_rationale = self._get_field_value(data, f"{field_name}_rationale")
            if vlm_rationale:
                rat_label = QLabel(f"    VLM: {vlm_rationale}")
                rat_label.setStyleSheet("font-style: italic; color: #666; font-size: 10px;")
                rat_label.setWordWrap(True)
                vlayout.addWidget(rat_label)

                # Rating
                rating = self._get_field_value(data, f"{field_name}_rationale_rating")
                if rating == "good":
                    rating_text = "    Rating: 👍"
                elif rating == "bad":
                    rating_text = "    Rating: 👎"
                else:
                    rating_text = "    Rating: —"
                rating_label = QLabel(rating_text)
                rating_label.setStyleSheet("color: #888; font-size: 10px;")
                vlayout.addWidget(rating_label)

            # Human rationale
            human_rationale = self._get_field_value(data, f"{field_name}_human_rationale")
            if human_rationale:
                human_label = QLabel(f"    Human: {human_rationale}")
                human_label.setStyleSheet("font-style: italic; color: #484; font-size: 10px;")
                human_label.setWordWrap(True)
                vlayout.addWidget(human_label)

        return widget

    def _start_field_edit(
        self,
        tag_id: str,
        field_name: str,
        field_val: Any,
        field_type: str,
        data: dict,
        tag_type: str,
    ) -> None:
        """Start inline editing for a single field."""
        from PyQt6.QtWidgets import QDialog as QD, QDialogButtonBox

        # Get annotator
        annotator = self._frontend_state.require_current_annotator()
        if not annotator:
            QMessageBox.warning(
                self,
                "Annotator Required",
                "Please set an annotator name to edit VLM tags.",
            )
            return

        # Create edit dialog for this field
        edit_dialog = QD(self)
        edit_dialog.setWindowTitle(f"Edit {field_name}")
        edit_dialog.setModal(True)
        edit_dialog.resize(400, 300)

        dlg_layout = QVBoxLayout(edit_dialog)
        dlg_layout.setSpacing(8)

        # Field value editor
        dlg_layout.addWidget(QLabel(f"<b>{field_name}</b>:"))

        if field_type == "boolean":
            value_editor = QCheckBox("Value")
            value_editor.setChecked(bool(field_val))
        elif field_type == "num":
            value_editor = QLineEdit(str(field_val))
        else:
            value_editor = QLineEdit(str(field_val))

        dlg_layout.addWidget(value_editor)

        # Rationale section (if this field supports it)
        rationale_fields = self._get_rationale_fields_for_tag(tag_type)
        rating_group = None
        human_editor = None

        if field_name in rationale_fields:
            vlm_rationale = self._get_field_value(data, f"{field_name}_rationale")

            if vlm_rationale:
                dlg_layout.addWidget(QLabel("VLM Rationale (read-only):"))
                vlm_label = QLabel(vlm_rationale)
                vlm_label.setStyleSheet("font-style: italic; color: #666; padding: 4px; background: #f0f0f0;")
                vlm_label.setWordWrap(True)
                dlg_layout.addWidget(vlm_label)

                # Rating buttons
                dlg_layout.addWidget(QLabel("Rate VLM rationale:"))
                rating_row = QHBoxLayout()
                good_btn = QRadioButton("👍 Good")
                bad_btn = QRadioButton("👎 Bad")
                unrated_btn = QRadioButton("Unrated")

                current_rating = self._get_field_value(data, f"{field_name}_rationale_rating") or "unrated"
                if current_rating == "good":
                    good_btn.setChecked(True)
                elif current_rating == "bad":
                    bad_btn.setChecked(True)
                else:
                    unrated_btn.setChecked(True)

                rating_group = QButtonGroup(edit_dialog)
                rating_group.addButton(good_btn, 1)
                rating_group.addButton(bad_btn, 2)
                rating_group.addButton(unrated_btn, 0)

                rating_row.addWidget(good_btn)
                rating_row.addWidget(bad_btn)
                rating_row.addWidget(unrated_btn)
                rating_row.addStretch()
                rating_container = QWidget()
                rating_container.setLayout(rating_row)
                dlg_layout.addWidget(rating_container)

            # Human rationale
            dlg_layout.addWidget(QLabel("Your rationale:"))
            human_editor = QTextEdit()
            human_editor.setPlainText(
                self._get_field_value(data, f"{field_name}_human_rationale") or ""
            )
            human_editor.setMaximumHeight(60)
            human_editor.setPlaceholderText("Add your rationale here...")
            dlg_layout.addWidget(human_editor)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(edit_dialog.accept)
        button_box.rejected.connect(edit_dialog.reject)
        dlg_layout.addWidget(button_box)

        if edit_dialog.exec() == QD.DialogCode.Accepted:
            self._apply_field_edit(
                tag_id,
                field_name,
                field_type,
                value_editor,
                rating_group,
                human_editor,
                annotator,
                data,
            )

    def _apply_field_edit(
        self,
        tag_id: str,
        field_name: str,
        field_type: str,
        value_editor,
        rating_group,
        human_editor,
        annotator: str,
        old_data: dict,
    ) -> None:
        """Apply the field edit and update via undo system."""
        new_data = deepcopy(old_data)

        # Get new value
        if field_type == "boolean":
            new_val = value_editor.isChecked()
        elif field_type == "num":
            try:
                new_val = float(value_editor.text())
            except ValueError:
                return  # Invalid input
        else:
            new_val = value_editor.text()

        # Update field value
        for item in new_data.get("text", []):
            if item.get("name") == field_name:
                item["val"] = new_val
                break
        for item in new_data.get("num", []):
            if item.get("name") == field_name:
                item["val"] = new_val
                break
        for item in new_data.get("boolean", []):
            if item.get("name") == field_name:
                item["val"] = new_val
                break

        # Helper to find or create a text field
        def ensure_text_field(name: str) -> dict:
            for item in new_data.get("text", []):
                if item.get("name") == name:
                    return item
            if "text" not in new_data:
                new_data["text"] = []
            new_item = {"name": name, "val": ""}
            new_data["text"].append(new_item)
            return new_item

        # Helper to find or create a vec field
        def ensure_vec_field(name: str) -> dict:
            if "vec" not in new_data:
                new_data["vec"] = []
            for item in new_data["vec"]:
                if item.get("name") == name:
                    return item
            new_item = {"name": name, "val": []}
            new_data["vec"].append(new_item)
            return new_item

        # Update per-field annotator (prepend)
        ann_field = ensure_vec_field(f"{field_name}_annotator")
        if isinstance(ann_field["val"], list):
            ann_field["val"].insert(0, annotator)
        else:
            ann_field["val"] = [annotator]

        # Update per-field confidence (prepend 1.0 for human)
        conf_field = ensure_vec_field(f"{field_name}_confidence")
        if isinstance(conf_field["val"], list):
            conf_field["val"].insert(0, 1.0)
        else:
            conf_field["val"] = [1.0]

        # Handle rating
        if rating_group is not None:
            checked_id = rating_group.checkedId()
            if checked_id == 1:
                rating_val = "good"
            elif checked_id == 2:
                rating_val = "bad"
            else:
                rating_val = "unrated"
            rating_field = ensure_text_field(f"{field_name}_rationale_rating")
            rating_field["val"] = rating_val

        # Handle human rationale
        if human_editor is not None:
            human_text = human_editor.toPlainText().strip()
            if human_text:
                human_field = ensure_text_field(f"{field_name}_human_rationale")
                human_field["val"] = human_text
            else:
                # Remove empty human rationale
                new_data["text"] = [
                    item for item in new_data.get("text", [])
                    if item.get("name") != f"{field_name}_human_rationale"
                ]

        # Execute via undo system
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

        # Update local state
        self._tags[tag_id]["tag_data"] = new_data

        # Update the field widget display
        field_key = f"{tag_id}:{field_name}"
        if field_key in self._field_widgets:
            info = self._field_widgets[field_key]
            info["data"] = new_data
            # Update label
            if isinstance(new_val, bool):
                display_val = "Yes" if new_val else "No"
            elif isinstance(new_val, float):
                display_val = f"{new_val:.2f}"
            else:
                display_val = str(new_val)
            info["value_label"].setText(f"  <b>{field_name}</b>: {display_val}")

    def _create_data_widget(
        self, title: str, data: dict, tag_type: str = "", editable: bool = False
    ) -> QWidget:
        """Create a read-only widget for context data."""
        widget = QWidget()
        widget.setStyleSheet(
            "background-color: rgba(128,128,128,0.1); border-radius: 4px;"
        )
        vlayout = QVBoxLayout(widget)
        vlayout.setContentsMargins(10, 6, 10, 6)
        vlayout.setSpacing(2)

        title_label = QLabel(f"<b>{title}</b>")
        vlayout.addWidget(title_label)

        # Simple field display for contexts (read-only)
        for item in data.get("text", []):
            name = item.get("name", "")
            if name.endswith("_rationale") or name.endswith("_rating"):
                continue
            val = item.get("val") or ""
            vlayout.addWidget(QLabel(f"  {name}: {val}"))

        for item in data.get("num", []):
            name = item.get("name", "")
            val = item.get("val", 0)
            val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
            vlayout.addWidget(QLabel(f"  {name}: {val_str}"))

        for item in data.get("boolean", []):
            name = item.get("name", "")
            val = "Yes" if item.get("val", False) else "No"
            vlayout.addWidget(QLabel(f"  {name}: {val}"))

        return widget
