# edit/frontend/utils/edit_panels.py
from __future__ import annotations
from typing import Callable, Dict, Any
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QToolButton,
    QLabel,
    QLineEdit,
    QComboBox,
    QSizePolicy,
)


def create_collapsible_object_details(
    *,
    parent: QWidget,
    title: str = "Object details",
    populate_types: Callable[[QComboBox], None],
    on_name_edited: Callable[[], None],
    on_type_changed: Callable[[str], None],
) -> Dict[str, Any]:
    """
    Build a collapsible 'Object details' panel
    """
    container = QFrame(parent)
    container.setEnabled(False)
    root_v = QVBoxLayout(container)
    root_v.setContentsMargins(0, 0, 0, 0)
    root_v.setSpacing(6)

    header = QHBoxLayout()
    header.setContentsMargins(0, 0, 0, 0)
    header.setSpacing(6)

    toggle = QToolButton(container)
    toggle.setCheckable(True)
    toggle.setChecked(False)  # collapsed initially
    toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    toggle.setArrowType(Qt.ArrowType.RightArrow)
    toggle.setText(title)

    container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    toggle.setStyleSheet("QToolButton { text-align: left }")

    header.addWidget(toggle, 1)
    root_v.addLayout(header)

    content = QWidget(container)
    form = QFormLayout(content)
    form.setContentsMargins(8, 0, 0, 0)

    id_label = QLabel("-", content)
    name_edit = QLineEdit(content)

    type_combo = QComboBox(content)
    type_combo.setEditable(False)

    form.addRow("ID:", id_label)
    form.addRow("Name:", name_edit)
    form.addRow("Type:", type_combo)

    content.setVisible(False)
    root_v.addWidget(content)

    def _on_toggle(checked: bool):
        toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        content.setVisible(bool(checked))

    toggle.toggled.connect(_on_toggle)
    name_edit.editingFinished.connect(on_name_edited)
    type_combo.currentTextChanged.connect(on_type_changed)

    return {
        "container": container,
        "toggle": toggle,
        "content": content,
        "id_label": id_label,
        "name_edit": name_edit,
        "type_combo": type_combo,
    }
