"""Utility delegates and helpers for sidebar confidence entries."""

from PyQt6.QtWidgets import (
    QStyledItemDelegate,
    QApplication,
    QStyle,
    QStyleOptionViewItem,
)
from PyQt6.QtGui import QIcon, QPainter
from PyQt6.QtCore import Qt, QSize, QRect

from savant_app.frontend.theme.constants import SIDEBAR_CONFIDENCE_ICON_SIZE


class SidebarConfidenceIssueItemDelegate(QStyledItemDelegate):
    """Override the default item painting to render the issue text followed by the severity icon."""

    _ICON_SPACING = 6

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index):
        option_copy = QStyleOptionViewItem(option)
        self.initStyleOption(option_copy, index)

        icon = QIcon(option_copy.icon)
        has_icon = not icon.isNull()
        icon_side = SIDEBAR_CONFIDENCE_ICON_SIZE if has_icon else 0
        spacing = self._ICON_SPACING if has_icon else 0

        option_copy.features &= ~QStyleOptionViewItem.ViewItemFeature.HasDecoration
        option_copy.decorationSize = QSize(0, 0)
        option_copy.displayAlignment = (
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )

        full_rect = option_copy.rect
        text_rect = QRect(full_rect.left(), full_rect.top(),
                          full_rect.width() - icon_side - spacing, full_rect.height())
        option_copy.rect = text_rect
        option_copy.icon = QIcon()

        style = option_copy.widget.style() if option_copy.widget else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem,
                          option_copy, painter, option_copy.widget)

        if not has_icon:
            return

        icon_rect = QRect(
            text_rect.right() + spacing,
            full_rect.top() + max(0, (full_rect.height() - icon_side) // 2),
            icon_side,
            icon_side,
        )
        icon.paint(painter, icon_rect, Qt.AlignmentFlag.AlignCenter)

    def sizeHint(self, option: QStyleOptionViewItem, index):
        base = super().sizeHint(option, index)
        height = max(base.height(), SIDEBAR_CONFIDENCE_ICON_SIZE + 4)
        width = base.width() + SIDEBAR_CONFIDENCE_ICON_SIZE + self._ICON_SPACING
        return QSize(width, height)
