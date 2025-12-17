"""Analysis panel widget with stats, chart, and thumbnails."""

import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QGroupBox,
    QScrollArea,
    QSplitter,
    QSizePolicy,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from ..states.app_state import AppState
from ...models.dataset import AggregatedStats

logger = logging.getLogger(__name__)


class AnalysisPanel(QWidget):
    """Panel displaying analysis results with stats, chart, and thumbnails."""

    def __init__(self, app_state: AppState, parent=None):
        super().__init__(parent)

        self.app_state = app_state

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Splitter for stats/chart and thumbnails
        splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(splitter)

        # Top section: stats and chart
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)

        # Stats table
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)

        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        self.stats_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        stats_layout.addWidget(self.stats_table)

        # Class distribution table
        self.class_table = QTableWidget()
        self.class_table.setColumnCount(3)
        self.class_table.setHorizontalHeaderLabels(["ID", "Class", "Count"])
        self.class_table.horizontalHeader().setStretchLastSection(True)
        self.class_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        stats_layout.addWidget(self.class_table)

        top_layout.addWidget(stats_group)

        # Chart
        chart_group = QGroupBox("Class Distribution")
        chart_layout = QVBoxLayout(chart_group)

        if HAS_MATPLOTLIB:
            self.figure = Figure(figsize=(5, 4), dpi=100)
            self.canvas = FigureCanvas(self.figure)
            self.canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            chart_layout.addWidget(self.canvas)
        else:
            chart_layout.addWidget(QLabel("matplotlib not available"))
            self.figure = None
            self.canvas = None

        top_layout.addWidget(chart_group)

        splitter.addWidget(top_widget)

        # Bottom section: thumbnails
        thumb_group = QGroupBox("Sample Images")
        thumb_layout = QVBoxLayout(thumb_group)

        self.thumbnail_scroll = QScrollArea()
        self.thumbnail_scroll.setWidgetResizable(True)
        self.thumbnail_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self.thumbnail_container = QWidget()
        self.thumbnail_layout = QHBoxLayout(self.thumbnail_container)
        self.thumbnail_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.thumbnail_scroll.setWidget(self.thumbnail_container)

        thumb_layout.addWidget(self.thumbnail_scroll)

        splitter.addWidget(thumb_group)

        # Set initial splitter sizes
        splitter.setSizes([400, 200])

        # Placeholder for no data
        self._show_no_data()

    def _connect_signals(self):
        """Connect app state signals."""
        self.app_state.analysis_updated.connect(self._on_analysis_updated)

    def _on_analysis_updated(self, stats: Optional[AggregatedStats]):
        """Handle analysis update."""
        if stats and stats.is_valid:
            self._show_stats(stats)
            self._show_chart(stats)
            self._show_thumbnails(stats)
        else:
            self._show_no_data()

    def _show_no_data(self):
        """Show placeholder when no data is available."""
        self.stats_table.setRowCount(1)
        self.stats_table.setItem(0, 0, QTableWidgetItem("No data"))
        self.stats_table.setItem(
            0, 1, QTableWidgetItem("Select datasets and click Analyze")
        )

        self.class_table.setRowCount(0)

        if self.figure:
            self.figure.clear()
            self.canvas.draw()

        self._clear_thumbnails()

    def _show_stats(self, stats: AggregatedStats):
        """Display statistics in the table."""
        rows = [
            ("Datasets", str(len(stats.dataset_names))),
            ("Total Images", str(stats.total_images)),
            ("  - Training", str(stats.total_train_images)),
            ("  - Validation", str(stats.total_val_images)),
            ("Total Objects", str(stats.total_objects)),
            ("Classes", str(stats.num_classes)),
        ]

        # Add per-dataset breakdown
        for name in stats.dataset_names:
            img_count = stats.per_dataset_images.get(name, 0)
            obj_count = stats.per_dataset_objects.get(name, 0)
            rows.append((f"  {name}", f"{img_count} imgs, {obj_count} objs"))

        self.stats_table.setRowCount(len(rows))
        for i, (metric, value) in enumerate(rows):
            self.stats_table.setItem(i, 0, QTableWidgetItem(metric))
            self.stats_table.setItem(i, 1, QTableWidgetItem(value))

        # Class distribution table
        self.class_table.setRowCount(len(stats.class_names))
        for i, (class_id, name) in enumerate(sorted(stats.class_names.items())):
            count = stats.class_distribution.get(class_id, 0)
            self.class_table.setItem(i, 0, QTableWidgetItem(str(class_id)))
            self.class_table.setItem(i, 1, QTableWidgetItem(name))
            self.class_table.setItem(i, 2, QTableWidgetItem(str(count)))

        self.class_table.resizeColumnsToContents()

    def _show_chart(self, stats: AggregatedStats):
        """Display bar chart of class distribution."""
        if not self.figure:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Get data for chart
        class_names = []
        counts = []
        for class_id in sorted(stats.class_names.keys()):
            name = stats.class_names[class_id]
            count = stats.class_distribution.get(class_id, 0)
            # Truncate long names
            if len(name) > 15:
                name = name[:12] + "..."
            class_names.append(name)
            counts.append(count)

        if not counts:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            self.canvas.draw()
            return

        # Create bar chart
        bars = ax.bar(range(len(class_names)), counts, color="steelblue")
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Count")
        ax.set_title("Objects per Class")

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    str(count),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        self.figure.tight_layout()
        self.canvas.draw()

    def _show_thumbnails(self, stats: AggregatedStats):
        """Display sample image thumbnails."""
        self._clear_thumbnails()

        for name, image_path in stats.sample_images.items():
            self._add_thumbnail(name, image_path)

    def _clear_thumbnails(self):
        """Clear all thumbnails."""
        while self.thumbnail_layout.count():
            child = self.thumbnail_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def _add_thumbnail(self, name: str, image_path: str):
        """Add a thumbnail widget."""
        try:
            container = QWidget()
            layout = QVBoxLayout(container)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(2)

            # Image
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                logger.warning(f"Failed to load image: {image_path}")
                return

            # Scale to thumbnail size
            scaled = pixmap.scaled(
                200,
                150,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

            image_label = QLabel()
            image_label.setPixmap(scaled)
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(image_label)

            # Dataset name
            name_label = QLabel(name)
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_label.setStyleSheet("font-size: 10px;")
            layout.addWidget(name_label)

            self.thumbnail_layout.addWidget(container)

        except Exception as e:
            logger.error(f"Failed to add thumbnail for {name}: {e}")
