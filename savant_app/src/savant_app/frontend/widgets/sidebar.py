from PyQt6.QtCore import (
    QEvent,
    QItemSelectionModel,
    QPoint,
    QSignalBlocker,
    QSize,
    Qt,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import QFont, QIcon, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from savant_app.controllers.annotation_controller import AnnotationController
from savant_app.controllers.project_state_controller import ProjectStateController
from savant_app.controllers.video_controller import VideoController
from savant_app.frontend.exceptions import InvalidObjectIDFormat
from savant_app.frontend.states.frontend_state import FrontendState
from savant_app.frontend.states.sidebar_state import SidebarState
from savant_app.frontend.theme.constants import (
    SIDEBAR_CONFIDENCE_ICON_SIZE,
    SIDEBAR_CONFIDENCE_LIST_HEIGHT,
    SIDEBAR_ERROR_HIGHLIGHT,
    SIDEBAR_HIGHLIGHT_TEXT_COLOUR,
    SIDEBAR_WARNING_HIGHLIGHT,
    get_error_icon,
    get_warning_icon,
)
from savant_app.frontend.theme.sidebar_styles import apply_issue_sort_button_style
from savant_app.frontend.utils.assets import icon
from savant_app.frontend.utils.edit_panel import create_collapsible_object_details
from savant_app.frontend.utils.settings_store import get_ontology_path
from savant_app.frontend.utils.sidebar_confidence_items import (
    SidebarConfidenceIssueItemDelegate,
)
from savant_app.frontend.utils.undo import (
    AddFrameTagCommand,
    FrameTagSnapshot,
    InterpolateAnnotationsCommand,
    RemoveFrameTagCommand,
)
from savant_app.frontend.widgets.interpolation_dialog import InterpolationDialog
from savant_app.frontend.widgets.settings import get_action_interval_offset
from savant_app.services.exceptions import VideoLoadError


class Sidebar(QWidget):

    open_video = pyqtSignal(str)
    open_config = pyqtSignal(str)
    # TODO: Rename to add_new_bbox_new_obj
    start_bbox_drawing = pyqtSignal(str)
    add_new_bbox_existing_obj = pyqtSignal(str)
    open_project_dir = pyqtSignal(str)
    quick_save = pyqtSignal()
    highlight_selected_object = pyqtSignal(str)
    object_selected = pyqtSignal(str)  # New signal for selection changes
    object_details_changed = pyqtSignal()
    create_relationship = pyqtSignal()  # Signal for creating relationships

    def __init__(
        self,
        video_actors: list[str],
        annotation_controller: AnnotationController,
        video_controller: VideoController,
        project_state_controller: ProjectStateController,
        frontend_state: FrontendState,
        state: SidebarState,
    ):
        super().__init__()
        self.setFixedWidth(200)

        # Setup Scroll Area Hierarchy
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self.content_widget = QWidget()
        main_layout = QVBoxLayout(self.content_widget)

        # Temporary controller until refactor.
        self.annotation_controller: AnnotationController = annotation_controller
        self.video_controller: VideoController = video_controller
        self.project_state_controller: ProjectStateController = project_state_controller
        self.frontend_state: FrontendState = frontend_state

        # State for sidebar
        self.state: SidebarState = state

        # Track the currently selected object ID
        self._selected_annotation_object_id: str | None = None
        self._current_confidence_issues: dict[int, list] = {}
        self._warning_icon = self._create_confidence_icon(get_warning_icon())
        self._error_icon = self._create_confidence_icon(get_error_icon())
        self._confidence_sort_mode: str = "frame"
        self._confidence_sort_ascending: bool = True

        # --- Horizontal layout for New / Load / Save ---
        top_buttons_layout = QHBoxLayout()

        def make_icon_btn(filename: str, tooltip: str) -> QPushButton:
            btn = QPushButton()
            btn.setIcon(icon(filename))
            btn.setIconSize(QSize(30, 30))
            btn.setFlat(True)
            btn.setToolTip(tooltip)
            return btn

        new_btn = make_icon_btn("new_file.svg", "New Project")
        load_btn = make_icon_btn("open_file.svg", "Load Project")
        save_btn = make_icon_btn("save_file.svg", "Save Project")
        # load_config_btn = make_icon_btn(
        #     "open_file.svg", "Load project"
        # )  # same icon as loading video. Will be removed in a
        #             future ticket when config dir is made.

        load_btn.clicked.connect(self._choose_project_dir)
        save_btn.clicked.connect(self._trigger_quick_save)
        # load_config_btn.clicked.connect(self._choose_openlabel_file)

        top_buttons_layout.addWidget(new_btn)
        top_buttons_layout.addWidget(load_btn)
        top_buttons_layout.addWidget(save_btn)
        # top_buttons_layout.addWidget(
        #     load_config_btn
        # )  # to be removed when refactor to config dir

        main_layout.addLayout(top_buttons_layout)

        # --- New BBox Button with dropdown ---
        new_bbox_btn = QPushButton("New BBox")
        new_bbox_btn.clicked.connect(
            lambda: self.create_bbox(video_actors=video_actors)
        )
        main_layout.addWidget(new_bbox_btn)

        # --- New Frame Tag Button ---
        new_tag_btn = QPushButton("New frame tag")
        new_tag_btn.clicked.connect(self._open_frame_tag_dialog)
        main_layout.addWidget(new_tag_btn)

        # --- Interpolation Button ---
        self.interpolate_btn = QPushButton("Interpolate")
        self.interpolate_btn.setEnabled(True)  # Enable now that we have implementation
        self.interpolate_btn.clicked.connect(self._open_interpolation_dialog)
        main_layout.addWidget(self.interpolate_btn)

        # --- Create Relationship Button ---
        self.create_relationship_btn = QPushButton("Create Relationship")
        self.create_relationship_btn.clicked.connect(self._open_relationship_dialog)
        main_layout.addWidget(self.create_relationship_btn)
        # --- Object Details ---
        parts = create_collapsible_object_details(
            parent=self,
            title="Object details",
            populate_types=self._populate_bbox_type_combo_grouped,
            on_name_edited=self._on_details_name_edited,
            on_type_changed=self._on_details_type_changed,
        )

        self.details_container = parts["container"]
        self._details_toggle = parts["toggle"]
        self._details_content = parts["content"]
        self._details_id_label = parts["id_label"]
        self._details_name_edit = parts["name_edit"]
        self._details_type_combo = parts["type_combo"]
        self._editing_object_id: str | None = None

        # --- Relationships Section ---
        relationships_label = QLabel("Relationships:")
        self.relationships_list = QListWidget()
        self.relationships_list.setMinimumHeight(60)
        self.relationships_list.setMaximumHeight(120)
        self.relationships_list.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.relationships_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        # Add relationships section below the form layout
        relationships_layout = QVBoxLayout()
        relationships_layout.addWidget(relationships_label)
        relationships_layout.addWidget(self.relationships_list)
        self._details_content.layout().addRow(relationships_layout)

        main_layout.addWidget(self.details_container)

        # --- Active Objects ---
        main_layout.addWidget(QLabel("Active Objects:"))
        self.active_objects = QListWidget()
        self.active_objects.setMinimumHeight(100)
        self.active_objects.model().rowsInserted.connect(self.adjust_list_sizes)
        self.active_objects.model().rowsRemoved.connect(self.adjust_list_sizes)
        self.active_objects.itemClicked.connect(self._on_active_object_selected)
        self.active_objects.itemSelectionChanged.connect(
            self._on_active_objects_selection_changed
        )
        main_layout.addWidget(self.active_objects)

        warnings_header = QVBoxLayout()
        title_row = QHBoxLayout()
        title_row.addWidget(QLabel("Issues:"))
        title_row.addStretch(1)
        sort_controls = QHBoxLayout()
        sort_controls.setSpacing(12)
        self._frame_sort_btn = QPushButton()
        self._frame_sort_btn.setFlat(True)
        self._frame_sort_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._frame_sort_btn.clicked.connect(
            lambda: self._change_confidence_sort_mode("frame")
        )
        sort_controls.addWidget(self._frame_sort_btn)
        self._id_sort_btn = QPushButton()
        self._id_sort_btn.setFlat(True)
        self._id_sort_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._id_sort_btn.clicked.connect(
            lambda: self._change_confidence_sort_mode("object_id")
        )
        sort_controls.addWidget(self._id_sort_btn)
        title_row.addLayout(sort_controls)
        warnings_header.addLayout(title_row)
        main_layout.addLayout(warnings_header)
        self._update_confidence_sort_buttons()
        self.confidence_issue_list = QListWidget()
        self.confidence_issue_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.confidence_issue_list.setFixedHeight(SIDEBAR_CONFIDENCE_LIST_HEIGHT)
        self.confidence_issue_list.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.confidence_issue_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.confidence_issue_list.setUniformItemSizes(True)
        self.confidence_issue_list.setItemDelegate(
            SidebarConfidenceIssueItemDelegate(self.confidence_issue_list)
        )
        self.confidence_issue_list.setStyleSheet(
            "QListWidget::item { padding-left: 0px; }"
        )
        self.confidence_issue_list.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.confidence_issue_list.customContextMenuRequested.connect(
            self._show_confidence_issue_context_menu
        )
        self.confidence_issue_list.model().rowsInserted.connect(self.adjust_list_sizes)
        self.confidence_issue_list.model().rowsRemoved.connect(self.adjust_list_sizes)
        main_layout.addWidget(self.confidence_issue_list)

        # --- Active Frame Tags ---
        main_layout.addWidget(QLabel("Frame Tags:"))
        self.frame_tag_list = QListWidget()
        self.frame_tag_list.setMinimumHeight(100)
        self.frame_tag_list.model().rowsInserted.connect(self.adjust_list_sizes)
        self.frame_tag_list.model().rowsRemoved.connect(self.adjust_list_sizes)
        self.frame_tag_list.installEventFilter(self)
        main_layout.addWidget(self.frame_tag_list)

        self._frame_tag_del = QShortcut(
            QKeySequence(Qt.Key.Key_Delete), self.frame_tag_list
        )
        self._frame_tag_del.setContext(Qt.ShortcutContext.WidgetShortcut)
        self._frame_tag_del.activated.connect(self._delete_selected_frame_tag)

        self.frontend_state.confidenceIssuesChanged.connect(
            self._on_confidence_issues_changed
        )
        self._current_confidence_issues = dict(self.frontend_state.confidence_issues())
        self.refresh_confidence_issue_list()

        # Finalize Scroll Area
        main_layout.addStretch()  # Pushes content to top
        self.scroll_area.setWidget(self.content_widget)
        outer_layout.addWidget(self.scroll_area)

        try:
            current_index = int(self.video_controller.current_index())
        except Exception:
            current_index = 0
        self._refresh_active_frame_tags(current_index)

    def _extract_object_id_from_text(self, text: str) -> str:
        """Extract object ID from list item text in format 'Type (ID: 123)'"""
        # Find the ID part and remove trailing parenthesis
        try:
            id_part = text.split("ID: ")[1]
            return id_part.rstrip(")").strip()
        except IndexError as e:
            raise InvalidObjectIDFormat(
                f"Cannot extract object ID from text: {text}"
            ) from e

    def _item_object_id(self, item: QListWidgetItem) -> str:
        stored_id = item.data(Qt.ItemDataRole.UserRole)
        if stored_id is not None:
            return str(stored_id)
        return self._extract_object_id_from_text(item.text())

    def _on_active_object_selected(self, item):
        # Trigger highlight in the UI
        object_id = self._item_object_id(item)
        self._selected_annotation_object_id = object_id
        self.highlight_selected_object.emit(object_id)
        self.show_object_editor(object_id, expand=False)

    def select_active_object_by_id(self, object_id: str):
        """Select the active object in the list by its ID."""
        self._selected_annotation_object_id = object_id
        with QSignalBlocker(self.active_objects):
            self.active_objects.clearSelection()
            for i in range(self.active_objects.count()):
                item = self.active_objects.item(i)
                if object_id == self._extract_object_id_from_text(item.text()):
                    item.setSelected(True)
                    self.active_objects.setCurrentItem(item)
                    self.active_objects.scrollToItem(item)
                    break

    def refresh_active_objects(
        self,
        active_objects: list[dict],
        confidence_flags: dict[str, str] | None = None,
    ):
        """Refresh the list of active objects and update recent IDs."""
        with QSignalBlocker(self.active_objects):
            self.active_objects.clear()
            flags = confidence_flags or {}
            for item in active_objects:
                obj_name = item.get("name") or ""
                obj_key = item.get("id") or obj_name
                # Display only numeric part of ID
                numeric_id = obj_name.split("-")[-1] if "-" in obj_name else obj_name
                display_text = f'{item.get("type", "Object")} (ID: {numeric_id})'
                list_item = QListWidgetItem(display_text)
                list_item.setData(Qt.ItemDataRole.UserRole, obj_key)

                severity = flags.get(obj_key)
                if severity == "error":
                    list_item.setBackground(SIDEBAR_ERROR_HIGHLIGHT)
                    list_item.setForeground(SIDEBAR_HIGHLIGHT_TEXT_COLOUR)
                elif severity == "warning":
                    list_item.setBackground(SIDEBAR_WARNING_HIGHLIGHT)
                    list_item.setForeground(SIDEBAR_HIGHLIGHT_TEXT_COLOUR)

                self.active_objects.addItem(list_item)

            self.select_active_object_by_id(
                self._selected_annotation_object_id
            )  # Reselect previously selected object if still present

        self.update()

    def refresh_confidence_issue_list(self, frame_override: int | None = None) -> None:
        """Rebuild the list of confidence issues within the configured frame window."""
        if not hasattr(self, "confidence_issue_list"):
            return
        selected_keys = {
            tuple(item.data(Qt.ItemDataRole.UserRole) or ())
            for item in self.confidence_issue_list.selectedItems()
        }
        visible_issues = self._collect_visible_confidence_issues(frame_override)
        with QSignalBlocker(self.confidence_issue_list):
            self.confidence_issue_list.clear()
            for frame_index, object_id, severity in visible_issues:
                display_text = f"Frame {frame_index} – ID: {object_id}"
                entry_key = (frame_index, object_id, severity)
                list_item = QListWidgetItem(display_text)
                list_item.setData(Qt.ItemDataRole.UserRole, entry_key)
                list_item.setIcon(self._confidence_icon_for(severity))
                if entry_key in selected_keys:
                    list_item.setSelected(True)
                self.confidence_issue_list.addItem(list_item)
        self.adjust_list_sizes()

    def _collect_visible_confidence_issues(
        self, frame_override: int | None = None
    ) -> list[tuple[int, str, str]]:
        """Return sorted (frame, object_id, severity) tuples within the visible range."""
        frame_history = max(0, int(self.state.historic_obj_frame_count))
        if frame_override is not None:
            current_frame = int(frame_override)
        else:
            current_index_value = self.video_controller.current_index()
            current_frame = (
                int(current_index_value) if current_index_value is not None else 0
            )
        min_frame = max(0, current_frame - frame_history)
        max_frame = current_frame + frame_history

        entries: list[tuple[int, str, str]] = []
        for frame_key, issues in self._current_confidence_issues.items():
            if isinstance(frame_key, int):
                frame_index = frame_key
            elif isinstance(frame_key, str) and frame_key.isdigit():
                frame_index = int(frame_key)
            else:
                continue
            if frame_index < min_frame or frame_index > max_frame:
                continue
            for issue in issues or []:
                severity = getattr(issue, "severity", None)
                object_id = getattr(issue, "object_id", None)
                if not object_id or severity not in ("warning", "error"):
                    continue
                entries.append((frame_index, str(object_id), severity))

        entries.sort(
            key=self._confidence_issue_sort_key,
            reverse=not self._confidence_sort_ascending,
        )
        return entries

    def _confidence_issue_sort_key(self, value: tuple[int, str, str]):
        if self._confidence_sort_mode == "object_id":
            return self._object_id_sort_value(value[1])
        return value[0]

    def _object_id_sort_value(self, object_id: str) -> tuple[int, int | str]:
        obj = str(object_id)
        if obj.isdigit():
            return (0, int(obj))
        return (1, obj)

    def _change_confidence_sort_mode(self, mode: str) -> None:
        if self._confidence_sort_mode == mode:
            self._confidence_sort_ascending = not self._confidence_sort_ascending
        else:
            self._confidence_sort_mode = mode
            self._confidence_sort_ascending = True
        self._update_confidence_sort_buttons()
        self.refresh_confidence_issue_list()

    def _update_confidence_sort_buttons(self) -> None:
        frame_arrow = ""
        id_arrow = ""
        if self._confidence_sort_mode == "frame":
            frame_arrow = "↑" if self._confidence_sort_ascending else "↓"
        else:
            id_arrow = "↑" if self._confidence_sort_ascending else "↓"
        self._frame_sort_btn.setText(f"Frame {frame_arrow}".strip())
        self._id_sort_btn.setText(f"ID {id_arrow}".strip())
        palette = self.palette()
        apply_issue_sort_button_style(
            self._frame_sort_btn,
            active=self._confidence_sort_mode == "frame",
            palette=palette,
        )
        apply_issue_sort_button_style(
            self._id_sort_btn,
            active=self._confidence_sort_mode == "object_id",
            palette=palette,
        )

    def _confidence_icon_for(self, severity: str) -> QIcon:
        return self._error_icon if severity == "error" else self._warning_icon

    def _create_confidence_icon(self, pixmap) -> QIcon:
        if pixmap is None or pixmap.isNull():
            return QIcon()
        scaled = pixmap.scaled(
            SIDEBAR_CONFIDENCE_ICON_SIZE,
            SIDEBAR_CONFIDENCE_ICON_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        return QIcon(scaled)

    def _show_confidence_issue_context_menu(self, position: QPoint) -> None:
        if not hasattr(self, "confidence_issue_list"):
            return
        item = self.confidence_issue_list.itemAt(position)
        if item is not None and not item.isSelected():
            selection = self.confidence_issue_list.selectionModel()
            if selection is not None:
                selection.select(
                    self.confidence_issue_list.indexFromItem(item),
                    QItemSelectionModel.SelectionFlag.ClearAndSelect,
                )
        if not self.confidence_issue_list.selectedItems():
            return
        menu = QMenu(self)
        mark_resolved_action = menu.addAction("Mark as resolved")
        chosen_action = menu.exec(
            self.confidence_issue_list.viewport().mapToGlobal(position)
        )
        if chosen_action == mark_resolved_action:
            self._mark_selected_confidence_issues_resolved()

    def _mark_selected_confidence_issues_resolved(self) -> None:
        items = self.confidence_issue_list.selectedItems()
        if not items:
            return
        annotator = self.frontend_state.get_current_annotator() or ""
        unique_keys = {
            tuple(item.data(Qt.ItemDataRole.UserRole) or ()) for item in items
        }
        for frame_index, object_id, _severity in unique_keys:
            frame_value = frame_index
            if isinstance(frame_value, str):
                if frame_value.isdigit():
                    frame_value = int(frame_value)
                else:
                    continue
            elif not isinstance(frame_value, int):
                continue
            if not object_id:
                continue
            self.annotation_controller.mark_confidence_resolved(
                frame_value, str(object_id), annotator
            )
        self._request_confidence_issue_refresh()

    def _request_confidence_issue_refresh(self) -> None:
        host_window = self.window()
        refresh = getattr(host_window, "refresh_confidence_issues", None)
        if callable(refresh):
            refresh()

    @pyqtSlot(dict)
    def _on_confidence_issues_changed(self, issues: dict | None) -> None:
        self._current_confidence_issues = dict(issues or {})
        self.refresh_confidence_issue_list()

    def _choose_project_dir(self):
        """Let the user pick a folder containing the video + OpenLabel JSON."""
        path = QFileDialog.getExistingDirectory(self, "Open Project Folder", "")
        if path:
            self.open_project_dir.emit(path)

    def _trigger_quick_save(self):
        """Trigger quick save signal."""
        self.quick_save.emit()

    def _choose_video_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            "",
            "Videos (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
        )
        if path:
            self.open_video.emit(path)

    def _choose_openlabel_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open OpenLabel Config",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if path:
            self.open_config.emit(path)

    def create_bbox(self, video_actors: list[str]):
        """Popup with 'New Object' and 'Link to Existing ID' options."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Object Options")

        layout = QVBoxLayout(dialog)

        # Buttons for new object and linking
        button_layout = QHBoxLayout()

        create_new_obj_bbox_btn = self.create_new_obj_bbox_button(
            video_actors=video_actors, dialog=dialog
        )
        create_existing_obj_bbox_btn = self.create_existing_obj_bbox_button(
            dialog=dialog, object_type=""
        )

        new_obj_btn_layout = QVBoxLayout()
        new_obj_btn_layout.addWidget(QLabel("Create New Object"))
        new_obj_btn_layout.addWidget(create_new_obj_bbox_btn)

        link_obj_btn_layout = QVBoxLayout()
        link_obj_btn_layout.addWidget(QLabel("Link to Existing ID"))
        link_obj_btn_layout.addWidget(create_existing_obj_bbox_btn)

        button_layout.addLayout(new_obj_btn_layout)
        button_layout.addLayout(link_obj_btn_layout)

        layout.addLayout(button_layout)

        # Cancel button
        cancel_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        cancel_box.rejected.connect(dialog.reject)
        layout.addWidget(cancel_box)

        dialog.exec()

    def create_new_obj_bbox_button(self, video_actors: str, dialog: QDialog):
        """Creates a dropdown button for creating a new bbox for a new object."""
        new_obj_bbox_btn = QComboBox()
        new_obj_bbox_btn.setPlaceholderText("Select new object type")
        types = self.annotation_controller.allowed_bbox_types()
        self._add_combo_separator(new_obj_bbox_btn, "— DynamicObject —")
        for label in types.get("DynamicObject", []):
            new_obj_bbox_btn.addItem(label)

        self._add_combo_separator(new_obj_bbox_btn, "— StaticObject —")
        for label in types.get("StaticObject", []):
            new_obj_bbox_btn.addItem(label)

        new_obj_bbox_btn.setCurrentIndex(-1)

        new_obj_bbox_btn.currentTextChanged.connect(
            lambda text: self.start_bbox_drawing.emit(text)
        )
        new_obj_bbox_btn.currentTextChanged.connect(dialog.accept)

        return new_obj_bbox_btn

    def get_recent_frame_object_ids(self):
        """Fetch object IDs from previous frames and update sidebar."""
        current_frame = self.video_controller.current_index()
        return set(
            self.annotation_controller.get_frame_object_ids(
                frame_limit=self.state.historic_obj_frame_count,
                current_frame=current_frame,
            )
        )

    def create_existing_obj_bbox_button(self, dialog: QDialog, object_type: str):
        """Creates a dropdown button for creating a new bbox for an existing object."""
        link_obj_bbox_btn = QComboBox()
        link_obj_bbox_btn.setEditable(True)  # Allow users to type their own ID
        link_obj_bbox_btn.setInsertPolicy(
            QComboBox.InsertPolicy.NoInsert
        )  # Prevent user input from being added to the list
        placeholder_text = "Type or select ID"
        link_obj_bbox_btn.lineEdit().setPlaceholderText(placeholder_text)
        link_obj_bbox_btn.setMinimumWidth(len(placeholder_text) * 10)

        recent_obj_ids = self.get_recent_frame_object_ids()
        unique_ids = set()
        for obj_id in recent_obj_ids:
            unique_ids.add(obj_id)
        link_obj_bbox_btn.addItems(sorted(unique_ids))
        link_obj_bbox_btn.setCurrentIndex(
            -1
        )  # This is done to reset the index back to the placeholder,
        # more info: https://stackoverflow.com/questions/18274508/setplaceholdertext-for-qcombobox

        # Emit signal once editing is finished OR user selects from the dropdown
        def _emit_id():
            text = link_obj_bbox_btn.currentText()
            if text:  # Only emit if something is typed/selected
                self.add_new_bbox_existing_obj.emit(text)
                dialog.accept()

        link_obj_bbox_btn.lineEdit().editingFinished.connect(_emit_id)
        link_obj_bbox_btn.activated.connect(
            lambda _: _emit_id()
        )  # triggered when user selects an item

        return link_obj_bbox_btn

    def adjust_list_sizes(self):
        """Keep lists at min height when empty, let them expand when populated."""
        widgets = (
            self.active_objects,
            self.frame_tag_list,
        )
        for widget in widgets:
            if widget is None:
                continue
            rows = widget.count()
            if rows == 0:
                widget.setMinimumHeight(widget.minimumHeight())
                widget.setMaximumHeight(16777215)
                continue
            row_height = widget.sizeHintForRow(0)
            content_height = row_height * rows + 6
            widget.setMinimumHeight(max(widget.minimumHeight(), content_height))
            widget.setMaximumHeight(16777215)

    def refresh_annotations_list(self):
        """Refresh the list of active annotations."""
        self.active_objects.clear()
        # TODO: Implement annotation data loading from controller
        # self.active_objects.addItems(annotations)

    def _open_frame_tag_dialog(self):
        path = get_ontology_path()
        if path is None:
            QMessageBox.warning(
                self, "No Ontology", "Set an ontology file in settings first."
            )
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Add Frame Tag")
        form = QFormLayout(dlg)

        tag_combo = QComboBox(dlg)
        tag_combo.addItems(self.annotation_controller.allowed_frame_tags())
        form.addRow("Tag:", tag_combo)

        try:
            total = max(0, int(self.project_state_controller.get_frame_count()))
        except Exception:
            total = 0
        try:
            current_index = max(0, int(self.video_controller.current_index()))
        except Exception:
            current_index = 0

        try:
            offset = max(0, int(get_action_interval_offset()))
        except Exception:
            offset = 0

        start_default = max(0, current_index - offset)
        end_default = current_index + offset
        if total > 0:
            end_default = min(total - 1, end_default)

        start_spin = QSpinBox(dlg)
        end_spin = QSpinBox(dlg)
        for spin_box in (start_spin, end_spin):
            spin_box.setRange(0, max(0, total - 1))
            spin_box.setSingleStep(1)
        start_spin.setValue(start_default)
        end_spin.setValue(end_default)
        form.addRow("Start frame:", start_spin)
        form.addRow("End frame:", end_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=dlg,
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)

        if not dlg.exec():
            return

        tag = tag_combo.currentText().strip()
        start = int(start_spin.value())
        end = int(end_spin.value())
        if start > end:
            QMessageBox.warning(
                self, "Invalid range", "Start frame cannot be after end frame."
            )
            return

        snapshot = FrameTagSnapshot(tag_name=tag, start_frame=start, end_frame=end)
        host_window = self.window()
        try:
            if host_window is not None and hasattr(
                host_window, "execute_undoable_command"
            ):
                host_window.execute_undoable_command(AddFrameTagCommand(snapshot))
            else:
                self.annotation_controller.add_frame_tag(tag, start, end)

            current_index = int(self.video_controller.current_index())
            self._refresh_active_frame_tags(current_index)
            QMessageBox.information(self, "Tag added", f"{tag}: {start} → {end}")
        except Exception as exc:
            QMessageBox.critical(self, "Failed to add tag", str(exc))

    def _refresh_active_frame_tags(self, frame_index: int) -> None:
        """
        Rebuild the 'Frame Tags' list to show only tags active at frame_index.
        """
        if not hasattr(self, "frame_tag_list") or self.frame_tag_list is None:
            return
        try:
            active = self.annotation_controller.active_frame_tags(int(frame_index))
        except Exception:
            active = []

        self.frame_tag_list.clear()
        for tag, start, end in active:
            item = QListWidgetItem(f"{tag} [{start}-{end}]")
            item.setData(Qt.ItemDataRole.UserRole, (tag, int(start), int(end)))
            self.frame_tag_list.addItem(item)
        self.refresh_confidence_issue_list(frame_index)

    @pyqtSlot(int)
    def on_frame_changed(self, frame_index: int) -> None:
        """
        Slot called when SeekBar emits frame_changed(int).
        Updates the list to only show tags active at that frame.
        """
        active = []
        try:
            active = self.annotation_controller.active_frame_tags(int(frame_index))
        except Exception:
            pass

        self.frame_tag_list.clear()
        for tag, start, end in active:
            item = QListWidgetItem(f"{tag} [{start}-{end}]")
            item.setData(Qt.ItemDataRole.UserRole, (tag, int(start), int(end)))
            self.frame_tag_list.addItem(item)

    def _add_combo_separator(self, combo: QComboBox, text: str) -> None:
        """Insert a disabled, italic separator item into a combo box."""
        combo.addItem(text)
        idx = combo.count() - 1
        item = combo.model().item(idx)
        if item is None:
            return
        item.setEnabled(False)
        item.setSelectable(False)
        font = item.font() or QFont()
        font.setItalic(True)
        item.setFont(font)

    def reload_bbox_type_combo(self, _: object | None = None) -> None:
        """Repopulate the type combo after the ontology path changes."""
        if not hasattr(self, "_details_type_combo") or self._details_type_combo is None:
            return

        current_text = self._details_type_combo.currentText()
        self._populate_bbox_type_combo_grouped(self._details_type_combo)

        if current_text and not current_text.strip().startswith("—"):
            for idx in range(self._details_type_combo.count()):
                if (
                    self._details_type_combo.itemText(idx) or ""
                ).lower() == current_text.lower():
                    self._details_type_combo.setCurrentIndex(idx)
                    break

    def _populate_bbox_type_combo_grouped(self, type_combo) -> None:
        """
        Group bbox types by ontology category in the combo box.
        """
        try:
            types = self.annotation_controller.allowed_bbox_types()
        except Exception as e:
            QMessageBox.critical(
                self, "Ontology Error", f"Failed to load bbox types.\n{e}"
            )
            types = {"DynamicObject": [], "StaticObject": []}

        type_combo.blockSignals(True)
        type_combo.clear()

        self._add_combo_separator(type_combo, "— DynamicObject —")
        for label in types.get("DynamicObject", []):
            type_combo.addItem(label.lower())

        self._add_combo_separator(type_combo, "— StaticObject —")
        for label in types.get("StaticObject", []):
            type_combo.addItem(label.lower())

        type_combo.blockSignals(False)

    def eventFilter(self, obj, event):
        if obj is self.frame_tag_list and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Delete:
                self._delete_selected_frame_tag()
                event.accept()
                return True
        return super().eventFilter(obj, event)

    def changeEvent(self, event: QEvent) -> None:
        if event.type() == QEvent.Type.PaletteChange:
            self._update_confidence_sort_buttons()
        super().changeEvent(event)

    def _delete_selected_frame_tag(self):
        item = self.frame_tag_list.currentItem()
        if not item:
            return

        payload = item.data(Qt.ItemDataRole.UserRole)
        if not payload:
            QMessageBox.warning(
                self, "Delete Frame Tag", "No tag data on the selected item."
            )
            return

        tag, start, end = payload
        res = QMessageBox.question(
            self,
            "Delete Frame Tag",
            f"Delete {tag} [{start}-{end}]?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if res != QMessageBox.StandardButton.Yes:
            return

        snapshot = FrameTagSnapshot(
            tag_name=tag,
            start_frame=int(start),
            end_frame=int(end),
        )
        host_window = self.window()
        try:
            if host_window is not None and hasattr(
                host_window, "execute_undoable_command"
            ):
                host_window.execute_undoable_command(RemoveFrameTagCommand(snapshot))
            else:
                removed = self.annotation_controller.remove_frame_tag(tag, start, end)
                if not removed:
                    QMessageBox.information(
                        self, "Delete Frame Tag", "Nothing was deleted."
                    )
                    return
            current_index = int(self.video_controller.current_index())
            self._refresh_active_frame_tags(current_index)
        except Exception as exc:
            QMessageBox.warning(self, "Delete Frame Tag", str(exc))

    def show_object_editor(self, object_id: str, *, expand: bool) -> None:
        """
        Enable the details panel and load the object metadata.
        If expand=True, also expand the content.
        """
        if not object_id:
            return
        self._editing_object_id = object_id
        meta = self.annotation_controller.get_object_metadata(object_id)

        self.details_container.setEnabled(True)
        self._details_id_label.setText(object_id)
        self._details_name_edit.blockSignals(True)
        self._details_name_edit.setText(meta.get("name", "") or "")
        self._details_name_edit.blockSignals(False)
        desired_type_lc = (meta.get("type", "") or "").lower()
        self._details_type_combo.blockSignals(True)
        try:
            idx = -1
            for i in range(self._details_type_combo.count()):
                if (
                    self._details_type_combo.itemText(i) or ""
                ).lower() == desired_type_lc:
                    idx = i
                    break
            self._details_type_combo.setCurrentIndex(idx)
        finally:
            self._details_type_combo.blockSignals(False)

        # Load and display relationships for this object
        self._refresh_relationships(object_id)

        if expand and not self._details_toggle.isChecked():
            self._details_toggle.setChecked(True)

    def _refresh_relationships(self, object_id: str):
        """Load and display relationships for this object."""
        if not hasattr(self, "relationships_list") or self.relationships_list is None:
            return

        try:
            # Get relationships for the current object
            relationships = self.annotation_controller.get_object_relationship(
                object_id
            )

            with QSignalBlocker(self.relationships_list):
                self.relationships_list.clear()

                if not relationships:
                    # Show placeholder text when no relationships exist
                    item = QListWidgetItem("No relationships")
                    item.setFlags(Qt.ItemFlag.NoItemFlags)  # Make it non-selectable
                    self.relationships_list.addItem(item)
                    return

                # Add each relationship to the list
                for relation in relationships:
                    # Extract relationship information
                    relation_type = getattr(relation, "type", "Unknown")
                    relation_name = getattr(relation, "name", "")

                    # Get subject and object IDs
                    subjects = getattr(relation, "rdf_subjects", [])
                    objects = getattr(relation, "rdf_objects", [])

                    # Always show "Object B towed-by Object A" format
                    # Find the subject (Object A) and object (Object B) of the relationship
                    subject_id = None
                    object_id = None

                    for subject in subjects:
                        if hasattr(subject, "uid"):
                            subject_id = subject.uid
                            break

                    for obj in objects:
                        if hasattr(obj, "uid"):
                            object_id = obj.uid
                            break

                    # Get metadata for both objects
                    if subject_id:
                        subject_metadata = (
                            self.annotation_controller.get_object_metadata(subject_id)
                        )
                        subject_name = subject_metadata.get("name", "Unknown")

                    if object_id:
                        object_metadata = (
                            self.annotation_controller.get_object_metadata(object_id)
                        )
                        object_name = object_metadata.get("name", "Unknown")

                    display_text = f"{subject_name} {relation_type} {object_name})"

                    list_item = QListWidgetItem(display_text)
                    list_item.setData(Qt.ItemDataRole.UserRole, relation_name)
                    list_item.setToolTip(display_text)  # Add tooltip to show full text
                    self.relationships_list.addItem(list_item)

        except Exception as e:
            self.relationships_list.clear()
            item = QListWidgetItem("Error loading relationships")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.relationships_list.addItem(item)
            raise e

    def hide_object_editor(self) -> None:
        """Collapse, disable, and clear details when nothing is selected."""
        self._editing_object_id = None
        self.details_container.setEnabled(False)
        if self._details_toggle.isChecked():
            self._details_toggle.setChecked(False)

        self._details_id_label.setText("-")
        self._details_name_edit.blockSignals(True)
        self._details_type_combo.blockSignals(True)
        try:
            self._details_name_edit.clear()
            self._details_type_combo.setCurrentIndex(-1)
        finally:
            self._details_name_edit.blockSignals(False)
            self._details_type_combo.blockSignals(False)

    def _on_details_name_edited(self):
        if not self._editing_object_id:
            return
        new_name = (self._details_name_edit.text() or "").strip()
        if not new_name:
            QMessageBox.warning(self, "Invalid name", "Name cannot be empty.")
            try:
                meta = self.annotation_controller.get_object_metadata(
                    self._editing_object_id
                )
                self._details_name_edit.blockSignals(True)
                self._details_name_edit.setText(meta.get("name", "") or "")
            finally:
                self._details_name_edit.blockSignals(False)
            return

        self.annotation_controller.update_object_name(self._editing_object_id, new_name)
        self._refresh_active_objects_after_edit()

    def _on_details_type_changed(self, text: str):
        if not self._editing_object_id:
            return
        chosen = (text or "").strip()
        if not chosen:
            return

        self.annotation_controller.update_object_type(self._editing_object_id, chosen)
        self._refresh_active_objects_after_edit()
        self.object_details_changed.emit()

    def _refresh_active_objects_after_edit(self):
        """Reloads Active Objects list to reflect edited metadata."""

        current_index = int(self.video_controller.current_index())

        with QSignalBlocker(self.active_objects):
            self.active_objects.clear()
            active = self.annotation_controller.get_active_objects(current_index)
            for item in active:
                obj_name = item["name"]
                obj_type = (item["type"] or "").lower()
                numeric_id = obj_name.split("-")[-1] if "-" in obj_name else obj_name
                self.active_objects.addItem(f"{obj_type} (ID: {numeric_id})")

            self.select_active_object_by_id(self._editing_object_id)

    def _on_active_objects_selection_changed(self):
        if self.active_objects.selectedItems():
            return
        self.hide_object_editor()

    def _open_interpolation_dialog(self):
        # Get current frame and total frames
        current_frame = int(self.video_controller.current_index())
        total_frames = int(self.project_state_controller.get_frame_count())

        # Get active objects in current frame
        active_objs = self.annotation_controller.get_active_objects(current_frame)
        if not active_objs:
            QMessageBox.warning(
                self, "No Objects", "No active objects in current frame"
            )
            return

        # Extract object IDs
        obj_ids = [obj["id"] for obj in active_objs if "id" in obj]

        # Create and show dialog
        dialog = InterpolationDialog(
            self, obj_ids, current_frame, total_frames, self.on_interpolate
        )
        dialog.exec()

    def on_interpolate(self, object_id: str, start_frame: int, end_frame: int):
        current_annotator = self.frontend_state.get_current_annotator()
        if end_frame - start_frame <= 1:
            QMessageBox.information(
                self,
                "Interpolate",
                "Select start and end frames with at least one frame in between.",
            )
            return

        host_window = self.window()
        if host_window is not None and hasattr(host_window, "execute_undoable_command"):
            command = InterpolateAnnotationsCommand(
                object_id=str(object_id),
                start_frame=int(start_frame),
                end_frame=int(end_frame),
                annotator=current_annotator,
            )
            host_window.execute_undoable_command(command)
        else:
            self.annotation_controller.interpolate_annotations(
                object_id, start_frame, end_frame, current_annotator
            )
        if host_window is not None:
            try:
                host_window.refresh_confidence_issues()
            except VideoLoadError:
                pass
        try:
            current_index = int(self.video_controller.current_index())
        except Exception:
            current_index = 0
        self._refresh_active_frame_tags(current_index)
        refresh_confidence_list = getattr(self, "refresh_confidence_issue_list", None)
        if callable(refresh_confidence_list):
            refresh_confidence_list(current_index)

    def _open_relationship_dialog(self):
        """Open the relationship creation dialog."""
        # Get current frame objects for the linker widget
        current_frame = int(self.video_controller.current_index())
        current_objects = self.annotation_controller.get_active_objects(current_frame)

        if not current_objects:
            QMessageBox.warning(
                self, "No Objects", "No active objects in current frame"
            )
            return

        self.create_relationship.emit()

    def _on_details_toggle_clicked(self, checked: bool):
        self._details_toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        self._details_content.setVisible(bool(checked))
