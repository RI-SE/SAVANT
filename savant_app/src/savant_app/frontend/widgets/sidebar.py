from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QListWidget,
    QLabel,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QFileDialog,
    QComboBox,  # Added for recent objects dropdown
)
from PyQt6.QtCore import QSize, pyqtSignal
from savant_app.frontend.utils.assets import icon
from savant_app.controllers.annotation_controller import AnnotationController
from savant_app.controllers.video_controller import VideoController
from savant_app.frontend.states.sidebar_state import SidebarState


class Sidebar(QWidget):

    open_video = pyqtSignal(str)
    open_config = pyqtSignal(str)
    # TODO: Rename to add_new_bbox_new_obj
    start_bbox_drawing = pyqtSignal(str)
    add_new_bbox_existing_obj = pyqtSignal(str)
    open_project_dir = pyqtSignal(str)
    quick_save = pyqtSignal()

    def __init__(
        self,
        video_actors: list[str],
        annotation_controller: AnnotationController,
        video_controller: VideoController,
        state: SidebarState,
    ):
        super().__init__()
        self.setFixedWidth(200)
        main_layout = QVBoxLayout()

        # Temporary controller until refactor.
        self.annotation_controller: AnnotationController = annotation_controller
        self.video_controller: VideoController = video_controller

        # State for sidebar
        self.state: SidebarState = state

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
        #           future ticket when config dir is made.

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
        main_layout.addWidget(QPushButton("New frame tag"))

        # --- Active Objects ---
        main_layout.addWidget(QLabel("Active Objects:"))
        self.active_objects = QListWidget()
        self.active_objects.setMinimumHeight(100)
        self.active_objects.model().rowsInserted.connect(self.adjust_list_sizes)
        self.active_objects.model().rowsRemoved.connect(self.adjust_list_sizes)
        main_layout.addWidget(self.active_objects)

        # --- Frame ID ---
        # main_layout.addWidget(QLabel("Frame ID:"))
        # self.frame_id = QListWidget()
        # self.frame_id.setFixedHeight(40)
        # main_layout.addWidget(self.frame_id)

        # --- Active Manoeuvre ---
        main_layout.addWidget(QLabel("Active Manoeuvre:"))
        self.active_manoeuvre = QListWidget()
        self.active_manoeuvre.setMinimumHeight(60)
        self.active_manoeuvre.model().rowsInserted.connect(self.adjust_list_sizes)
        self.active_manoeuvre.model().rowsRemoved.connect(self.adjust_list_sizes)
        main_layout.addWidget(self.active_manoeuvre)

        self.setLayout(main_layout)

    def refresh_active_objects(self, active_objects: list[str]):
        """Refresh the list of active objects and update recent IDs."""
        self.active_objects.clear()
        current_ids = []
        for item in active_objects:
            obj_id = item["name"]
            self.active_objects.addItem(f'{item["type"]} (ID: {obj_id})')
            current_ids.append(obj_id)

        self.update()

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
        new_obj_bbox_btn.addItems(video_actors)

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
        print(recent_obj_ids)
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
        for widget in [self.active_objects, self.active_manoeuvre]:
            rows = widget.count()
            if rows == 0:
                widget.setMinimumHeight(widget.minimumHeight())
                widget.setMaximumHeight(16777215)
            else:
                content_height = widget.sizeHintForRow(0) * rows + 6
                widget.setMinimumHeight(max(widget.minimumHeight(), content_height))
                widget.setMaximumHeight(16777215)

    def refresh_annotations_list(self):
        """Refresh the list of active annotations."""
        self.active_objects.clear()
        # TODO: Implement annotation data loading from controller
        # self.active_objects.addItems(annotations)
