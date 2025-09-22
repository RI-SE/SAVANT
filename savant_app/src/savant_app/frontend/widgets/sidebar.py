from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QListWidget,
    QLabel,
    QMenu,
    QDialog,
    QSizePolicy,
    QDialogButtonBox,
    QLineEdit,
    QHBoxLayout,
    QFileDialog,
    QComboBox,  # Added for recent objects dropdown
)
from PyQt6.QtCore import QSize, pyqtSignal
from PyQt6.QtGui import QAction
from collections import deque  # For tracking recent objects
from savant_app.frontend.utils.assets import icon


class Sidebar(QWidget):

    open_video = pyqtSignal(str)
    open_config = pyqtSignal(str)
    # TODO: Rename to add_new_bbox_new_obj
    start_bbox_drawing = pyqtSignal(str)
    add_new_bbox_existing_obj = pyqtSignal(str)
    open_project_dir = pyqtSignal(str)
    quick_save = pyqtSignal()

    def __init__(self, video_actors: list[str]):
        super().__init__()
        self.setFixedWidth(200)
        main_layout = QVBoxLayout()
        
        # Track recent object IDs (last 10 frames)
        self.recent_object_ids = deque(maxlen=10)

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
        #bbox_menu = QMenu()

        # TODO - Get labels from config file

        #for label in video_actors:
        #    action = QAction(label, self)
        #    action.triggered.connect(
        #        lambda checked, l=label: self.open_object_options_popup(l)  # noqa: E741
        #    )
        #    bbox_menu.addAction(action)

        #action = QAction(self)
        #action.triggered.connect(
        #lambda checked: self.open_object_options_popup(l)  # noqa: E741
        #)
        #bbox_menu.addAction(action)
        #bbox_menu.aboutToShow.connect(
        #    lambda: bbox_menu.setMinimumWidth(new_bbox_btn.width())
        #)
        #new_bbox_btn.setMenu(bbox_menu)
        new_bbox_btn.clicked.connect(lambda: self.create_bbox(video_actors=video_actors))
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
        
        # Add current IDs to recent objects
        self.recent_object_ids.append(current_ids)
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
        #layout.addWidget(QLabel(f"Selected Type: {object_type}"))

        # Buttons for new object and linking
        button_layout = QHBoxLayout()

        create_new_obj_bbox_btn = self.create_new_obj_bbox_button(video_actors=video_actors, dialog=dialog)
        create_existing_obj_bbox_btn = self.create_existing_obj_bbox_button(dialog=dialog, object_type="") 

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

        # Drawing logic comes after the bbox selected

        # Connect new object button to start drawing
    
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
    
    def create_existing_obj_bbox_button(self, dialog: QDialog, object_type: str):
        """Creates a dropdown button for creating a new bbox for an existing object."""
        link_obj_bbox_btn = QComboBox()
        link_obj_bbox_btn.setEditable(True) # Allow users to type their own ID
        link_obj_bbox_btn.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)  # Prevent user input from being added to the list
        placeholder_text = "Type or select ID"
        link_obj_bbox_btn.lineEdit().setPlaceholderText(placeholder_text)

        unique_ids = set()
        for frame_ids in self.recent_object_ids:
            for obj_id in frame_ids:
                unique_ids.add(obj_id)
        link_obj_bbox_btn.addItems(sorted(unique_ids))
        link_obj_bbox_btn.setCurrentIndex(-1) # This is done to reset the index back to the placeholder. https://stackoverflow.com/questions/18274508/setplaceholdertext-for-qcombobox

        # Ensure the widget itself has enough width
        placeholder_len = len(placeholder_text)
        max_item_len = max((len(i) for i in unique_ids), default=0)
        link_obj_bbox_btn.setMinimumContentsLength(max(placeholder_len, max_item_len))
        link_obj_bbox_btn.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)

        link_obj_bbox_btn.currentTextChanged.connect(
            lambda text: self.add_new_bbox_existing_obj.emit(text)
        )
        link_obj_bbox_btn.currentTextChanged.connect(dialog.accept)

        return link_obj_bbox_btn


    """
    def link_to_existing(self, dialog, object_type, object_id):
        object_id = object_id.strip()
        if object_id:
            self.active_objects.addItem(f"{object_type} (ID: {object_id})")
        dialog.accept()
    """

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
