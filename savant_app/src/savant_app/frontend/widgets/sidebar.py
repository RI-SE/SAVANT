from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QListWidget,
    QLabel,
    QMenu,
    QDialog,
    QDialogButtonBox,
    QLineEdit,
    QHBoxLayout,
    QFileDialog,
)
from PyQt6.QtCore import QSize, pyqtSignal
from PyQt6.QtGui import QAction
from savant_app.frontend.utils.assets import icon


class Sidebar(QWidget):

    open_video = pyqtSignal(str)
    open_config = pyqtSignal(str)
    start_bbox_drawing = pyqtSignal(str)
    open_project_dir = pyqtSignal(str)

    def __init__(self, video_actors: list[str]):
        super().__init__()
        self.setFixedWidth(200)
        main_layout = QVBoxLayout()

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
        bbox_menu = QMenu()

        # TODO - Get labels from config file
        for label in video_actors:
            action = QAction(label, self)
            action.triggered.connect(
                lambda checked, l=label: self.open_object_options_popup(l)  # noqa: E741
            )
            bbox_menu.addAction(action)

        bbox_menu.aboutToShow.connect(
            lambda: bbox_menu.setMinimumWidth(new_bbox_btn.width())
        )
        new_bbox_btn.setMenu(bbox_menu)
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

    def _choose_project_dir(self):
        """Let the user pick a folder containing the video + OpenLabel JSON."""
        path = QFileDialog.getExistingDirectory(self, "Open Project Folder", "")
        if path:
            self.open_project_dir.emit(path)

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

    def open_object_options_popup(self, object_type):
        """Popup with 'New Object' and 'Link to Existing ID' options."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Object Options")

        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(f"Selected Type: {object_type}"))

        # Buttons for new object and linking
        button_layout = QHBoxLayout()

        new_btn = QPushButton("New Object")

        # Drawing logic comes after the bbox selected

        # Connect new object button to start drawing
        new_btn.clicked.connect(lambda: self.start_bbox_drawing.emit(object_type))
        new_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(new_btn)

        link_btn = QPushButton("Link to ID")
        button_layout.addWidget(link_btn)
        layout.addLayout(button_layout)

        # ID input field (only used for linking)
        id_input = QLineEdit()
        id_input.setPlaceholderText("Type object ID")
        layout.addWidget(id_input)

        # Link button logic
        link_btn.clicked.connect(
            lambda: self.link_to_existing(dialog, object_type, id_input.text())
        )

        # Cancel button
        cancel_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        cancel_box.rejected.connect(dialog.reject)
        layout.addWidget(cancel_box)

        dialog.exec()

    # Method removed to resolve naming conflict with signal
    # Actual object creation now handled via drawing workflow

    def link_to_existing(self, dialog, object_type, object_id):
        """Link object to an existing ID."""
        object_id = object_id.strip()
        if object_id:
            self.active_objects.addItem(f"{object_type} (ID: {object_id})")
        dialog.accept()

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
