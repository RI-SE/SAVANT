# menu.py
from PyQt6.QtGui import QAction


class AppMenu:
    """Owns the menubar and actions; main window passes callbacks in."""

    def __init__(
        self,
        window,
        *,
        on_new,
        on_load,
        on_save,
        on_settings,
        on_new_bbox,
        on_new_frame_tag,
        on_interpolate,
        on_create_relationship,
        on_change_annotator,
        on_about,
    ):

        mb = window.menuBar()

        file_menu = mb.addMenu("File")
        edit_menu = mb.addMenu("Edit")
        # Add the "About" action directly to the menubar
        about_action = QAction("About", window)
        about_action.triggered.connect(on_about)
        mb.addAction(about_action)

        self.new_action = QAction("New project", window)
        self.new_action.triggered.connect(on_new)

        self.load_action = QAction("Load project", window)
        self.load_action.triggered.connect(on_load)

        self.save_action = QAction("Save project", window)
        self.save_action.triggered.connect(on_save)

        self.settings_action = QAction("Settings", window)
        self.settings_action.triggered.connect(on_settings)

        file_menu.addAction(self.new_action)
        file_menu.addAction(self.load_action)
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.settings_action)

        self.new_bbox_action = QAction("New bounding box", window)
        self.new_bbox_action.triggered.connect(on_new_bbox)
        self.new_frame_tag_action = QAction("New frame tag", window)
        self.new_frame_tag_action.triggered.connect(on_new_frame_tag)
        self.interpolate_action = QAction("Interpolate change", window)
        self.interpolate_action.triggered.connect(on_interpolate)
        self.create_relationship_action = QAction("Create relationship", window)
        self.create_relationship_action.triggered.connect(on_create_relationship)
        self.change_annotator_action = QAction("Change annotator", window)
        self.change_annotator_action.triggered.connect(on_change_annotator)

        edit_menu.addAction(self.new_bbox_action)
        edit_menu.addAction(self.new_frame_tag_action)
        edit_menu.addAction(self.interpolate_action)
        edit_menu.addAction(self.create_relationship_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.change_annotator_action)

        # expose menus if you want to add more later
        self.file_menu = file_menu
