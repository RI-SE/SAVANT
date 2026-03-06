# menu.py
from PyQt6.QtGui import QAction, QKeySequence


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
        on_exit,
        on_undo,
        on_redo,
        on_new_bbox,
        on_new_frame_tag,
        on_interpolate,
        on_create_relationship,
        on_change_annotator,
        on_bookmarks,
        on_vlm_analysis,
        on_shortcuts,
        on_about,
    ):

        mb = window.menuBar()

        file_menu = mb.addMenu("File")
        edit_menu = mb.addMenu("Edit")

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
        file_menu.addSeparator()

        self.exit_action = QAction("Exit", window)
        self.exit_action.setShortcut("Ctrl+Q")
        self.exit_action.triggered.connect(on_exit)
        file_menu.addAction(self.exit_action)

        self.undo_action = QAction("Undo", window)
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_action.setEnabled(False)
        self.undo_action.triggered.connect(on_undo)

        self.redo_action = QAction("Redo", window)
        self.redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        self.redo_action.setEnabled(False)
        self.redo_action.triggered.connect(on_redo)

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

        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.new_bbox_action)
        edit_menu.addAction(self.new_frame_tag_action)
        edit_menu.addAction(self.interpolate_action)
        edit_menu.addAction(self.create_relationship_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.change_annotator_action)

        # View menu
        view_menu = mb.addMenu("View")
        self.bookmarks_action = QAction("Bookmarks...", window)
        self.bookmarks_action.triggered.connect(on_bookmarks)
        view_menu.addAction(self.bookmarks_action)

        self.vlm_analysis_action = QAction("VLM Analysis...", window)
        self.vlm_analysis_action.triggered.connect(on_vlm_analysis)
        self.vlm_analysis_action.setEnabled(False)  # Disabled until VLM data loaded
        view_menu.addAction(self.vlm_analysis_action)

        # Help menu
        help_menu = mb.addMenu("Help")
        self.shortcuts_action = QAction("Keyboard Shortcuts...", window)
        self.shortcuts_action.triggered.connect(on_shortcuts)
        help_menu.addAction(self.shortcuts_action)

        help_menu.addSeparator()

        about_action = QAction("About", window)
        about_action.triggered.connect(on_about)
        help_menu.addAction(about_action)

        # expose menus if you want to add more later
        self.file_menu = file_menu
        self.view_menu = view_menu
        self.help_menu = help_menu
