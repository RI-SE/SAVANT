# menu.py
from PyQt6.QtGui import QAction


class AppMenu:
    """Owns the menubar and actions; main window passes callbacks in."""

    def __init__(self, window, *, on_new, on_load, on_save, on_settings):
        mb = window.menuBar()

        file_menu = mb.addMenu("File")
        edit_menu = mb.addMenu("Edit")
        help_menu = mb.addMenu("Help")

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

        # expose menus if you want to add more later
        self.file_menu = file_menu
        self.edit_menu = edit_menu
        self.help_menu = help_menu
