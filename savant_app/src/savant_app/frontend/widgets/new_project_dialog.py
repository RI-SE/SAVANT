"""Dialog that displays what the selected project directory already contains."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLayout,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from savant_app.frontend.exceptions import FrontendException
from savant_app.frontend.utils.project_io import (
    ProjectDirectoryContents,
    create_openlabel_template,
    import_openlabel_file,
    import_video_file,
    resolve_ontology_path,
    scan_project_directory,
)
from savant_app.frontend.utils.settings_store import (
    set_ontology_path,
)


@dataclass
class _FileSection:
    """Keeps UI widgets grouped for easy updates."""

    title: str
    missing_hint: str
    container: QWidget
    header_label: QLabel
    detail_label: QLabel
    action_layout: QHBoxLayout


class NewProjectDialog(QDialog):
    project_ready = pyqtSignal(str, str)
    """Simple progress dialog for the staged project creation flow."""

    def __init__(
        self,
        contents: ProjectDirectoryContents,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Project")
        self._contents = contents

        layout = QVBoxLayout(self)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

        folder_header = QLabel("Selected project folder:")
        header_font = QFont()
        header_font.setBold(True)
        folder_header.setFont(header_font)
        layout.addWidget(folder_header)

        self._path_label = QLabel(str(contents.directory))
        path_font = QFont()
        path_font.setItalic(True)
        self._path_label.setFont(path_font)
        self._path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._path_label.setWordWrap(False)
        self._path_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        layout.addWidget(self._path_label)

        name_row = QHBoxLayout()
        name_label = QLabel("Project name:")
        name_label.setMinimumWidth(110)
        self._project_name_edit = QLineEdit(contents.directory.name, self)
        self._project_name_edit.setPlaceholderText(contents.directory.name)
        name_row.addWidget(name_label)
        name_row.addWidget(self._project_name_edit, stretch=1)
        layout.addLayout(name_row)

        sections_layout = QHBoxLayout()
        self._sections_layout = sections_layout

        self._video_section = self._create_file_section(
            title="Video file:",
            files=contents.video_files,
            empty_hint="No video detected.",
        )
        self._openlabel_section = self._create_file_section(
            title="OpenLabel file:",
            files=contents.openlabel_files,
            empty_hint="No OpenLabel detected.",
        )

        self._video_select_button = self._create_action_button(
            "Select video…", self._prompt_video_selection
        )
        self._video_section.action_layout.addWidget(self._video_select_button)

        self._openlabel_resolve_button = self._create_action_button(
            "Resolve…", self._prompt_openlabel_resolution
        )
        self._openlabel_section.action_layout.addWidget(self._openlabel_resolve_button)

        sections_layout.addWidget(self._video_section.container)
        sections_layout.addWidget(self._openlabel_section.container)
        layout.addLayout(sections_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        button_box.accepted.connect(self.accept)
        self._load_project_button = QPushButton("Load project", self)
        self._load_project_button.setEnabled(False)
        self._load_project_button.clicked.connect(self._on_load_project_clicked)
        button_box.addButton(
            self._load_project_button, QDialogButtonBox.ButtonRole.AcceptRole
        )
        layout.addWidget(button_box)
        self._update_project_ready_state()

    @property
    def current_directory(self) -> Path:
        return self._contents.directory

    def update_contents(self, contents: ProjectDirectoryContents) -> None:
        """Refresh the dialog after the user picks a new folder."""

        self._contents = contents
        self._path_label.setText(str(contents.directory))
        self._project_name_edit.blockSignals(True)
        self._project_name_edit.setText(contents.directory.name)
        self._project_name_edit.setPlaceholderText(contents.directory.name)
        self._project_name_edit.blockSignals(False)
        self._populate_file_section(self._video_section, contents.video_files)
        self._populate_file_section(self._openlabel_section, contents.openlabel_files)
        self._update_project_ready_state()

    def _create_file_section(
        self, title: str, files: list[Path], empty_hint: str
    ) -> _FileSection:
        container = QWidget(self)
        container_layout = QVBoxLayout(container)

        header_label = QLabel(title)
        header_font = QFont()
        header_font.setBold(True)
        header_label.setFont(header_font)
        container_layout.addWidget(header_label)

        detail_label = QLabel()
        detail_font = QFont()
        detail_font.setItalic(True)
        detail_label.setFont(detail_font)
        detail_label.setWordWrap(False)
        detail_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        container_layout.addWidget(detail_label)

        actions_layout = QHBoxLayout()
        container_layout.addLayout(actions_layout)

        section = _FileSection(
            title=title,
            missing_hint=empty_hint,
            container=container,
            header_label=header_label,
            detail_label=detail_label,
            action_layout=actions_layout,
        )
        self._populate_file_section(section, files)
        return section

    def _populate_file_section(self, section: _FileSection, files: list[Path]) -> None:
        if not files:
            section.detail_label.setText(section.missing_hint)
            section.detail_label.setToolTip("")
            return

        if len(files) > 1:
            warning = f"Multiple files detected ({len(files)}). Keep exactly one."
            section.detail_label.setText(warning)
            section.detail_label.setToolTip(warning)
            return

        file_path = files[0]
        section.detail_label.setText(file_path.name)
        section.detail_label.setToolTip(str(file_path))
        section.detail_label.setWordWrap(False)

    def _update_project_ready_state(self) -> None:
        """Enable the load button once both assets are available."""

        has_single_video = self._contents.single_video is not None
        has_single_config = self._contents.single_openlabel is not None
        ready = has_single_video and has_single_config
        self._load_project_button.setEnabled(ready)
        if not ready:
            self._load_project_button.setToolTip(
                "Select both a video and an OpenLabel file to continue."
            )
        else:
            self._load_project_button.setToolTip("")

    def _on_load_project_clicked(self) -> None:
        """Notify listeners that the project folder is ready to load."""

        if not self._load_project_button.isEnabled():
            return
        if not self._apply_project_name_change():
            return
        openlabel_file = self._contents.single_openlabel
        if openlabel_file is None:
            QMessageBox.critical(
                self, "Missing OpenLabel", "OpenLabel file must be present."
            )
            return
        try:
            ontology_path, used_default = resolve_ontology_path(
                self._contents.directory, openlabel_file
            )
        except FrontendException as exc:
            QMessageBox.critical(self, "Ontology Error", str(exc))
            return
        try:
            set_ontology_path(ontology_path)
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid Ontology", str(exc))
            return
        if used_default:
            QMessageBox.information(
                self,
                "Default Ontology Loaded",
                "Could not access the ontology referenced in the OpenLabel file.\n\n"
                "The bundled default ontology was loaded automatically. Update the "
                "OpenLabel file if you need to reference a different ontology.",
            )
        directory_path = str(self._contents.directory)
        self.project_ready.emit(directory_path, self._project_name_edit.text().strip())
        self.accept()

    def _prompt_video_selection(self) -> None:
        """Ask the user to provide a video file for the project."""

        video_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Project Video",
            str(self._contents.directory),
            "Videos (*.mp4 *.avi *.mov *.mkv *.mpg *.mpeg *.m4v);;All Files (*)",
        )
        if not video_path:
            return

        if self._contents.video_files and not self._confirm_replacement(
            "video", self._contents.video_files
        ):
            return
        if self._contents.video_files and not self._delete_existing_files(
            self._contents.video_files, "video"
        ):
            return

        try:
            import_video_file(self._contents.directory, video_path)
        except FrontendException as exc:
            QMessageBox.critical(self, "Video Import Failed", str(exc))
            return

        self._rescan_directory()

    def _prompt_openlabel_resolution(self) -> None:
        """Ask if the user already has an OpenLabel file."""

        question = QMessageBox.question(
            self,
            "OpenLabel File",
            "Do you already have an OpenLabel file for this project?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if question == QMessageBox.StandardButton.Yes:
            self._prompt_openlabel_import()
        else:
            self._generate_openlabel_template()

    def _prompt_openlabel_import(self) -> None:
        """Import an existing OpenLabel file into the project directory."""

        config_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select OpenLabel Config",
            str(self._contents.directory),
            "JSON Files (*.json);;All Files (*)",
        )
        if not config_path:
            return

        if self._contents.openlabel_files and not self._confirm_replacement(
            "OpenLabel", self._contents.openlabel_files
        ):
            return
        if self._contents.openlabel_files and not self._delete_existing_files(
            self._contents.openlabel_files, "OpenLabel"
        ):
            return

        try:
            import_openlabel_file(self._contents.directory, config_path)
        except FrontendException as exc:
            QMessageBox.critical(self, "OpenLabel Import Failed", str(exc))
            return

        self._rescan_directory()

    def _generate_openlabel_template(self) -> None:
        """Create a blank OpenLabel file if the user does not have one."""

        if self._contents.openlabel_files and not self._confirm_replacement(
            "OpenLabel", self._contents.openlabel_files
        ):
            return
        if self._contents.openlabel_files and not self._delete_existing_files(
            self._contents.openlabel_files, "OpenLabel"
        ):
            return

        video_name = None
        if self._contents.single_video:
            video_name = self._contents.single_video.name

        try:
            create_openlabel_template(
                self._contents.directory, video_filename=video_name
            )
        except FrontendException as exc:
            QMessageBox.critical(self, "Template Creation Failed", str(exc))
            return

        self._rescan_directory()

    def _confirm_replacement(self, label: str, files: list[Path]) -> bool:
        """Prompt the user to confirm removal of existing project files."""

        file_list = ", ".join(path.name for path in files)
        result = QMessageBox.question(
            self,
            f"Replace {label} file",
            f"The folder already contains {label} file(s): {file_list}.\n"
            "Do you want to replace them?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        return result == QMessageBox.StandardButton.Yes

    def _delete_existing_files(self, files: list[Path], label: str) -> bool:
        """Remove existing files so the new selection can be stored."""

        for file_path in files:
            try:
                file_path.unlink(missing_ok=True)
            except FileNotFoundError:
                continue
            except OSError as exc:
                QMessageBox.critical(
                    self,
                    f"Could not replace {label}",
                    f"Failed to remove '{file_path.name}': {exc}",
                )
                return False
        return True

    def _rescan_directory(self) -> None:
        """Refresh the dialog state after new files are added."""

        try:
            refreshed = scan_project_directory(self._contents.directory)
        except FrontendException as exc:
            QMessageBox.critical(self, "Project Directory Error", str(exc))
            return
        self.update_contents(refreshed)

    def _create_action_button(self, text: str, handler) -> QPushButton:
        """Create consistently sized buttons for dialog actions."""

        button = QPushButton(text, self)
        button.setMinimumHeight(30)
        button.setStyleSheet("padding: 4px 12px;")
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(handler)
        button.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed
        )
        return button

    def _apply_project_name_change(self) -> bool:
        """Rename the project directory if the user edits the project name."""

        desired_name = (self._project_name_edit.text() or "").strip()
        current_dir = self._contents.directory
        current_name = current_dir.name
        if not desired_name:
            QMessageBox.warning(
                self,
                "Invalid Name",
                "Project name cannot be empty.",
            )
            self._project_name_edit.setText(current_name)
            return False
        if Path(desired_name).name != desired_name:
            QMessageBox.warning(
                self,
                "Invalid Name",
                "Project name must not contain path separators.",
            )
            self._project_name_edit.setText(current_name)
            return False
        if desired_name == current_name:
            return True

        target_dir = current_dir.parent / desired_name
        if target_dir.exists():
            QMessageBox.critical(
                self,
                "Rename Failed",
                f"A folder named '{desired_name}' already exists.",
            )
            self._project_name_edit.setText(current_name)
            return False
        try:
            current_dir = current_dir.rename(target_dir)
        except OSError as exc:
            QMessageBox.critical(
                self,
                "Rename Failed",
                f"Could not rename the folder: {exc}",
            )
            self._project_name_edit.setText(current_name)
            return False

        try:
            refreshed = scan_project_directory(target_dir)
        except FrontendException as exc:
            QMessageBox.critical(self, "Project Directory Error", str(exc))
            return False
        self.update_contents(refreshed)
        return True
