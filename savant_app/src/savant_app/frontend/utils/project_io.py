import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtWidgets import QMessageBox

from ..exceptions import (
    FileOperationError,
    FrontendException,
    InvalidConfigFileError,
    InvalidDirectoryError,
    InvalidVideoFileError,
    MissingConfigError,
    MissingVideoError,
    TemplateCreationError,
)
from ..utils.settings_store import (
    get_default_ontology_path,
    get_frame_history_count,
    get_zoom_rate,
    set_ontology_path,
    update_tag_options,
)
from .project_config import (
    apply_project_settings,
    ensure_project_config,
    persist_current_settings,
    record_annotator_login,
    restore_tag_option_states,
    set_active_project_dir,
)
from .playback import _stop as stop
from .render import show_frame
from savant_app.utils import read_json


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v"}
OPENLABEL_EXTENSIONS = {".json"}


@dataclass(frozen=True)
class ProjectDirectoryContents:
    """Describes which project-critical files live inside a directory."""

    directory: Path
    video_files: list[Path]
    openlabel_files: list[Path]

    @property
    def has_video(self) -> bool:
        return bool(self.video_files)

    @property
    def has_openlabel(self) -> bool:
        return bool(self.openlabel_files)

    @property
    def single_video(self) -> Path | None:
        if len(self.video_files) == 1:
            return self.video_files[0]
        return None

    @property
    def single_openlabel(self) -> Path | None:
        if len(self.openlabel_files) == 1:
            return self.openlabel_files[0]
        return None


def scan_project_directory(dir_path: str | Path) -> ProjectDirectoryContents:
    """Inspect a directory and report which video and JSON files exist."""

    folder = Path(dir_path).expanduser()
    if not folder.is_dir():
        raise InvalidDirectoryError(f"Not a directory: {dir_path}")

    video_files = sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ],
        key=lambda path: path.name.lower(),
    )
    json_files = sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in OPENLABEL_EXTENSIONS
        ],
        key=lambda path: path.name.lower(),
    )
    return ProjectDirectoryContents(
        directory=folder, video_files=video_files, openlabel_files=json_files
    )


def _prefer_openlabel_file(json_files: list[Path]) -> Path:
    """Return the best openlabel candidate from the provided list."""

    if not json_files:
        raise MissingConfigError("No JSON (OpenLabel) found in folder.")
    preferred = [path for path in json_files if "openlabel" in path.stem.lower()]
    return preferred[0] if preferred else json_files[0]


def import_video_file(destination_dir: Path, source_path: str | Path) -> Path:
    """Copy a user-provided video file into the project directory."""

    return _copy_file_into_project(
        destination_dir=destination_dir,
        source_path=source_path,
        allowed_extensions=VIDEO_EXTENSIONS,
        error_type=InvalidVideoFileError,
        error_message="Unsupported video format selected.",
    )


def import_openlabel_file(destination_dir: Path, source_path: str | Path) -> Path:
    """Copy a user-provided OpenLabel file into the project directory."""

    return _copy_file_into_project(
        destination_dir=destination_dir,
        source_path=source_path,
        allowed_extensions=OPENLABEL_EXTENSIONS,
        error_type=InvalidConfigFileError,
        error_message="OpenLabel configs must be JSON files.",
    )


def create_openlabel_template(
    destination_dir: Path,
    *,
    video_filename: str | None = None,
    template_name: str | None = None,
) -> Path:
    """Create a minimal OpenLabel template file in the project directory."""

    folder = Path(destination_dir).expanduser()
    if not folder.is_dir():
        raise TemplateCreationError("Project directory no longer exists.")

    default_name = template_name or f"{folder.name}_openlabel.json"
    destination_path = folder / default_name
    if destination_path.exists():
        raise TemplateCreationError(
            f"The file '{destination_path.name}' already exists in this project."
        )

    template_payload = {
        "openlabel": {
            "metadata": {
                "schema_version": "1.0.0",
                "tagged_file": video_filename or "",
                "annotator": "",
            },
            "ontologies": {},
            "objects": {},
            "actions": {},
            "relations": {},
            "frames": {},
        }
    }

    try:
        with destination_path.open("w", encoding="utf-8") as output_file:
            json.dump(template_payload, output_file, indent=2)
    except OSError as exc:
        raise TemplateCreationError(f"Failed to create template file: {exc}") from exc

    return destination_path


def _read_openlabel_config(openlabel_path: Path) -> dict:
    try:
        return read_json(str(openlabel_path)) or {}
    except (OSError, ValueError) as exc:
        raise FileOperationError(
            f"Failed to read OpenLabel file '{openlabel_path}': {exc}"
        ) from exc


def discover_ontology_references(openlabel_path: Path) -> list[str]:
    """Return ontology references declared inside an OpenLabel file."""

    config = _read_openlabel_config(openlabel_path)
    openlabel_section = config.get("openlabel", {})
    ontologies = openlabel_section.get("ontologies")
    if not isinstance(ontologies, dict):
        return []
    references: list[str] = []
    for entry in ontologies.values():
        candidate = None
        if isinstance(entry, str):
            candidate = entry.strip()
        elif isinstance(entry, dict):
            uri = entry.get("uri")
            if isinstance(uri, str):
                candidate = uri.strip()
        if candidate:
            references.append(candidate)
    return references


def _resolve_local_path(reference: str, *search_dirs: Path) -> Path | None:
    candidate = Path(reference)
    if candidate.is_absolute() and candidate.is_file():
        return candidate
    for base in search_dirs:
        resolved = (base / reference).resolve()
        if resolved.is_file():
            return resolved
    return None


def resolve_ontology_path(project_dir: Path, openlabel_path: Path) -> tuple[Path, bool]:
    """Resolve an ontology path referenced in OpenLabel or fall back to default.

    Returns (path, used_default_flag).
    """

    references = discover_ontology_references(openlabel_path)
    label_dir = openlabel_path.parent
    for reference in references:
        lowered = reference.lower()
        if lowered.startswith("http://") or lowered.startswith("https://"):
            continue
        local_path = _resolve_local_path(reference, label_dir, project_dir)
        if local_path:
            return local_path, False

    default_path = get_default_ontology_path()
    if default_path is None:
        raise FileOperationError(
            "Default ontology file is not available in the SAVANT assets. "
            "Restore the bundled ontology files to continue."
        )
    return default_path, True


def _copy_file_into_project(
    *,
    destination_dir: Path,
    source_path: str | Path,
    allowed_extensions: set[str],
    error_type: type[FrontendException],
    error_message: str,
) -> Path:
    """Copy a file into the project directory after validation."""

    folder = Path(destination_dir).expanduser()
    if not folder.is_dir():
        raise FileOperationError("Project directory no longer exists.")

    source = Path(source_path).expanduser()
    if not source.is_file():
        raise FileOperationError(f"File not found: {source}")

    if source.suffix.lower() not in allowed_extensions:
        raise error_type(error_message)

    destination = folder / source.name
    if destination.exists():
        try:
            if destination.resolve() == source.resolve():
                # The user selected the already-present file. Nothing to do.
                return destination
        except OSError:
            # Fallback to copying if resolve fails.
            pass
        raise FileOperationError(
            f"{destination.name} already exists in the project directory. "
            "Remove it before importing a replacement."
        )

    try:
        shutil.copy2(source, destination)
    except OSError as exc:
        raise FileOperationError(f"Failed to copy '{source.name}': {exc}") from exc

    return destination


def wire(main_window):
    main_window.sidebar.open_video.connect(lambda p: on_open_video(main_window, p))
    main_window.sidebar.open_config.connect(
        lambda p: open_openlabel_config(main_window, p)
    )
    main_window.sidebar.open_project_dir.connect(
        lambda p, name: on_open_project_dir(main_window, p, name)
    )
    main_window.sidebar.quick_save.connect(lambda: quick_save(main_window))


def on_open_video(main_window, path: str):
    main_window.video_controller.load_video(path)
    pixmap, idx = main_window.video_controller.jump_to_frame(0)
    show_frame(main_window, pixmap, idx)

    frame_count = main_window.project_state_controller.get_frame_count()
    main_window.seek_bar.update_range(frame_count or 0)
    if hasattr(main_window, "apply_confidence_markers"):
        main_window.apply_confidence_markers()

    if hasattr(main_window.playback_controls, "set_fps"):
        main_window.playback_controls.set_fps(
            main_window.project_state_controller.get_fps()
        )
    stop(main_window)


def open_openlabel_config(main_window, path: str):
    main_window.project_state_controller.load_openlabel_config(path)
    try:
        set_active_project_dir(Path(path).expanduser().resolve().parent)
    except OSError:
        set_active_project_dir(Path(path).parent)
    tag_map = main_window.project_state_controller.get_tag_categories() or {}
    update_tag_options(tag_map)
    restore_tag_option_states()
    tag_details = main_window.project_state_controller.get_tag_frame_details() or {}
    if hasattr(main_window, "state"):
        main_window.state.set_frame_tag_details(tag_details)
    if hasattr(main_window, "refresh_confidence_issues"):
        main_window.refresh_confidence_issues()
    persist_current_settings()


def quick_save(main_window):
    main_window.project_state_controller.validate_before_save()
    main_window.project_state_controller.save_openlabel_config()
    tag_map = main_window.project_state_controller.get_tag_categories() or {}
    update_tag_options(tag_map)
    tag_details = main_window.project_state_controller.get_tag_frame_details() or {}
    if hasattr(main_window, "state"):
        main_window.state.set_frame_tag_details(tag_details)
    should_save_settings = QMessageBox.question(
        main_window,
        "Save Settings?",
        "Save the current application settings to this project's config?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.Yes,
    )
    if should_save_settings == QMessageBox.StandardButton.Yes:
        persist_current_settings()
    QMessageBox.information(
        main_window, "Save Successful", "Project saved successfully."
    )


def on_open_project_dir(main_window, dir_path: str, project_name: str | None = None):
    contents = scan_project_directory(dir_path)
    if not contents.video_files:
        raise MissingVideoError("No video found in folder.")
    if not contents.openlabel_files:
        raise MissingConfigError("No JSON (OpenLabel) found in folder.")
    json_path = _prefer_openlabel_file(contents.openlabel_files)
    video_path = contents.video_files[0]

    set_active_project_dir(contents.directory)
    config = ensure_project_config(contents.directory)
    apply_project_settings(config)
    if hasattr(main_window, "set_default_zoom"):
        main_window.set_default_zoom(get_zoom_rate(), apply=True)
    if hasattr(main_window, "sidebar_state"):
        main_window.sidebar_state.historic_obj_frame_count = (
            get_frame_history_count()
        )

    resolved_name = (project_name or contents.directory.name).strip()
    if resolved_name:
        set_project_name = getattr(main_window, "set_project_name", None)
        if callable(set_project_name):
            set_project_name(resolved_name)
        else:
            main_window.project_name = resolved_name
            main_window.update_title()

    try:
        ontology_path, _ = resolve_ontology_path(contents.directory, json_path)
    except FrontendException as exc:
        QMessageBox.critical(main_window, "Ontology Error", str(exc))
        return
    try:
        set_ontology_path(ontology_path)
    except ValueError as exc:
        QMessageBox.critical(main_window, "Invalid Ontology", str(exc))
        return

    annotator_name = ""
    if hasattr(main_window, "state"):
        getter = getattr(main_window.state, "get_current_annotator", None)
        if callable(getter):
            annotator_name = getter() or ""
    if annotator_name:
        record_annotator_login(annotator_name)

    open_openlabel_config(main_window, str(json_path))
    # TODO: Refactor the getattr stuff?
    if (
        getattr(main_window.project_state_controller, "project_state", None)
        and getattr(
            main_window.project_state_controller.project_state,
            "annotation_config",
            None,
        )
        is None
    ):
        return
    on_open_video(main_window, str(video_path))
    persist_current_settings()
