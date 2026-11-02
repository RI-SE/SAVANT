from unittest.mock import MagicMock

import pytest

from edit.frontend.exceptions import FileOperationError, FrontendException

from edit.frontend.utils import project_io


def test_open_openlabel_config_resets_undo_history(monkeypatch, tmp_path):
    for name in (
        "set_active_project_dir",
        "update_tag_options",
        "restore_tag_option_states",
        "persist_current_settings",
    ):
        monkeypatch.setattr(project_io, name, MagicMock())
    main_window = MagicMock()
    main_window.project_state_controller.get_tag_categories.return_value = {}
    main_window.project_state_controller.get_tag_frame_details.return_value = {}

    project_io.open_openlabel_config(main_window, str(tmp_path / "a.json"))

    main_window.project_state_controller.load_openlabel_config.assert_called_once()
    main_window.reset_undo_history.assert_called_once()
    main_window.reset_bbox_edit_state.assert_called_once()


def _patch_project_open(monkeypatch, tmp_path):
    """Stub the module-level steps of opening a project and return those that change state."""
    contents = MagicMock(
        directory=tmp_path, video_files=[tmp_path / "v.mp4"], openlabel_files=[tmp_path / "a.json"]
    )
    monkeypatch.setattr(project_io, "scan_project_directory", MagicMock(return_value=contents))
    monkeypatch.setattr(project_io, "_prefer_openlabel_file", MagicMock(return_value=tmp_path / "a.json"))
    monkeypatch.setattr(project_io, "resolve_ontology_path", MagicMock(return_value=(tmp_path / "o.ttl", None)))
    monkeypatch.setattr(project_io, "ensure_project_config", MagicMock())
    mutators = [
        "set_ontology_path",
        "set_active_project_dir",
        "apply_project_settings",
        "persist_current_settings",
    ]
    for name in mutators:
        monkeypatch.setattr(project_io, name, MagicMock())
    return [getattr(project_io, name) for name in mutators]


def _assert_project_untouched(main_window, mutators):
    for mutator in mutators:
        mutator.assert_not_called()
    main_window.project_state_controller.load_openlabel_config.assert_not_called()
    main_window.video_controller.load_video.assert_not_called()


def test_open_project_changes_nothing_when_config_is_unreadable(monkeypatch, tmp_path):
    """Verify a bad config cannot redirect later saves to the new project's folder."""
    mutators = _patch_project_open(monkeypatch, tmp_path)
    project_io.ensure_project_config.side_effect = FileOperationError("bad config")
    main_window = MagicMock()

    with pytest.raises(FileOperationError):
        project_io.on_open_project_dir(main_window, str(tmp_path))

    _assert_project_untouched(main_window, mutators)


def test_open_project_changes_nothing_when_annotations_are_invalid(monkeypatch, tmp_path):
    """Verify a bad annotation file leaves the open project and its save target in place."""
    mutators = _patch_project_open(monkeypatch, tmp_path)
    main_window = MagicMock()
    main_window.project_state_controller.check_openlabel_file.side_effect = FrontendException("bad file")

    with pytest.raises(FrontendException):
        project_io.on_open_project_dir(main_window, str(tmp_path))

    _assert_project_untouched(main_window, mutators)
    project_io.ensure_project_config.assert_not_called()


def test_open_project_changes_nothing_when_video_is_unreadable(monkeypatch, tmp_path):
    """Verify a bad video does not release the open video or replace the open annotations."""
    mutators = _patch_project_open(monkeypatch, tmp_path)
    main_window = MagicMock()
    main_window.video_controller.check_video.side_effect = FrontendException("bad video")

    with pytest.raises(FrontendException):
        project_io.on_open_project_dir(main_window, str(tmp_path))

    _assert_project_untouched(main_window, mutators)
    project_io.ensure_project_config.assert_not_called()


def _write(path, text="x"):
    path.write_text(text)
    return path


def test_import_selected_in_folder_file_keeps_it_and_drops_others(tmp_path):
    keep = _write(tmp_path / "a.json", "keep")
    other = _write(tmp_path / "b.json")

    result = project_io.import_openlabel_file(tmp_path, keep, replace=[keep, other])

    assert result == keep and keep.read_text() == "keep"
    assert not other.exists()


def test_import_replaces_same_named_file(tmp_path):
    old = _write(tmp_path / "a.json", "old")
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    source = _write(source_dir / "a.json", "new")

    project_io.import_openlabel_file(tmp_path, source, replace=[old])

    assert old.read_text() == "new"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.json", "src"]


def test_failed_import_keeps_existing_files(tmp_path):
    old = _write(tmp_path / "a.json")

    with pytest.raises(FrontendException):
        project_io.import_openlabel_file(tmp_path, tmp_path / "missing.json", replace=[old])

    assert old.exists()


def test_template_replaces_existing_files(tmp_path):
    old = _write(tmp_path / f"{tmp_path.name}_openlabel.json", "old")
    other = _write(tmp_path / "other.json")

    path = project_io.create_openlabel_template(tmp_path, replace=[old, other])

    assert path == old and "openlabel" in path.read_text()
    assert not other.exists()


def test_ontology_reference_outside_project_is_ignored(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    (tmp_path / "outside.ttl").write_text("")
    (project / "inside.ttl").write_text("")

    assert project_io._resolve_local_path("inside.ttl", project) == (project / "inside.ttl").resolve()
    assert project_io._resolve_local_path("../outside.ttl", project) is None
    assert project_io._resolve_local_path(str(tmp_path / "outside.ttl"), project) is None
