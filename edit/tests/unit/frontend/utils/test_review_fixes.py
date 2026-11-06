"""Regression tests for plain-text message boxes and out-of-range bookmarks."""

import importlib
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

from edit.frontend.main_window import MainWindow  # noqa: E402
from edit.frontend.utils import (  # noqa: E402
    _annotation_helpers,
    annotation_ops,
    playback,
    project_config,
    settings_store,
)
from edit.frontend.widgets import bookmark_dialog  # noqa: E402
from edit.frontend.widgets.sidebar import Sidebar  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qapp():
    """Provide the QApplication that widgets require."""
    yield QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def reset_bookmarks():
    """Ensure each test starts with fresh bookmark state, also in modules holding its functions."""
    importlib.reload(settings_store)
    importlib.reload(project_config)
    importlib.reload(bookmark_dialog)
    importlib.reload(playback)
    yield
    settings_store.set_bookmarks({})
    project_config.set_active_project_dir(None)


def _stored_bookmarks(project_dir):
    return project_config.load_project_config(project_dir).settings["bookmarks"]


@pytest.mark.parametrize(
    "clicked, expected",
    [(QMessageBox.StandardButton.Yes, True), (QMessageBox.StandardButton.No, False)],
)
def test_plain_question_renders_literally_and_defaults_to_no(monkeypatch, clicked, expected):
    """Verify object IDs cannot be rendered as HTML and an unconfirmed question is refused."""
    seen = {}

    def _exec(box):
        seen.update(text_format=box.textFormat(), default=box.standardButton(box.defaultButton()))
        return clicked

    monkeypatch.setattr(QMessageBox, "exec", _exec)
    assert _annotation_helpers._plain_question(None, "Title", "<a href='http://e.com'>id</a>") is expected
    assert seen["text_format"] == Qt.TextFormat.PlainText
    assert seen["default"] == QMessageBox.StandardButton.No


def _bookmark_window(current: int, frame_count: int):
    return SimpleNamespace(
        video_controller=MagicMock(
            current_index=MagicMock(return_value=current),
            jump_to_frame=MagicMock(return_value=("pixmap", 0)),
        ),
        project_state_controller=MagicMock(get_frame_count=MagicMock(return_value=frame_count)),
    )


def test_jump_to_bookmark_skips_frames_past_end(monkeypatch):
    """Verify wrap-around lands on a reachable bookmark rather than one past the video end."""
    monkeypatch.setattr(playback, "show_frame", MagicMock())
    settings_store.set_bookmarks({2: "", 500: ""})
    window = _bookmark_window(current=5, frame_count=10)
    playback._jump_to_bookmark(window, 1)
    window.video_controller.jump_to_frame.assert_called_once_with(2)


def test_jump_to_bookmark_does_nothing_when_all_past_end():
    """Verify no seek is attempted when every bookmark is out of range."""
    settings_store.set_bookmarks({500: ""})
    window = _bookmark_window(current=5, frame_count=10)
    playback._jump_to_bookmark(window, 1)
    window.video_controller.jump_to_frame.assert_not_called()


def test_bookmark_dialog_flags_past_end_and_blocks_jump():
    """Verify past-end bookmarks stay visible and deletable but cannot be jumped to."""
    settings_store.set_bookmarks({2: "ok", 500: "stale"})
    dialog = bookmark_dialog.BookmarkManagerDialog(frame_count=10)
    assert dialog.list_widget.item(0).text() == "Frame 2  —  ok"
    assert dialog.list_widget.item(1).text() == "Frame 500  —  stale  (past end of video)"

    dialog.list_widget.setCurrentRow(1)
    assert not dialog.jump_button.isEnabled()
    dialog._on_jump()
    assert dialog.selected_frame is None
    assert settings_store.get_bookmark_notes()[500] == "stale"

    dialog.list_widget.setCurrentRow(0)
    assert dialog.jump_button.isEnabled()
    dialog._on_jump()
    assert dialog.selected_frame == 2


def test_toggle_bookmark_saves_immediately(tmp_path, monkeypatch):
    """Verify a toggled bookmark reaches disk without a project save."""
    monkeypatch.setattr(playback, "apply_bookmark_markers", MagicMock())
    project_config.set_active_project_dir(tmp_path)
    window = _bookmark_window(current=3, frame_count=10)

    playback._toggle_bookmark(window)
    assert _stored_bookmarks(tmp_path) == {"3": ""}

    playback._toggle_bookmark(window)
    assert _stored_bookmarks(tmp_path) == {}


def test_bookmark_dialog_saves_note_and_delete_immediately(tmp_path):
    """Verify note edits and deletions reach disk without a project save."""
    project_config.set_active_project_dir(tmp_path)
    settings_store.set_bookmarks({2: "", 5: ""})
    dialog = bookmark_dialog.BookmarkManagerDialog(frame_count=10)

    dialog.list_widget.setCurrentRow(0)
    dialog.note_edit.setText("edited")
    dialog._on_note_changed()
    assert _stored_bookmarks(tmp_path) == {"2": "edited", "5": ""}

    dialog._on_delete()
    assert _stored_bookmarks(tmp_path) == {"5": ""}


def test_reset_bbox_edit_state_drops_previous_project_references():
    """Verify repeat deltas and selection cannot carry over to same-named objects in a new project."""
    window = SimpleNamespace(
        last_bbox_deltas={"1": (1, 1, 0, 0, 0)},
        _delta_base={(0, "1"): object()},
        overlay=MagicMock(),
        sidebar=MagicMock(),
    )

    MainWindow.reset_bbox_edit_state(window)

    assert window.last_bbox_deltas == {} and window._delta_base == {}
    window.overlay.clear_selection.assert_called_once()
    window.sidebar.clear_object_selection.assert_called_once()


def _window_with_changes(monkeypatch, reply, dirty_after_save=False):
    monkeypatch.setattr(QMessageBox, "question", MagicMock(return_value=reply))
    controller = MagicMock()
    controller.has_unsaved_changes.side_effect = [True, dirty_after_save]
    return SimpleNamespace(project_state_controller=controller, quick_save_project=MagicMock())


@pytest.mark.parametrize(
    "reply, dirty_after_save, expected, saved",
    [
        (QMessageBox.StandardButton.Save, False, True, True),
        (QMessageBox.StandardButton.Save, True, False, True),
        (QMessageBox.StandardButton.Discard, False, True, False),
        (QMessageBox.StandardButton.Cancel, False, False, False),
    ],
)
def test_resolve_unsaved_changes(monkeypatch, reply, dirty_after_save, expected, saved):
    """Verify replacing a project proceeds only after a successful save or an explicit discard."""
    window = _window_with_changes(monkeypatch, reply, dirty_after_save)
    assert MainWindow.resolve_unsaved_changes(window, "Q?") is expected
    assert window.quick_save_project.called is saved


def test_resolve_unsaved_changes_does_not_prompt_when_clean(monkeypatch):
    """Verify no prompt is shown when there is nothing to lose."""
    question = MagicMock()
    monkeypatch.setattr(QMessageBox, "question", question)
    controller = MagicMock(has_unsaved_changes=MagicMock(return_value=False))
    window = SimpleNamespace(project_state_controller=controller)
    assert MainWindow.resolve_unsaved_changes(window, "Q?") is True
    question.assert_not_called()


@pytest.mark.parametrize("flow", [Sidebar.open_project_folder_dialog, Sidebar.start_new_project_flow])
def test_declining_project_switch_stops_before_folder_picker(monkeypatch, flow):
    """Verify both ways of opening a project stop when the user cancels the unsaved-changes prompt."""
    picker = MagicMock(return_value="")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", picker)
    sidebar = SimpleNamespace(confirm_project_switch=lambda: False, _choose_project_dir=MagicMock())
    flow(sidebar)
    picker.assert_not_called()
    sidebar._choose_project_dir.assert_not_called()


def test_cycle_bbox_surfaces_missing_zoom_function():
    """Verify a missing zoom function raises rather than silently skipping the zoom."""
    box = SimpleNamespace(object_id="1")
    overlay = MagicMock(boxes=(box,), selected_index=None)
    with pytest.raises(AttributeError):
        annotation_ops._cycle_bbox(SimpleNamespace(overlay=overlay), 1)
