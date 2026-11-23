"""Autosave writes unsaved annotations next to the file and offers them back on the next open."""

import os
import shutil
from pathlib import Path

from src.edit.services.autosave_service import AUTOSAVE_DIR, AutosaveService, autosave_path, pending_autosave
from src.edit.services.project_state import ProjectState

ASSET = Path(__file__).parent.parent.parent / "assets" / "Kraklanda_short.json"


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _project(tmp_path):
    target = tmp_path / ASSET.name
    shutil.copy2(ASSET, target)
    return _project_from(target)


def _edit(state):
    state.annotation_config.append_object_bbox(frame_id=999, bbox_coordinates=[10, 10, 5, 5, 0], obj_id="9",
                                               annotator="alice")


def _autosave(state, clock, minutes=5):
    service = AutosaveService(state, lambda: minutes, clock=clock)
    clock.now += minutes * 60
    service.tick()
    service.wait()
    return service


def _make_older(path: Path, than: Path):
    stamp = than.stat().st_mtime - 10
    os.utime(path, (stamp, stamp))


def test_nothing_written_without_changes(tmp_path):
    state, target = _project(tmp_path)
    _autosave(state, FakeClock())
    assert not (tmp_path / AUTOSAVE_DIR).exists()


def test_unsaved_changes_are_autosaved_after_the_interval(tmp_path):
    state, target = _project(tmp_path)
    _edit(state)
    clock = FakeClock()
    service = AutosaveService(state, lambda: 5, clock=clock)
    clock.now += 4 * 60
    service.tick()
    service.wait()
    assert not autosave_path(target).exists()
    clock.now += 60
    service.tick()
    service.wait()
    assert autosave_path(target).exists()
    assert state.has_unsaved_changes()  # autosave does not count as saving


def test_interval_zero_turns_autosave_off(tmp_path):
    state, target = _project(tmp_path)
    _edit(state)
    _autosave(state, FakeClock(), minutes=0)
    assert not autosave_path(target).exists()


def test_manual_save_deletes_the_autosave(tmp_path):
    state, target = _project(tmp_path)
    _edit(state)
    _autosave(state, FakeClock())
    state.save_openlabel_config()
    assert not autosave_path(target).exists()
    assert not (tmp_path / AUTOSAVE_DIR).exists()


def test_newer_autosave_is_offered_and_restores_to_the_original_file(tmp_path):
    state, target = _project(tmp_path)
    _edit(state)
    _autosave(state, FakeClock())
    _make_older(target, than=autosave_path(target))

    reopened, _ = _project_from(target)
    autosaved = pending_autosave(reopened)
    assert autosaved is not None
    reopened.restore_config(autosaved)
    assert reopened.open_label_path == str(target)  # saves go to the real file, not to .autosave/
    assert reopened.has_unsaved_changes()
    reopened.save_openlabel_config()
    assert "9" in ProjectState.parse_openlabel_file(str(target)).objects
    assert not autosave_path(target).exists()


def test_autosave_older_than_the_file_is_ignored(tmp_path):
    state, target = _project(tmp_path)
    _edit(state)
    _autosave(state, FakeClock())
    _make_older(autosave_path(target), than=target)
    reopened, _ = _project_from(target)
    assert pending_autosave(reopened) is None


def test_autosave_equal_to_the_file_is_deleted(tmp_path):
    state, target = _project(tmp_path)
    (tmp_path / AUTOSAVE_DIR).mkdir()
    shutil.copy2(target, autosave_path(target))
    _make_older(target, than=autosave_path(target))
    assert pending_autosave(state) is None
    assert not autosave_path(target).exists()


def test_unreadable_autosave_is_ignored(tmp_path):
    state, target = _project(tmp_path)
    (tmp_path / AUTOSAVE_DIR).mkdir()
    autosave_path(target).write_text("{not json")
    _make_older(target, than=autosave_path(target))
    assert pending_autosave(state) is None


def _project_from(target: Path):
    state = ProjectState()
    state.load_openlabel_config(str(target))
    return state, target
