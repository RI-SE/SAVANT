"""Tests for ProjectController unsaved-change detection."""

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtWidgets import QApplication  # noqa: E402

from trainit.trainit_gui.controllers.project_controller import ProjectController  # noqa: E402
from trainit.trainit_gui.frontend.states.app_state import AppState  # noqa: E402
from trainit.trainit_gui.models.project import TrainingConfig  # noqa: E402
from trainit.trainit_gui.services.dataset_service import DatasetService  # noqa: E402
from trainit.trainit_gui.services.project_service import ProjectService  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_unsaved_changes_follow_edits_and_saves(qapp, mock_datasets_root: Path, temp_dir: Path):
    controller = ProjectController(AppState(), ProjectService(), DatasetService())
    folder = temp_dir / "proj"
    folder.mkdir()
    assert controller.create_project("p", str(folder), str(mock_datasets_root))
    assert not controller.has_unsaved_changes()

    controller.app_state.project.configs.append(TrainingConfig(name="new"))
    assert controller.has_unsaved_changes()

    assert controller.save_project()
    assert not controller.has_unsaved_changes()

    controller.app_state.config_dirty = True
    assert controller.has_unsaved_changes()


def test_analysis_reads_each_dataset_once(qapp, mock_datasets_root: Path):
    from unittest.mock import patch

    from trainit.trainit_gui.controllers.dataset_controller import DatasetController

    app_state = AppState()
    app_state.datasets_root = str(mock_datasets_root)
    app_state.selected_datasets = ["dataset_a", "dataset_b"]
    service = DatasetService()
    controller = DatasetController(app_state, service)

    with patch.object(service, "load_dataset_info", wraps=service.load_dataset_info) as load:
        assert controller.analyze_selected()
    assert load.call_count == 2
    assert app_state.analysis.total_images == 6
