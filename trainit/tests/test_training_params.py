"""Training parameters are listed once per side; these tests keep the sides in step."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtWidgets import QApplication  # noqa: E402

from trainit.train_yolo_obb import OPTIONAL_PARAMS, build_parser  # noqa: E402
from trainit.trainit_gui.models.project import (  # noqa: E402
    OPTIONAL_TRAINING_PARAMS,
    ProjectDefaults,
    TrainingConfig,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_gui_and_cli_optional_params_match():
    assert set(OPTIONAL_TRAINING_PARAMS) == set(OPTIONAL_PARAMS)


def test_every_optional_param_has_a_cli_option():
    dests = {action.dest for action in build_parser()._actions}
    assert set(OPTIONAL_PARAMS) <= dests


def test_project_defaults_cover_all_training_params():
    non_param = {"name", "description", "selected_datasets", "include_val_in_pool"}
    training_fields = set(TrainingConfig.model_fields) - non_param
    assert training_fields <= set(ProjectDefaults.model_fields)


def test_config_editor_has_a_widget_per_optional_param(qapp):
    from unittest.mock import MagicMock

    from trainit.trainit_gui.frontend.states.app_state import AppState
    from trainit.trainit_gui.frontend.widgets.config_editor import ConfigEditor

    editor = ConfigEditor(AppState(), MagicMock())
    assert set(OPTIONAL_TRAINING_PARAMS) <= set(editor._widgets)
