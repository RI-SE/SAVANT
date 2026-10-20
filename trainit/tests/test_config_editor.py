"""Tests for the config editor widget (headless)."""

import os
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtWidgets import QApplication  # noqa: E402

from trainit.trainit_gui.frontend.states.app_state import AppState  # noqa: E402
from trainit.trainit_gui.frontend.widgets.config_editor import ConfigEditor  # noqa: E402
from trainit.trainit_gui.models.project import TrainingConfig  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_save_preserves_split_settings(qapp):
    app_state = AppState()
    controller = MagicMock()
    editor = ConfigEditor(app_state, controller)
    app_state.current_config = TrainingConfig(
        name="cfg",
        split_enabled=True,
        split_ratio=0.8,
        split_seed=7,
        include_val_in_pool={"ds": True},
    )

    editor._on_save()

    saved = controller.update_config.call_args.args[0]
    assert (saved.split_enabled, saved.split_ratio, saved.split_seed) == (True, 0.8, 7)
    assert saved.include_val_in_pool == {"ds": True}
