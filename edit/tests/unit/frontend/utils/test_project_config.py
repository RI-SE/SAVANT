import importlib
import json

import pytest

from edit.frontend.utils import project_config
from edit.frontend.utils import settings_store


@pytest.fixture(autouse=True)
def reset_settings_modules():
    """Ensure each test starts with fresh module state."""
    importlib.reload(settings_store)
    importlib.reload(project_config)
    yield
    importlib.reload(settings_store)
    importlib.reload(project_config)


def test_persist_and_apply_round_trip(tmp_path):
    """Verify settings are saved to disk and re-applied correctly."""
    settings_store.set_zoom_rate(2.5)
    settings_store.set_movement_sensitivity(1.6)
    settings_store.set_rotation_sensitivity(0.45)
    settings_store.set_frame_history_count(120)
    settings_store.set_ontology_namespace("http://example.com/ns#")
    settings_store.set_action_interval_offset(77)
    settings_store.set_threshold_ranges(
        warning_range=(0.2, 0.3),
        error_range=(0.8, 0.9),
        show_warnings=True,
        show_errors=False,
    )
    settings_store.set_show_warnings(True)
    settings_store.set_show_errors(False)

    project_config.set_active_project_dir(tmp_path)
    project_config.persist_current_settings()

    config_path = tmp_path / project_config.PROJECT_CONFIG_FILENAME
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    saved = payload["settings"]
    assert saved["zoom_rate"] == pytest.approx(2.5)
    assert saved["movement_sensitivity"] == pytest.approx(1.6)
    assert saved["rotation_sensitivity"] == pytest.approx(0.45)
    assert saved["frame_history_count"] == 120
    assert saved["ontology_namespace"] == "http://example.com/ns#"
    assert saved["action_interval_offset"] == 77
    assert tuple(saved["warning_range"]) == (0.2, 0.3)
    assert tuple(saved["error_range"]) == (0.8, 0.9)
    assert saved["show_warnings"] is True
    assert saved["show_errors"] is False

    # Overwrite the in-memory values to confirm reapply works.
    settings_store.set_zoom_rate(1.0)
    settings_store.set_frame_history_count(10)
    settings_store.set_ontology_namespace("http://override/ns#")
    settings_store.set_action_interval_offset(5)
    settings_store.set_threshold_ranges(
        warning_range=(0.1, 0.2),
        error_range=(0.3, 0.4),
        show_warnings=False,
        show_errors=True,
    )

    loaded = project_config.load_project_config(tmp_path)
    project_config.apply_project_settings(loaded)

    assert settings_store.get_zoom_rate() == pytest.approx(2.5)
    assert settings_store.get_frame_history_count() == 120
    assert settings_store.get_ontology_namespace() == "http://example.com/ns#"
    assert settings_store.get_action_interval_offset() == 77
    assert settings_store.get_warning_range() == (0.2, 0.3)
    assert settings_store.get_error_range() == (0.8, 0.9)
    assert settings_store.get_show_warnings() is True
    assert settings_store.get_show_errors() is False


def test_apply_project_settings_ignores_invalid_values():
    """Ensure invalid stored values do not override current settings."""
    settings_store.set_zoom_rate(1.4)
    settings_store.set_frame_history_count(25)
    settings_store.set_ontology_namespace("http://valid/ns#")
    settings_store.set_action_interval_offset(40)

    bad_config = project_config.ProjectConfig(
        settings={
            "zoom_rate": 0.05,  # below minimum
            "frame_history_count": 0,
            "ontology_namespace": "",
            "action_interval_offset": -10,
        }
    )

    project_config.apply_project_settings(bad_config)

    assert settings_store.get_zoom_rate() == pytest.approx(1.4)
    assert settings_store.get_frame_history_count() == 25
    assert settings_store.get_ontology_namespace() == "http://valid/ns#"
    assert settings_store.get_action_interval_offset() == 40
