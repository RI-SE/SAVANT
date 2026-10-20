"""Tests for ConfigGenerator output handling."""

from pathlib import Path

import pytest

from trainit.trainit_gui.models.project import Project, TrainingConfig
from trainit.trainit_gui.services.config_generator import ConfigGenerator


def _generate(root: Path, out: Path, **config_kwargs):
    project = Project(name="p", datasets_root=str(root))
    config = TrainingConfig(name="cfg", selected_datasets=["dataset_a"], **config_kwargs)
    ConfigGenerator().generate_training_files(project, config, str(out), generate_manifest=False)


def _names(folder: Path):
    return sorted(p.name for p in folder.iterdir())


def test_regenerating_removes_stale_links(mock_datasets_root: Path, temp_dir: Path):
    out = temp_dir / "out"
    _generate(mock_datasets_root, out)
    stale = out / "images" / "val" / "stale.jpg"
    stale.symlink_to(temp_dir / "gone.jpg")  # broken link from an earlier run

    _generate(mock_datasets_root, out)

    assert not stale.is_symlink()
    assert len(_names(out / "images" / "train")) == 3
    assert _names(out / "images" / "val") == []


def test_output_inside_source_dataset_is_refused(mock_datasets_root: Path):
    source = mock_datasets_root / "dataset_a"
    with pytest.raises(ValueError, match="overlaps source dataset"):
        _generate(mock_datasets_root, source)
    assert len(_names(source / "images" / "train")) == 3
