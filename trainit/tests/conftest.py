"""Pytest configuration and shared fixtures for trainit tests."""

import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
import yaml

from trainit.trainit_gui.models.dataset import DatasetInfo
from trainit.trainit_gui.models.manifest import ManifestInfo
from trainit.trainit_gui.models.project import Project, ProjectDefaults, TrainingConfig


# === Path Fixtures ===


@pytest.fixture
def test_fixtures_dir() -> Path:
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create and cleanup a temporary directory."""
    tmp = Path(tempfile.mkdtemp())
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


# === Sample Data Fixtures ===


@pytest.fixture
def sample_dataset_yaml() -> dict:
    """Sample dataset.yaml content."""
    return {
        "path": "/datasets/sample",
        "train": "images/train",
        "val": "images/val",
        "nc": 3,
        "names": {0: "car", 1: "truck", 2: "bus"},
    }


@pytest.fixture
def sample_class_names() -> dict[int, str]:
    """Sample class name mapping."""
    return {0: "car", 1: "truck", 2: "bus"}


@pytest.fixture
def sample_yolo_label_lines() -> list[str]:
    """Sample YOLO OBB label file content (class x1 y1 x2 y2 x3 y3 x4 y4)."""
    return [
        "0 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4",
        "1 0.5 0.5 0.7 0.5 0.7 0.7 0.5 0.7",
        "0 0.8 0.1 0.9 0.1 0.9 0.2 0.8 0.2",
    ]


# === Mock Dataset Fixtures ===


@pytest.fixture
def mock_dataset(temp_dir: Path, sample_dataset_yaml: dict) -> Path:
    """Create a mock YOLO dataset structure with train/val images and labels."""
    dataset_path = temp_dir / "test_dataset"

    # Create directories
    (dataset_path / "images" / "train").mkdir(parents=True)
    (dataset_path / "images" / "val").mkdir(parents=True)
    (dataset_path / "labels" / "train").mkdir(parents=True)
    (dataset_path / "labels" / "val").mkdir(parents=True)

    # Write dataset.yaml
    yaml_content = sample_dataset_yaml.copy()
    yaml_content["path"] = str(dataset_path)
    with open(dataset_path / "dataset.yaml", "w") as f:
        yaml.dump(yaml_content, f)

    # Create sample images and labels for train (with sequence pattern)
    for i in range(5):
        img_path = dataset_path / "images" / "train" / f"M0101_{i:05d}.jpg"
        img_path.touch()
        label_path = dataset_path / "labels" / "train" / f"M0101_{i:05d}.txt"
        label_path.write_text(f"{i % 3} 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n")

    # Create sample images and labels for val
    for i in range(2):
        img_path = dataset_path / "images" / "val" / f"M0101_{i + 5:05d}.jpg"
        img_path.touch()
        label_path = dataset_path / "labels" / "val" / f"M0101_{i + 5:05d}.txt"
        label_path.write_text(f"{i % 3} 0.5 0.5 0.7 0.5 0.7 0.7 0.5 0.7\n")

    return dataset_path


@pytest.fixture
def mock_dataset_multi_sequence(temp_dir: Path, sample_dataset_yaml: dict) -> Path:
    """Create a mock dataset with multiple sequences for stratified split testing."""
    dataset_path = temp_dir / "multi_seq_dataset"

    (dataset_path / "images" / "train").mkdir(parents=True)
    (dataset_path / "images" / "val").mkdir(parents=True)
    (dataset_path / "labels" / "train").mkdir(parents=True)
    (dataset_path / "labels" / "val").mkdir(parents=True)

    yaml_content = sample_dataset_yaml.copy()
    yaml_content["path"] = str(dataset_path)
    with open(dataset_path / "dataset.yaml", "w") as f:
        yaml.dump(yaml_content, f)

    # Create images from multiple sequences
    for seq in ["M0101", "M0102", "M0103"]:
        for i in range(10):
            img_path = dataset_path / "images" / "train" / f"{seq}_{i:05d}.jpg"
            img_path.touch()
            label_path = dataset_path / "labels" / "train" / f"{seq}_{i:05d}.txt"
            label_path.write_text(f"{i % 3} 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n")

    return dataset_path


@pytest.fixture
def mock_datasets_root(temp_dir: Path, sample_dataset_yaml: dict) -> Path:
    """Create a directory with multiple mock datasets."""
    root = temp_dir / "datasets"
    root.mkdir()

    for ds_name in ["dataset_a", "dataset_b"]:
        ds_path = root / ds_name
        (ds_path / "images" / "train").mkdir(parents=True)
        (ds_path / "images" / "val").mkdir(parents=True)
        (ds_path / "labels" / "train").mkdir(parents=True)
        (ds_path / "labels" / "val").mkdir(parents=True)

        yaml_content = sample_dataset_yaml.copy()
        yaml_content["path"] = str(ds_path)
        with open(ds_path / "dataset.yaml", "w") as f:
            yaml.dump(yaml_content, f)

        for i in range(3):
            (ds_path / "images" / "train" / f"{ds_name}_{i:05d}.jpg").touch()
            (ds_path / "labels" / "train" / f"{ds_name}_{i:05d}.txt").write_text(
                f"{i % 3} 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n"
            )

    return root


# === Model Fixtures ===


@pytest.fixture
def sample_training_config() -> TrainingConfig:
    """Sample training configuration."""
    return TrainingConfig(
        name="test_config",
        description="Test configuration",
        selected_datasets=["dataset_a", "dataset_b"],
        model="yolo11s-obb.pt",
        epochs=10,
        imgsz=640,
        batch=16,
    )


@pytest.fixture
def sample_project(mock_datasets_root: Path) -> Project:
    """Sample project with datasets."""
    return Project(
        name="Test Project",
        description="A test project",
        datasets_root=str(mock_datasets_root),
        available_datasets=["dataset_a", "dataset_b"],
    )


@pytest.fixture
def sample_project_defaults() -> ProjectDefaults:
    """Sample project defaults."""
    return ProjectDefaults(model="yolo11m-obb.pt", epochs=100, batch=32, lr0=0.001)


@pytest.fixture
def sample_dataset_info(mock_dataset: Path) -> DatasetInfo:
    """Sample dataset info."""
    return DatasetInfo(
        path=str(mock_dataset),
        name=mock_dataset.name,
        num_classes=3,
        class_names={0: "car", 1: "truck", 2: "bus"},
        train_images=5,
        val_images=2,
        class_distribution={0: 3, 1: 2, 2: 2},
        is_valid=True,
    )


@pytest.fixture
def sample_manifest_info() -> ManifestInfo:
    """Sample manifest info."""
    return ManifestInfo(
        generated_at="2024-01-01T00:00:00",
        project_name="Test Project",
        config_name="test_config",
        split_enabled=False,
        source_datasets=["dataset_a"],
    )


@pytest.fixture
def sample_project_json(mock_datasets_root: Path) -> dict:
    """Sample project.json content."""
    return {
        "name": "Test Project",
        "version": "1.0",
        "description": "Test description",
        "datasets_root": str(mock_datasets_root),
        "available_datasets": ["dataset_a", "dataset_b"],
        "configs": [
            {
                "name": "config1",
                "description": "",
                "selected_datasets": ["dataset_a"],
                "model": "yolo11s-obb.pt",
                "epochs": 50,
                "imgsz": 640,
                "batch": 30,
                "device": "auto",
                "project": "runs/obb",
            }
        ],
        "created_at": "2024-01-01T00:00:00",
        "modified_at": "2024-01-01T00:00:00",
    }
