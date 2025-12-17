"""Unit tests for trainit GUI models."""

import pytest

from trainit.trainit_gui.models.dataset import AggregatedStats, ClassInfo, DatasetInfo
from trainit.trainit_gui.models.manifest import FileEntry, Manifest, ManifestInfo
from trainit.trainit_gui.models.project import Project, ProjectDefaults, TrainingConfig


class TestTrainingConfig:
    """Tests for TrainingConfig model."""

    @pytest.mark.unit
    def test_creation_with_defaults(self):
        """Test creating config with default values."""
        config = TrainingConfig(name="test")

        assert config.name == "test"
        assert config.model == "yolo11s-obb.pt"
        assert config.epochs == 50
        assert config.imgsz == 640
        assert config.batch == 30
        assert config.device == "auto"

    @pytest.mark.unit
    def test_creation_with_custom_values(self):
        """Test creating config with custom values."""
        config = TrainingConfig(name="custom", epochs=100, batch=16, lr0=0.001)

        assert config.epochs == 100
        assert config.batch == 16
        assert config.lr0 == 0.001

    @pytest.mark.unit
    def test_get_non_default_params(self):
        """Test getting non-default parameters."""
        config = TrainingConfig(name="test", lr0=0.001, epochs=100)

        params = config.get_non_default_params()

        assert "lr0" in params
        assert params["lr0"] == 0.001
        assert "epochs" in params
        assert params["epochs"] == 100
        assert "name" not in params
        assert "selected_datasets" not in params

    @pytest.mark.unit
    def test_get_non_default_params_excludes_split_fields(self):
        """Test that split fields are excluded from non-default params."""
        config = TrainingConfig(
            name="test", split_enabled=True, split_ratio=0.8, split_seed=123
        )

        params = config.get_non_default_params()

        assert "split_enabled" not in params
        assert "split_ratio" not in params
        assert "split_seed" not in params

    @pytest.mark.unit
    def test_split_configuration(self):
        """Test split-related fields."""
        config = TrainingConfig(
            name="test", split_enabled=True, split_ratio=0.8, split_seed=123
        )

        assert config.split_enabled is True
        assert config.split_ratio == 0.8
        assert config.split_seed == 123

    @pytest.mark.unit
    def test_selected_datasets_list(self):
        """Test selected_datasets field."""
        config = TrainingConfig(name="test", selected_datasets=["ds1", "ds2"])

        assert config.selected_datasets == ["ds1", "ds2"]


class TestProjectDefaults:
    """Tests for ProjectDefaults model."""

    @pytest.mark.unit
    def test_all_fields_default_to_none(self):
        """Test that all fields default to None."""
        defaults = ProjectDefaults()

        assert defaults.model is None
        assert defaults.epochs is None
        assert defaults.batch is None
        assert defaults.lr0 is None

    @pytest.mark.unit
    def test_apply_to_config(self, sample_project_defaults: ProjectDefaults):
        """Test applying defaults to a config."""
        config = TrainingConfig(name="test", epochs=50)

        updated = sample_project_defaults.apply_to_config(config)

        assert updated.name == "test"
        assert updated.model == "yolo11m-obb.pt"
        assert updated.epochs == 100
        assert updated.batch == 32
        assert updated.lr0 == 0.001

    @pytest.mark.unit
    def test_apply_partial_defaults(self):
        """Test applying only some defaults."""
        defaults = ProjectDefaults(epochs=200)
        config = TrainingConfig(name="test", batch=8)

        updated = defaults.apply_to_config(config)

        assert updated.epochs == 200
        assert updated.batch == 8
        assert updated.model == "yolo11s-obb.pt"


class TestProject:
    """Tests for Project model."""

    @pytest.mark.unit
    def test_creation(self, mock_datasets_root):
        """Test creating a project."""
        project = Project(name="Test", datasets_root=str(mock_datasets_root))

        assert project.name == "Test"
        assert project.version == "1.0"
        assert project.configs == []

    @pytest.mark.unit
    def test_add_and_get_config(self, sample_project: Project):
        """Test adding and retrieving configs."""
        config = TrainingConfig(name="new_config")
        sample_project.add_config(config)

        retrieved = sample_project.get_config_by_name("new_config")
        assert retrieved is not None
        assert retrieved.name == "new_config"

    @pytest.mark.unit
    def test_get_nonexistent_config(self, sample_project: Project):
        """Test getting a config that doesn't exist."""
        result = sample_project.get_config_by_name("nonexistent")
        assert result is None

    @pytest.mark.unit
    def test_remove_config(self, sample_project: Project):
        """Test removing a config."""
        config = TrainingConfig(name="to_remove")
        sample_project.add_config(config)

        result = sample_project.remove_config("to_remove")

        assert result is True
        assert sample_project.get_config_by_name("to_remove") is None

    @pytest.mark.unit
    def test_remove_nonexistent_config(self, sample_project: Project):
        """Test removing a config that doesn't exist."""
        result = sample_project.remove_config("nonexistent")
        assert result is False

    @pytest.mark.unit
    def test_update_modified(self, sample_project: Project):
        """Test modification timestamp updates."""
        original = sample_project.modified_at

        sample_project.update_modified()

        assert sample_project.modified_at != original


class TestDatasetInfo:
    """Tests for DatasetInfo model."""

    @pytest.mark.unit
    def test_total_images(self, sample_dataset_info: DatasetInfo):
        """Test total images property."""
        assert sample_dataset_info.total_images == 7

    @pytest.mark.unit
    def test_total_objects(self, sample_dataset_info: DatasetInfo):
        """Test total objects property."""
        assert sample_dataset_info.total_objects == 7

    @pytest.mark.unit
    def test_has_matching_classes_same(self, sample_dataset_info: DatasetInfo):
        """Test class compatibility check with matching classes."""
        other = DatasetInfo(
            path="/other",
            name="other",
            num_classes=3,
            class_names={0: "car", 1: "truck", 2: "bus"},
            is_valid=True,
        )

        assert sample_dataset_info.has_matching_classes(other) is True

    @pytest.mark.unit
    def test_has_matching_classes_different_count(
        self, sample_dataset_info: DatasetInfo
    ):
        """Test class incompatibility with different class count."""
        other = DatasetInfo(
            path="/other",
            name="other",
            num_classes=2,
            class_names={0: "car", 1: "truck"},
            is_valid=True,
        )

        assert sample_dataset_info.has_matching_classes(other) is False

    @pytest.mark.unit
    def test_has_matching_classes_different_names(
        self, sample_dataset_info: DatasetInfo
    ):
        """Test class incompatibility with different names."""
        other = DatasetInfo(
            path="/other",
            name="other",
            num_classes=3,
            class_names={0: "car", 1: "van", 2: "bus"},
            is_valid=True,
        )

        assert sample_dataset_info.has_matching_classes(other) is False

    @pytest.mark.unit
    def test_get_class_info_list(self, sample_dataset_info: DatasetInfo):
        """Test getting class info as list."""
        info_list = sample_dataset_info.get_class_info_list()

        assert len(info_list) == 3
        assert all(isinstance(ci, ClassInfo) for ci in info_list)
        assert info_list[0].class_id == 0
        assert info_list[0].name == "car"


class TestAggregatedStats:
    """Tests for AggregatedStats model."""

    @pytest.mark.unit
    def test_from_datasets_single(self, sample_dataset_info: DatasetInfo):
        """Test aggregation from single dataset."""
        stats = AggregatedStats.from_datasets([sample_dataset_info])

        assert stats.is_valid is True
        assert stats.total_train_images == 5
        assert stats.total_val_images == 2
        assert stats.total_images == 7

    @pytest.mark.unit
    def test_from_datasets_empty(self):
        """Test aggregation with no datasets."""
        stats = AggregatedStats.from_datasets([])

        assert stats.is_valid is False
        assert "No datasets" in stats.error_message

    @pytest.mark.unit
    def test_from_datasets_class_mismatch(self, sample_dataset_info: DatasetInfo):
        """Test aggregation detects class mismatch."""
        other = DatasetInfo(
            path="/other",
            name="other",
            num_classes=5,
            class_names={0: "a", 1: "b", 2: "c", 3: "d", 4: "e"},
            is_valid=True,
        )

        stats = AggregatedStats.from_datasets([sample_dataset_info, other])

        assert stats.is_valid is False
        assert "mismatch" in stats.error_message.lower()

    @pytest.mark.unit
    def test_from_datasets_multiple(self, sample_dataset_info: DatasetInfo):
        """Test aggregation from multiple compatible datasets."""
        other = DatasetInfo(
            path="/other",
            name="other",
            num_classes=3,
            class_names={0: "car", 1: "truck", 2: "bus"},
            train_images=10,
            val_images=5,
            class_distribution={0: 5, 1: 5, 2: 5},
            is_valid=True,
        )

        stats = AggregatedStats.from_datasets([sample_dataset_info, other])

        assert stats.is_valid is True
        assert stats.total_train_images == 15
        assert stats.total_val_images == 7
        assert len(stats.dataset_names) == 2


class TestManifest:
    """Tests for Manifest model."""

    @pytest.mark.unit
    def test_to_dict(self, sample_manifest_info: ManifestInfo):
        """Test manifest serialization."""
        manifest = Manifest(info=sample_manifest_info)
        manifest.files.append(
            FileEntry(
                relative_path="images/train/img.jpg",
                source_path="/src/img.jpg",
                sha256="abc123",
            )
        )

        result = manifest.to_dict()

        assert result["version"] == "1.0"
        assert result["info"]["project_name"] == "Test Project"
        assert len(result["files"]) == 1
        assert result["files"][0]["sha256"] == "abc123"

    @pytest.mark.unit
    def test_from_dict(self):
        """Test manifest deserialization."""
        data = {
            "version": "1.0",
            "info": {
                "generated_at": "2024-01-01T00:00:00",
                "project_name": "Test",
                "config_name": "config1",
                "split_enabled": False,
                "source_datasets": ["ds1"],
            },
            "files": [
                {
                    "relative_path": "img.jpg",
                    "source_path": "/src/img.jpg",
                    "sha256": "abc",
                }
            ],
        }

        manifest = Manifest.from_dict(data)

        assert manifest.version == "1.0"
        assert manifest.info.project_name == "Test"
        assert len(manifest.files) == 1
        assert manifest.files[0].sha256 == "abc"

    @pytest.mark.unit
    def test_roundtrip(self, sample_manifest_info: ManifestInfo):
        """Test to_dict/from_dict roundtrip."""
        original = Manifest(info=sample_manifest_info)
        original.files.append(
            FileEntry(
                relative_path="test.jpg", source_path="/src/test.jpg", sha256="hash123"
            )
        )

        data = original.to_dict()
        restored = Manifest.from_dict(data)

        assert restored.version == original.version
        assert restored.info.project_name == original.info.project_name
        assert len(restored.files) == len(original.files)
        assert restored.files[0].sha256 == original.files[0].sha256


class TestFileEntry:
    """Tests for FileEntry model."""

    @pytest.mark.unit
    def test_creation(self):
        """Test FileEntry creation."""
        entry = FileEntry(
            relative_path="images/train/img.jpg",
            source_path="/data/img.jpg",
            sha256="abc123def456",
        )

        assert entry.relative_path == "images/train/img.jpg"
        assert entry.source_path == "/data/img.jpg"
        assert entry.sha256 == "abc123def456"


class TestManifestInfo:
    """Tests for ManifestInfo model."""

    @pytest.mark.unit
    def test_creation_minimal(self):
        """Test minimal ManifestInfo creation."""
        info = ManifestInfo(
            generated_at="2024-01-01T00:00:00",
            project_name="Test",
            config_name="config",
            split_enabled=False,
        )

        assert info.project_name == "Test"
        assert info.split_enabled is False
        assert info.split_ratio is None

    @pytest.mark.unit
    def test_creation_with_split(self):
        """Test ManifestInfo with split info."""
        info = ManifestInfo(
            generated_at="2024-01-01T00:00:00",
            project_name="Test",
            config_name="config",
            split_enabled=True,
            split_ratio=0.8,
            split_seed=42,
            split_method="stratified",
        )

        assert info.split_enabled is True
        assert info.split_ratio == 0.8
        assert info.split_seed == 42
        assert info.split_method == "stratified"
