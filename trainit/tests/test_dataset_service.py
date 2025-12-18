"""Unit tests for DatasetService."""

from pathlib import Path

import pytest
import yaml

from trainit.trainit_gui.services.dataset_service import DatasetService


class TestDatasetServiceScan:
    """Tests for DatasetService directory scanning."""

    @pytest.fixture
    def service(self) -> DatasetService:
        """Create a DatasetService instance."""
        return DatasetService()

    @pytest.mark.unit
    def test_scan_directory(self, service: DatasetService, mock_datasets_root: Path):
        """Test scanning for valid datasets."""
        datasets = service.scan_directory(str(mock_datasets_root))

        assert len(datasets) == 2
        assert "dataset_a" in datasets
        assert "dataset_b" in datasets

    @pytest.mark.unit
    def test_scan_directory_sorted(
        self, service: DatasetService, mock_datasets_root: Path
    ):
        """Test that results are sorted."""
        datasets = service.scan_directory(str(mock_datasets_root))

        assert datasets == sorted(datasets)

    @pytest.mark.unit
    def test_scan_directory_empty(self, service: DatasetService, temp_dir: Path):
        """Test scanning empty directory."""
        datasets = service.scan_directory(str(temp_dir))

        assert datasets == []

    @pytest.mark.unit
    def test_scan_directory_invalid_path(self, service: DatasetService):
        """Test scanning nonexistent path."""
        datasets = service.scan_directory("/nonexistent/path")

        assert datasets == []

    @pytest.mark.unit
    def test_scan_directory_ignores_invalid_datasets(
        self, service: DatasetService, mock_datasets_root: Path
    ):
        """Test that folders without dataset.yaml are ignored."""
        invalid_dir = mock_datasets_root / "invalid_dataset"
        invalid_dir.mkdir()

        datasets = service.scan_directory(str(mock_datasets_root))

        assert "invalid_dataset" not in datasets


class TestDatasetServiceLoadInfo:
    """Tests for loading dataset info."""

    @pytest.fixture
    def service(self) -> DatasetService:
        return DatasetService()

    @pytest.mark.unit
    def test_load_dataset_info(self, service: DatasetService, mock_dataset: Path):
        """Test loading dataset metadata."""
        info = service.load_dataset_info(str(mock_dataset))

        assert info.is_valid is True
        assert info.num_classes == 3
        assert info.train_images == 5
        assert info.val_images == 2
        assert info.name == mock_dataset.name

    @pytest.mark.unit
    def test_load_dataset_info_class_names(
        self, service: DatasetService, mock_dataset: Path
    ):
        """Test that class names are loaded correctly."""
        info = service.load_dataset_info(str(mock_dataset))

        assert info.class_names == {0: "car", 1: "truck", 2: "bus"}

    @pytest.mark.unit
    def test_load_dataset_info_missing_yaml(
        self, service: DatasetService, temp_dir: Path
    ):
        """Test loading dataset without dataset.yaml."""
        info = service.load_dataset_info(str(temp_dir))

        assert info.is_valid is False
        assert "Missing" in info.error_message

    @pytest.mark.unit
    def test_load_dataset_info_finds_sample_image(
        self, service: DatasetService, mock_dataset: Path
    ):
        """Test that a sample image is found."""
        info = service.load_dataset_info(str(mock_dataset))

        assert info.sample_image_path is not None
        assert Path(info.sample_image_path).exists()

    @pytest.mark.unit
    def test_load_dataset_info_class_distribution(
        self, service: DatasetService, mock_dataset: Path
    ):
        """Test that class distribution is computed."""
        info = service.load_dataset_info(str(mock_dataset))

        assert info.class_distribution is not None
        assert sum(info.class_distribution.values()) > 0


class TestDatasetServiceAnalysis:
    """Tests for dataset analysis functionality."""

    @pytest.fixture
    def service(self) -> DatasetService:
        return DatasetService()

    @pytest.mark.unit
    def test_analyze_datasets(self, service: DatasetService, mock_datasets_root: Path):
        """Test analyzing multiple datasets."""
        stats = service.analyze_datasets(
            str(mock_datasets_root), ["dataset_a", "dataset_b"]
        )

        assert stats.is_valid is True
        assert len(stats.dataset_names) == 2

    @pytest.mark.unit
    def test_analyze_datasets_invalid_dataset(
        self, service: DatasetService, mock_datasets_root: Path
    ):
        """Test analysis with invalid dataset returns error."""
        stats = service.analyze_datasets(
            str(mock_datasets_root), ["dataset_a", "nonexistent"]
        )

        assert stats.is_valid is False

    @pytest.mark.unit
    def test_analyze_datasets_empty_list(
        self, service: DatasetService, mock_datasets_root: Path
    ):
        """Test analysis with empty dataset list."""
        stats = service.analyze_datasets(str(mock_datasets_root), [])

        assert stats.is_valid is False


class TestDatasetServicePrivateMethods:
    """Tests for private helper methods."""

    @pytest.fixture
    def service(self) -> DatasetService:
        return DatasetService()

    @pytest.mark.unit
    def test_count_images(self, service: DatasetService, mock_dataset: Path):
        """Test image counting."""
        count = service._count_images(mock_dataset / "images" / "train")

        assert count == 5

    @pytest.mark.unit
    def test_count_images_empty_dir(self, service: DatasetService, temp_dir: Path):
        """Test counting images in empty directory."""
        temp_dir.mkdir(parents=True, exist_ok=True)
        count = service._count_images(temp_dir)

        assert count == 0

    @pytest.mark.unit
    def test_count_images_nonexistent_dir(self, service: DatasetService):
        """Test counting images in nonexistent directory."""
        count = service._count_images(Path("/nonexistent"))

        assert count == 0

    @pytest.mark.unit
    def test_analyze_labels(self, service: DatasetService, mock_dataset: Path):
        """Test label analysis."""
        distribution = service._analyze_labels(mock_dataset / "labels" / "train")

        assert isinstance(distribution, dict)
        assert sum(distribution.values()) == 5

    @pytest.mark.unit
    def test_parse_yaml_dict_names(self, service: DatasetService, temp_dir: Path):
        """Test parsing YAML with dict format class names."""
        yaml_path = temp_dir / "dataset.yaml"
        yaml_path.write_text(
            yaml.dump({"nc": 3, "names": {0: "car", 1: "truck", 2: "bus"}})
        )

        data = service._parse_yaml(yaml_path)

        assert data is not None
        assert data["names"] == {0: "car", 1: "truck", 2: "bus"}

    @pytest.mark.unit
    def test_parse_yaml_list_names(self, service: DatasetService, temp_dir: Path):
        """Test parsing YAML with list format class names."""
        yaml_path = temp_dir / "dataset.yaml"
        yaml_path.write_text(yaml.dump({"nc": 3, "names": ["car", "truck", "bus"]}))

        data = service._parse_yaml(yaml_path)

        assert data is not None
        assert data["names"] == ["car", "truck", "bus"]

    @pytest.mark.unit
    def test_find_sample_image(self, service: DatasetService, mock_dataset: Path):
        """Test finding sample image."""
        sample = service._find_sample_image(mock_dataset / "images" / "train")

        assert sample is not None
        assert Path(sample).exists()

    @pytest.mark.unit
    def test_find_sample_image_empty_dir(self, service: DatasetService, temp_dir: Path):
        """Test finding sample in empty directory."""
        temp_dir.mkdir(parents=True, exist_ok=True)
        sample = service._find_sample_image(temp_dir)

        assert sample is None
