"""Integration tests for trainit module."""

from pathlib import Path

import pytest

from trainit.trainit_gui.models.project import TrainingConfig
from trainit.trainit_gui.services.dataset_service import DatasetService
from trainit.trainit_gui.services.manifest_service import ManifestService
from trainit.trainit_gui.services.project_service import ProjectService
from trainit.trainit_gui.services.split_service import SplitService


class TestServiceIntegration:
    """Integration tests for service interactions."""

    @pytest.mark.integration
    def test_project_with_datasets(self, mock_datasets_root: Path, temp_dir: Path):
        """Test creating project and scanning datasets."""
        project_service = ProjectService()
        dataset_service = DatasetService()

        project_folder = temp_dir / "project"
        project_folder.mkdir()

        project = project_service.create_project(
            name="Integration Test",
            project_folder=str(project_folder),
            datasets_root=str(mock_datasets_root),
            description="Integration test project",
        )

        available = dataset_service.scan_directory(str(mock_datasets_root))
        project.available_datasets = available

        assert len(available) == 2
        assert project.name == "Integration Test"

    @pytest.mark.integration
    def test_dataset_analysis(self, mock_datasets_root: Path):
        """Test analyzing multiple datasets."""
        dataset_service = DatasetService()

        available = dataset_service.scan_directory(str(mock_datasets_root))

        stats = dataset_service.analyze_datasets(str(mock_datasets_root), available)

        assert stats.is_valid is True
        assert len(stats.dataset_names) == 2
        assert stats.total_train_images > 0

    @pytest.mark.integration
    def test_split_service_with_real_dataset(self, mock_dataset_multi_sequence: Path):
        """Test split service with multiple sequences."""
        split_service = SplitService()

        files = split_service.collect_files_from_dataset(
            mock_dataset_multi_sequence, include_val=False
        )

        result = split_service.split_files(files, train_ratio=0.8, seed=42)

        assert result.method == "stratified"
        assert result.sequences_found == 3
        assert len(result.train_files) > 0
        assert len(result.val_files) > 0

    @pytest.mark.integration
    def test_manifest_roundtrip(self, temp_dir: Path, sample_manifest_info):
        """Test manifest generation and verification."""
        manifest_service = ManifestService()

        src = temp_dir / "src"
        src.mkdir()
        (src / "file1.txt").write_text("content1")
        (src / "file2.txt").write_text("content2")

        dest = temp_dir / "dest"
        dest.mkdir()

        mappings = [
            (dest / "a.txt", src / "file1.txt"),
            (dest / "b.txt", src / "file2.txt"),
        ]

        manifest = manifest_service.generate_manifest(
            dest, sample_manifest_info, mappings
        )

        manifest_path = temp_dir / "manifest.json"
        manifest_service.save_manifest(manifest, manifest_path)

        result = manifest_service.verify_manifest(manifest_path)

        assert result.all_valid is True
        assert result.files_ok == 2


class TestProjectWorkflow:
    """End-to-end workflow tests."""

    @pytest.mark.integration
    def test_create_project_add_config_save_load(
        self, mock_datasets_root: Path, temp_dir: Path
    ):
        """Test complete project workflow."""
        project_service = ProjectService()
        dataset_service = DatasetService()

        project_folder = temp_dir / "project"
        project_folder.mkdir()

        project = project_service.create_project(
            name="Workflow Test",
            project_folder=str(project_folder),
            datasets_root=str(mock_datasets_root),
        )

        available = dataset_service.scan_directory(str(mock_datasets_root))
        project.available_datasets = available

        config = TrainingConfig(
            name="test_training", selected_datasets=available, epochs=10, batch=16
        )
        project.add_config(config)

        project_file = project_service.get_project_file_path(str(project_folder))
        project_service.save_project(project, project_file)

        loaded = project_service.load_project(project_file)

        assert loaded.name == "Workflow Test"
        assert len(loaded.available_datasets) == 2
        assert len(loaded.configs) == 1
        assert loaded.configs[0].name == "test_training"

    @pytest.mark.integration
    def test_config_modification_workflow(
        self, mock_datasets_root: Path, temp_dir: Path
    ):
        """Test modifying configs through service."""
        project_service = ProjectService()

        project_folder = temp_dir / "project"
        project_folder.mkdir()

        project_service.create_project(
            name="Config Test",
            project_folder=str(project_folder),
            datasets_root=str(mock_datasets_root),
        )

        project_file = project_service.get_project_file_path(str(project_folder))

        config1 = TrainingConfig(name="config1", epochs=50)
        project_service.add_config_to_project(project_file, config1)

        config2 = TrainingConfig(name="config2", epochs=100)
        project_service.add_config_to_project(project_file, config2)

        loaded = project_service.load_project(project_file)
        assert len(loaded.configs) == 2

        project_service.remove_config_from_project(project_file, "config1")

        loaded = project_service.load_project(project_file)
        assert len(loaded.configs) == 1
        assert loaded.configs[0].name == "config2"


class TestDatasetValidation:
    """Tests for dataset validation workflows."""

    @pytest.mark.integration
    def test_validate_dataset_structure(self, mock_dataset: Path):
        """Test that valid dataset passes validation."""
        dataset_service = DatasetService()

        info = dataset_service.load_dataset_info(str(mock_dataset))

        assert info.is_valid is True
        assert info.num_classes == 3
        assert info.train_images > 0

    @pytest.mark.integration
    def test_class_compatibility_check(self, mock_datasets_root: Path):
        """Test class compatibility between datasets."""
        dataset_service = DatasetService()

        available = dataset_service.scan_directory(str(mock_datasets_root))

        stats = dataset_service.analyze_datasets(str(mock_datasets_root), available)

        assert stats.is_valid is True
        assert stats.num_classes == 3


class TestSplitWorkflow:
    """Tests for train/val splitting workflow."""

    @pytest.mark.integration
    def test_split_preserves_total_count(self, mock_dataset_multi_sequence: Path):
        """Test that split doesn't lose any files."""
        split_service = SplitService()

        files = split_service.collect_files_from_dataset(
            mock_dataset_multi_sequence, include_val=False
        )
        original_count = len(files)

        result = split_service.split_files(files, train_ratio=0.8, seed=42)

        split_count = len(result.train_files) + len(result.val_files)

        assert split_count == original_count

    @pytest.mark.integration
    def test_split_deterministic(self, mock_dataset_multi_sequence: Path):
        """Test that splits are deterministic with same seed."""
        split_service = SplitService()

        files = split_service.collect_files_from_dataset(
            mock_dataset_multi_sequence, include_val=False
        )

        result1 = split_service.split_files(files, train_ratio=0.8, seed=42)
        result2 = split_service.split_files(files, train_ratio=0.8, seed=42)

        train1_names = {f.name for f in result1.train_files}
        train2_names = {f.name for f in result2.train_files}

        assert train1_names == train2_names
