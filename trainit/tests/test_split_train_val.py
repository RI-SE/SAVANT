"""Unit tests for DatasetSplitter class from split_train_val.py."""

from pathlib import Path

import pytest

from trainit.split_train_val import DatasetSplitter


class TestDatasetSplitterInit:
    """Tests for DatasetSplitter initialization."""

    @pytest.mark.unit
    def test_init_default_values(self, mock_dataset: Path):
        """Test splitter initialization with defaults."""
        splitter = DatasetSplitter(mock_dataset)

        assert splitter.train_ratio == 0.9
        assert splitter.val_ratio == pytest.approx(0.1)
        assert splitter.seed == 42
        assert splitter.verbose is False

    @pytest.mark.unit
    def test_init_custom_values(self, mock_dataset: Path):
        """Test splitter initialization with custom values."""
        splitter = DatasetSplitter(
            mock_dataset, train_ratio=0.8, seed=123, verbose=True
        )

        assert splitter.train_ratio == 0.8
        assert splitter.val_ratio == pytest.approx(0.2)
        assert splitter.seed == 123
        assert splitter.verbose is True

    @pytest.mark.unit
    def test_directories_set_correctly(self, mock_dataset: Path):
        """Test that directory paths are set correctly."""
        splitter = DatasetSplitter(mock_dataset)

        assert splitter.train_images_dir == mock_dataset / "images" / "train"
        assert splitter.train_labels_dir == mock_dataset / "labels" / "train"
        assert splitter.val_images_dir == mock_dataset / "images" / "val"
        assert splitter.val_labels_dir == mock_dataset / "labels" / "val"


class TestSequenceIdExtraction:
    """Tests for sequence ID extraction logic."""

    @pytest.mark.unit
    def test_extract_sequence_id_standard_format(self, mock_dataset: Path):
        """Test extraction with standard M####_##### format."""
        splitter = DatasetSplitter(mock_dataset)

        assert splitter.extract_sequence_id("M0101_00001.jpg") == "M0101"
        assert splitter.extract_sequence_id("M0202_12345.jpg") == "M0202"

    @pytest.mark.unit
    def test_extract_sequence_id_different_prefixes(self, mock_dataset: Path):
        """Test extraction with various prefixes."""
        splitter = DatasetSplitter(mock_dataset)

        assert splitter.extract_sequence_id("video1_frame001.jpg") == "video1"
        assert splitter.extract_sequence_id("seq_001_img.jpg") == "seq"

    @pytest.mark.unit
    def test_extract_sequence_id_no_underscore(self, mock_dataset: Path):
        """Test extraction when no underscore present."""
        splitter = DatasetSplitter(mock_dataset)

        # Returns the part before underscore, which is the whole filename if none
        assert splitter.extract_sequence_id("image001.jpg") == "image001.jpg"


class TestGroupImagesBySequence:
    """Tests for sequence grouping logic."""

    @pytest.mark.unit
    def test_group_images_by_sequence(self, mock_dataset: Path):
        """Test grouping images by sequence ID."""
        splitter = DatasetSplitter(mock_dataset)
        sequences = splitter.group_images_by_sequence()

        assert isinstance(sequences, dict)
        assert len(sequences) >= 1
        assert "M0101" in sequences
        assert len(sequences["M0101"]) == 5

    @pytest.mark.unit
    def test_group_multiple_sequences(self, mock_dataset_multi_sequence: Path):
        """Test grouping with multiple sequences."""
        splitter = DatasetSplitter(mock_dataset_multi_sequence)
        sequences = splitter.group_images_by_sequence()

        assert len(sequences) == 3
        assert "M0101" in sequences
        assert "M0102" in sequences
        assert "M0103" in sequences
        assert len(sequences["M0101"]) == 10


class TestValidateDirectories:
    """Tests for directory validation."""

    @pytest.mark.unit
    def test_validate_directories_success(self, mock_dataset: Path):
        """Test validation passes for valid dataset."""
        splitter = DatasetSplitter(mock_dataset)
        assert splitter.validate_directories() is True

    @pytest.mark.unit
    def test_validate_directories_missing_train_images(self, temp_dir: Path):
        """Test validation fails when train images missing."""
        (temp_dir / "labels" / "train").mkdir(parents=True)

        splitter = DatasetSplitter(temp_dir)
        assert splitter.validate_directories() is False

    @pytest.mark.unit
    def test_validate_directories_missing_train_labels(self, temp_dir: Path):
        """Test validation fails when train labels missing."""
        (temp_dir / "images" / "train").mkdir(parents=True)

        splitter = DatasetSplitter(temp_dir)
        assert splitter.validate_directories() is False


class TestSplitDataset:
    """Tests for the split operation."""

    @pytest.mark.unit
    def test_split_dataset_dry_run(self, mock_dataset: Path):
        """Test dry run does not move files."""
        splitter = DatasetSplitter(mock_dataset, train_ratio=0.8)

        train_before = len(list((mock_dataset / "images" / "train").glob("*.jpg")))

        stats = splitter.split_dataset(dry_run=True)

        train_after = len(list((mock_dataset / "images" / "train").glob("*.jpg")))

        assert train_before == train_after
        assert stats.get("total_images", 0) > 0

    @pytest.mark.integration
    def test_split_dataset_actual_split(self, mock_dataset_multi_sequence: Path):
        """Test actual split moves files correctly."""
        splitter = DatasetSplitter(
            mock_dataset_multi_sequence, train_ratio=0.8, seed=42
        )

        val_before = len(
            list((mock_dataset_multi_sequence / "images" / "val").glob("*.jpg"))
        )

        stats = splitter.split_dataset(dry_run=False)

        val_after = len(
            list((mock_dataset_multi_sequence / "images" / "val").glob("*.jpg"))
        )

        assert val_after > val_before
        assert stats["moved_images"] > 0

    @pytest.mark.unit
    def test_split_returns_correct_stats(self, mock_dataset: Path):
        """Test that split returns correct statistics."""
        splitter = DatasetSplitter(mock_dataset, train_ratio=0.8)

        stats = splitter.split_dataset(dry_run=True)

        assert "sequences" in stats
        assert "total_images" in stats
        assert "train_images" in stats
        assert "val_images" in stats
        assert "moved_images" in stats


class TestRestoreDataset:
    """Tests for restore operation."""

    @pytest.mark.integration
    def test_restore_dataset(self, mock_dataset_multi_sequence: Path):
        """Test restore moves files back to train."""
        splitter = DatasetSplitter(
            mock_dataset_multi_sequence, train_ratio=0.8, seed=42
        )

        splitter.split_dataset(dry_run=False)
        val_after_split = len(
            list((mock_dataset_multi_sequence / "images" / "val").glob("*.jpg"))
        )
        assert val_after_split > 0

        splitter.restore_dataset(dry_run=False)
        val_after_restore = len(
            list((mock_dataset_multi_sequence / "images" / "val").glob("*.jpg"))
        )

        assert val_after_restore == 0

    @pytest.mark.unit
    def test_restore_nonexistent_val_dirs(self, temp_dir: Path):
        """Test restore handles missing val directories."""
        (temp_dir / "images" / "train").mkdir(parents=True)
        (temp_dir / "labels" / "train").mkdir(parents=True)

        splitter = DatasetSplitter(temp_dir)
        stats = splitter.restore_dataset(dry_run=True)

        assert stats == {}


class TestGetLabelPath:
    """Tests for label path resolution."""

    @pytest.mark.unit
    def test_get_label_path(self, mock_dataset: Path):
        """Test getting corresponding label path for an image."""
        splitter = DatasetSplitter(mock_dataset)
        img_path = mock_dataset / "images" / "train" / "M0101_00001.jpg"

        label_path = splitter.get_label_path(img_path)

        expected = mock_dataset / "labels" / "train" / "M0101_00001.txt"
        assert label_path == expected


class TestRemoveOrphanedImages:
    """Tests for orphan removal."""

    @pytest.mark.unit
    def test_remove_orphans_finds_orphans(self, mock_dataset: Path):
        """Test that orphan images are detected."""
        orphan = mock_dataset / "images" / "train" / "orphan_00001.jpg"
        orphan.touch()

        splitter = DatasetSplitter(mock_dataset, verbose=True)
        stats = splitter.remove_orphaned_images(dry_run=True)

        assert stats["train_orphans"] == 1
        assert stats["total_orphans"] == 1

    @pytest.mark.integration
    def test_remove_orphans_deletes_files(self, mock_dataset: Path):
        """Test that orphan images are actually deleted."""
        orphan = mock_dataset / "images" / "train" / "orphan_00001.jpg"
        orphan.touch()
        assert orphan.exists()

        splitter = DatasetSplitter(mock_dataset)
        splitter.remove_orphaned_images(dry_run=False)

        assert not orphan.exists()
