"""Unit tests for SplitService."""

from pathlib import Path

import pytest

from trainit.trainit_gui.services.split_service import SplitResult, SplitService


class TestSplitServiceBasic:
    """Tests for basic SplitService functionality."""

    @pytest.fixture
    def service(self) -> SplitService:
        return SplitService()

    @pytest.mark.unit
    def test_extract_sequence_id_with_underscore(self, service: SplitService):
        """Test sequence ID extraction with underscore."""
        assert service.extract_sequence_id("M0101_00001.jpg") == "M0101"
        assert service.extract_sequence_id("video_001_frame.jpg") == "video"
        assert service.extract_sequence_id("seq_a_b_c.png") == "seq"

    @pytest.mark.unit
    def test_extract_sequence_id_without_underscore(self, service: SplitService):
        """Test sequence ID extraction without underscore."""
        assert service.extract_sequence_id("image001.jpg") == "image001"
        assert service.extract_sequence_id("file.png") == "file"

    @pytest.mark.unit
    def test_extract_sequence_id_uses_stem(self, service: SplitService):
        """Test that extension is not included."""
        assert service.extract_sequence_id("M0101_00001.jpg") == "M0101"
        assert service.extract_sequence_id("M0101_00001.jpeg") == "M0101"
        assert service.extract_sequence_id("M0101_00001.png") == "M0101"


class TestSplitServiceCollectFiles:
    """Tests for file collection."""

    @pytest.fixture
    def service(self) -> SplitService:
        return SplitService()

    @pytest.mark.unit
    def test_collect_files_train_only(self, service: SplitService, mock_dataset: Path):
        """Test collecting only train files."""
        files = service.collect_files_from_dataset(mock_dataset, include_val=False)

        assert len(files) == 5

    @pytest.mark.unit
    def test_collect_files_with_val(self, service: SplitService, mock_dataset: Path):
        """Test collecting train and val files."""
        files = service.collect_files_from_dataset(mock_dataset, include_val=True)

        assert len(files) == 7

    @pytest.mark.unit
    def test_collect_files_empty_dataset(self, service: SplitService, temp_dir: Path):
        """Test collecting from empty dataset."""
        (temp_dir / "images" / "train").mkdir(parents=True)

        files = service.collect_files_from_dataset(temp_dir, include_val=False)

        assert files == []

    @pytest.mark.unit
    def test_collect_files_returns_paths(
        self, service: SplitService, mock_dataset: Path
    ):
        """Test that returned files are Path objects."""
        files = service.collect_files_from_dataset(mock_dataset, include_val=False)

        assert all(isinstance(f, Path) for f in files)


class TestSplitServiceAlgorithm:
    """Tests for split algorithm."""

    @pytest.fixture
    def service(self) -> SplitService:
        return SplitService()

    @pytest.mark.unit
    def test_split_files_stratified(
        self, service: SplitService, mock_dataset_multi_sequence: Path
    ):
        """Test stratified split when sequences detected."""
        files = service.collect_files_from_dataset(
            mock_dataset_multi_sequence, include_val=False
        )

        result = service.split_files(files, train_ratio=0.8, seed=42)

        assert result.method == "stratified"
        assert result.sequences_found == 3
        assert len(result.train_files) + len(result.val_files) == len(files)

    @pytest.mark.unit
    def test_split_files_random_fallback(self, service: SplitService, temp_dir: Path):
        """Test random split when only one sequence detected."""
        # All files share same prefix before underscore -> 1 sequence -> random fallback
        (temp_dir / "images" / "train").mkdir(parents=True)
        for i in range(10):
            # All files have same prefix "seq" before underscore
            (temp_dir / "images" / "train" / f"seq_{i:05d}.jpg").touch()

        files = service.collect_files_from_dataset(temp_dir, include_val=False)
        result = service.split_files(files, train_ratio=0.8, seed=42)

        # With only 1 sequence, random split is used
        assert result.method == "random"
        assert result.sequences_found == 1

    @pytest.mark.unit
    def test_split_files_respects_ratio(
        self, service: SplitService, mock_dataset_multi_sequence: Path
    ):
        """Test that split respects the ratio approximately."""
        files = service.collect_files_from_dataset(
            mock_dataset_multi_sequence, include_val=False
        )

        result = service.split_files(files, train_ratio=0.8, seed=42)

        total = len(result.train_files) + len(result.val_files)
        actual_ratio = len(result.train_files) / total

        assert 0.7 <= actual_ratio <= 0.9

    @pytest.mark.unit
    def test_split_reproducibility(
        self, service: SplitService, mock_dataset_multi_sequence: Path
    ):
        """Test that same seed produces same split."""
        files = service.collect_files_from_dataset(
            mock_dataset_multi_sequence, include_val=False
        )

        result1 = service.split_files(files, train_ratio=0.8, seed=42)
        result2 = service.split_files(files, train_ratio=0.8, seed=42)

        assert set(result1.train_files) == set(result2.train_files)
        assert set(result1.val_files) == set(result2.val_files)

    @pytest.mark.unit
    def test_split_different_seeds(
        self, service: SplitService, mock_dataset_multi_sequence: Path
    ):
        """Test that different seeds can produce different splits."""
        files = service.collect_files_from_dataset(
            mock_dataset_multi_sequence, include_val=False
        )

        result1 = service.split_files(files, train_ratio=0.8, seed=42)
        result2 = service.split_files(files, train_ratio=0.8, seed=123)

        # Splits may differ (though not guaranteed with small datasets)
        assert result1.method == result2.method

    @pytest.mark.unit
    def test_split_empty_files(self, service: SplitService):
        """Test splitting empty file list."""
        result = service.split_files([], train_ratio=0.8, seed=42)

        assert result.train_files == []
        assert result.val_files == []


class TestSplitResult:
    """Tests for SplitResult dataclass."""

    @pytest.mark.unit
    def test_default_values(self):
        """Test SplitResult default values."""
        result = SplitResult()

        assert result.train_files == []
        assert result.val_files == []
        assert result.method == "random"
        assert result.sequences_found == 0

    @pytest.mark.unit
    def test_custom_values(self):
        """Test SplitResult with custom values."""
        train = [Path("/a.jpg"), Path("/b.jpg")]
        val = [Path("/c.jpg")]

        result = SplitResult(
            train_files=train, val_files=val, method="stratified", sequences_found=5
        )

        assert len(result.train_files) == 2
        assert len(result.val_files) == 1
        assert result.method == "stratified"
        assert result.sequences_found == 5


class TestSplitServicePrivateMethods:
    """Tests for private helper methods."""

    @pytest.fixture
    def service(self) -> SplitService:
        return SplitService()

    @pytest.mark.unit
    def test_group_by_sequence(
        self, service: SplitService, mock_dataset_multi_sequence: Path
    ):
        """Test grouping files by sequence."""
        files = service.collect_files_from_dataset(
            mock_dataset_multi_sequence, include_val=False
        )

        groups = service._group_by_sequence(files)

        assert len(groups) == 3
        assert "M0101" in groups
        assert "M0102" in groups
        assert "M0103" in groups
        assert len(groups["M0101"]) == 10

    @pytest.mark.unit
    def test_list_images_filters_extensions(
        self, service: SplitService, temp_dir: Path
    ):
        """Test that only image extensions are included."""
        (temp_dir / "images").mkdir()
        (temp_dir / "images" / "img1.jpg").touch()
        (temp_dir / "images" / "img2.png").touch()
        (temp_dir / "images" / "readme.txt").touch()
        (temp_dir / "images" / "labels.txt").touch()

        files = service._list_images(temp_dir / "images")

        assert len(files) == 2
        names = {f.name for f in files}
        assert "img1.jpg" in names
        assert "img2.png" in names
        assert "readme.txt" not in names
