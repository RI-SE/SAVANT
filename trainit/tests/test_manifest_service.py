"""Unit tests for ManifestService."""

import hashlib
from pathlib import Path

import pytest

from trainit.trainit_gui.models.manifest import FileEntry, Manifest, ManifestInfo
from trainit.trainit_gui.services.manifest_service import (
    ManifestService,
    VerificationResult,
)


class TestManifestServiceHashing:
    """Tests for SHA256 hashing functionality."""

    @pytest.fixture
    def service(self) -> ManifestService:
        return ManifestService()

    @pytest.mark.unit
    def test_compute_sha256(self, service: ManifestService, temp_dir: Path):
        """Test SHA256 computation."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello, World!")

        expected = hashlib.sha256(b"Hello, World!").hexdigest()
        actual = service.compute_sha256(test_file)

        assert actual == expected

    @pytest.mark.unit
    def test_compute_sha256_binary_file(self, service: ManifestService, temp_dir: Path):
        """Test SHA256 on binary content."""
        test_file = temp_dir / "test.bin"
        content = bytes(range(256))
        test_file.write_bytes(content)

        expected = hashlib.sha256(content).hexdigest()
        actual = service.compute_sha256(test_file)

        assert actual == expected

    @pytest.mark.unit
    def test_compute_sha256_follows_symlinks(
        self, service: ManifestService, temp_dir: Path
    ):
        """Test SHA256 follows symlinks to actual file."""
        real_file = temp_dir / "real.txt"
        real_file.write_text("Content")

        link = temp_dir / "link.txt"
        link.symlink_to(real_file)

        real_hash = service.compute_sha256(real_file)
        link_hash = service.compute_sha256(link)

        assert real_hash == link_hash

    @pytest.mark.unit
    def test_compute_sha256_empty_file(self, service: ManifestService, temp_dir: Path):
        """Test SHA256 of empty file."""
        test_file = temp_dir / "empty.txt"
        test_file.write_text("")

        expected = hashlib.sha256(b"").hexdigest()
        actual = service.compute_sha256(test_file)

        assert actual == expected


class TestManifestServiceGeneration:
    """Tests for manifest generation."""

    @pytest.fixture
    def service(self) -> ManifestService:
        return ManifestService()

    @pytest.mark.unit
    def test_generate_manifest(
        self,
        service: ManifestService,
        temp_dir: Path,
        sample_manifest_info: ManifestInfo,
    ):
        """Test manifest generation."""
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

        manifest = service.generate_manifest(dest, sample_manifest_info, mappings)

        assert len(manifest.files) == 2
        assert all(f.sha256 for f in manifest.files)
        assert manifest.info == sample_manifest_info

    @pytest.mark.unit
    def test_generate_manifest_relative_paths(
        self,
        service: ManifestService,
        temp_dir: Path,
        sample_manifest_info: ManifestInfo,
    ):
        """Test that manifest uses relative paths."""
        src = temp_dir / "src"
        src.mkdir()
        (src / "file.txt").write_text("content")

        dest = temp_dir / "dest" / "images" / "train"
        dest.mkdir(parents=True)

        mappings = [(dest / "img.jpg", src / "file.txt")]

        manifest = service.generate_manifest(
            temp_dir / "dest", sample_manifest_info, mappings
        )

        assert manifest.files[0].relative_path == "images/train/img.jpg"


class TestManifestServicePersistence:
    """Tests for manifest save/load."""

    @pytest.fixture
    def service(self) -> ManifestService:
        return ManifestService()

    @pytest.mark.unit
    def test_save_manifest(
        self,
        service: ManifestService,
        temp_dir: Path,
        sample_manifest_info: ManifestInfo,
    ):
        """Test saving manifest to file."""
        manifest = Manifest(info=sample_manifest_info)
        manifest.files.append(
            FileEntry(
                relative_path="test.txt", source_path="/src/test.txt", sha256="abc123"
            )
        )

        manifest_path = temp_dir / "manifest.json"
        service.save_manifest(manifest, manifest_path)

        assert manifest_path.exists()

    @pytest.mark.unit
    def test_load_manifest(
        self,
        service: ManifestService,
        temp_dir: Path,
        sample_manifest_info: ManifestInfo,
    ):
        """Test loading manifest from file."""
        manifest = Manifest(info=sample_manifest_info)
        manifest.files.append(
            FileEntry(
                relative_path="test.txt", source_path="/src/test.txt", sha256="abc123"
            )
        )

        manifest_path = temp_dir / "manifest.json"
        service.save_manifest(manifest, manifest_path)

        loaded = service.load_manifest(manifest_path)

        assert loaded is not None
        assert loaded.info.project_name == "Test Project"
        assert len(loaded.files) == 1

    @pytest.mark.unit
    def test_load_nonexistent_manifest(self, service: ManifestService):
        """Test loading nonexistent manifest returns None."""
        result = service.load_manifest(Path("/nonexistent/manifest.json"))

        assert result is None

    @pytest.mark.unit
    def test_roundtrip(
        self,
        service: ManifestService,
        temp_dir: Path,
        sample_manifest_info: ManifestInfo,
    ):
        """Test save/load roundtrip preserves data."""
        original = Manifest(info=sample_manifest_info)
        original.files.append(
            FileEntry(
                relative_path="images/train/img.jpg",
                source_path="/data/src/img.jpg",
                sha256="abc123def456",
            )
        )

        manifest_path = temp_dir / "manifest.json"
        service.save_manifest(original, manifest_path)
        loaded = service.load_manifest(manifest_path)

        assert loaded.version == original.version
        assert loaded.info.project_name == original.info.project_name
        assert loaded.info.config_name == original.info.config_name
        assert len(loaded.files) == len(original.files)
        assert loaded.files[0].sha256 == original.files[0].sha256


class TestManifestServiceVerification:
    """Tests for manifest verification."""

    @pytest.fixture
    def service(self) -> ManifestService:
        return ManifestService()

    @pytest.mark.unit
    def test_verify_all_valid(
        self,
        service: ManifestService,
        temp_dir: Path,
        sample_manifest_info: ManifestInfo,
    ):
        """Test verification with all files valid."""
        src_file = temp_dir / "source.txt"
        src_file.write_text("content")

        manifest = Manifest(info=sample_manifest_info)
        manifest.files.append(
            FileEntry(
                relative_path="dest.txt",
                source_path=str(src_file),
                sha256=service.compute_sha256(src_file),
            )
        )

        manifest_path = temp_dir / "manifest.json"
        service.save_manifest(manifest, manifest_path)

        result = service.verify_manifest(manifest_path)

        assert result.all_valid is True
        assert result.files_ok == 1
        assert result.total_files == 1

    @pytest.mark.unit
    def test_verify_missing_file(
        self,
        service: ManifestService,
        temp_dir: Path,
        sample_manifest_info: ManifestInfo,
    ):
        """Test verification detects missing files."""
        manifest = Manifest(info=sample_manifest_info)
        manifest.files.append(
            FileEntry(
                relative_path="dest.txt",
                source_path="/nonexistent/file.txt",
                sha256="abc123",
            )
        )

        manifest_path = temp_dir / "manifest.json"
        service.save_manifest(manifest, manifest_path)

        result = service.verify_manifest(manifest_path)

        assert result.all_valid is False
        assert len(result.files_missing) == 1

    @pytest.mark.unit
    def test_verify_changed_file(
        self,
        service: ManifestService,
        temp_dir: Path,
        sample_manifest_info: ManifestInfo,
    ):
        """Test verification detects changed files."""
        src_file = temp_dir / "source.txt"
        src_file.write_text("original")
        original_hash = service.compute_sha256(src_file)

        manifest = Manifest(info=sample_manifest_info)
        manifest.files.append(
            FileEntry(
                relative_path="dest.txt",
                source_path=str(src_file),
                sha256=original_hash,
            )
        )

        manifest_path = temp_dir / "manifest.json"
        service.save_manifest(manifest, manifest_path)

        src_file.write_text("modified")

        result = service.verify_manifest(manifest_path)

        assert result.all_valid is False
        assert len(result.files_changed) == 1

    @pytest.mark.unit
    def test_verify_mixed_results(
        self,
        service: ManifestService,
        temp_dir: Path,
        sample_manifest_info: ManifestInfo,
    ):
        """Test verification with mixed results."""
        good_file = temp_dir / "good.txt"
        good_file.write_text("good")
        good_hash = service.compute_sha256(good_file)

        changed_file = temp_dir / "changed.txt"
        changed_file.write_text("original")
        changed_hash = service.compute_sha256(changed_file)

        manifest = Manifest(info=sample_manifest_info)
        manifest.files.extend(
            [
                FileEntry(
                    relative_path="good.txt",
                    source_path=str(good_file),
                    sha256=good_hash,
                ),
                FileEntry(
                    relative_path="changed.txt",
                    source_path=str(changed_file),
                    sha256=changed_hash,
                ),
                FileEntry(
                    relative_path="missing.txt",
                    source_path="/nonexistent.txt",
                    sha256="xxx",
                ),
            ]
        )

        manifest_path = temp_dir / "manifest.json"
        service.save_manifest(manifest, manifest_path)

        changed_file.write_text("different")

        result = service.verify_manifest(manifest_path)

        assert result.all_valid is False
        assert result.files_ok == 1
        assert len(result.files_changed) == 1
        assert len(result.files_missing) == 1


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    @pytest.mark.unit
    def test_all_valid_true(self):
        """Test all_valid is True when no errors."""
        result = VerificationResult(total_files=5, files_ok=5)

        assert result.all_valid is True

    @pytest.mark.unit
    def test_all_valid_false_changed(self):
        """Test all_valid is False when files changed."""
        result = VerificationResult(total_files=5, files_ok=4, files_changed=["file1"])

        assert result.all_valid is False

    @pytest.mark.unit
    def test_all_valid_false_missing(self):
        """Test all_valid is False when files missing."""
        result = VerificationResult(total_files=5, files_ok=4, files_missing=["file2"])

        assert result.all_valid is False

    @pytest.mark.unit
    def test_default_lists(self):
        """Test that lists default to empty."""
        result = VerificationResult()

        assert result.files_changed == []
        assert result.files_missing == []
