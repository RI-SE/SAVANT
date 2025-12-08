"""Service for generating and verifying file manifests."""

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..models.manifest import Manifest, ManifestInfo, FileEntry

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of manifest verification."""

    total_files: int = 0
    files_ok: int = 0
    files_changed: list[str] = None  # List of changed file descriptions
    files_missing: list[str] = None  # List of missing source paths

    def __post_init__(self):
        if self.files_changed is None:
            self.files_changed = []
        if self.files_missing is None:
            self.files_missing = []

    @property
    def all_valid(self) -> bool:
        """True if all files passed verification."""
        return len(self.files_changed) == 0 and len(self.files_missing) == 0


class ManifestService:
    """Service for generating and verifying file manifests with SHA256 hashes."""

    BUFFER_SIZE = 65536  # 64KB chunks for hashing

    def compute_sha256(self, file_path: Path) -> str:
        """Compute SHA256 hash of file, following symlinks.

        Args:
            file_path: Path to file (may be a symlink)

        Returns:
            Hex string of SHA256 hash
        """
        # Resolve symlinks to get actual file
        resolved = file_path.resolve()

        sha256 = hashlib.sha256()
        with open(resolved, 'rb') as f:
            while chunk := f.read(self.BUFFER_SIZE):
                sha256.update(chunk)
        return sha256.hexdigest()

    def generate_manifest(
        self,
        output_dir: Path,
        info: ManifestInfo,
        file_mappings: list[tuple[Path, Path]]
    ) -> Manifest:
        """Generate manifest from file mappings.

        Args:
            output_dir: Base output directory for relative paths
            info: Manifest metadata
            file_mappings: List of (destination_path, source_path) tuples

        Returns:
            Manifest object with computed hashes
        """
        manifest = Manifest(info=info)

        for dest_path, source_path in file_mappings:
            try:
                relative = dest_path.relative_to(output_dir)
                sha256 = self.compute_sha256(source_path)

                entry = FileEntry(
                    relative_path=str(relative),
                    source_path=str(source_path.resolve()),
                    sha256=sha256
                )
                manifest.files.append(entry)
            except Exception as e:
                logger.warning(f"Failed to hash {source_path}: {e}")

        return manifest

    def save_manifest(self, manifest: Manifest, output_path: Path) -> None:
        """Save manifest to JSON file.

        Args:
            manifest: Manifest to save
            output_path: Path to output JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(manifest.to_dict(), f, indent=2)
        logger.info(f"Saved manifest to {output_path}")

    def load_manifest(self, manifest_path: Path) -> Optional[Manifest]:
        """Load manifest from JSON file.

        Args:
            manifest_path: Path to manifest JSON file

        Returns:
            Manifest object or None if loading failed
        """
        try:
            with open(manifest_path, 'r') as f:
                data = json.load(f)
            return Manifest.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load manifest from {manifest_path}: {e}")
            return None

    def verify_manifest(self, manifest_path: Path) -> VerificationResult:
        """Verify source files against manifest.

        Checks that all source files exist and have matching hashes.

        Args:
            manifest_path: Path to manifest JSON file

        Returns:
            VerificationResult with verification details
        """
        manifest = self.load_manifest(manifest_path)
        if not manifest:
            return VerificationResult(
                files_missing=["Failed to load manifest file"]
            )

        result = VerificationResult(total_files=len(manifest.files))

        for entry in manifest.files:
            source = Path(entry.source_path)

            if not source.exists():
                result.files_missing.append(entry.source_path)
                continue

            try:
                current_hash = self.compute_sha256(source)
                if current_hash != entry.sha256:
                    result.files_changed.append(
                        f"{entry.source_path}: "
                        f"expected {entry.sha256[:16]}..., "
                        f"got {current_hash[:16]}..."
                    )
                else:
                    result.files_ok += 1
            except Exception as e:
                result.files_missing.append(
                    f"{entry.source_path}: error reading - {e}"
                )

        return result
