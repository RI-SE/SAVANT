"""Manifest model for tracking source file hashes."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FileEntry:
    """Single file entry in manifest."""

    relative_path: (
        str  # Path relative to output dir, e.g., "images/train/ds1_img001.jpg"
    )
    source_path: str  # Absolute path to original source file
    sha256: str  # SHA256 hash of source file content


@dataclass
class ManifestInfo:
    """Metadata about how the manifest was generated."""

    generated_at: str
    project_name: str
    config_name: str
    split_enabled: bool
    source_datasets: list[str] = field(default_factory=list)
    split_ratio: Optional[float] = None
    split_seed: Optional[int] = None
    split_method: Optional[str] = None  # "stratified" or "random"


@dataclass
class Manifest:
    """Complete manifest structure for a generated training dataset."""

    version: str = "1.0"
    info: Optional[ManifestInfo] = None
    files: list[FileEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "version": self.version,
            "info": (
                {
                    "generated_at": self.info.generated_at,
                    "project_name": self.info.project_name,
                    "config_name": self.info.config_name,
                    "split_enabled": self.info.split_enabled,
                    "split_ratio": self.info.split_ratio,
                    "split_seed": self.info.split_seed,
                    "split_method": self.info.split_method,
                    "source_datasets": self.info.source_datasets,
                }
                if self.info
                else None
            ),
            "files": [
                {
                    "relative_path": f.relative_path,
                    "source_path": f.source_path,
                    "sha256": f.sha256,
                }
                for f in self.files
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Manifest":
        """Create Manifest from dictionary (loaded from JSON)."""
        info_data = data.get("info")
        info = (
            ManifestInfo(
                generated_at=info_data["generated_at"],
                project_name=info_data["project_name"],
                config_name=info_data["config_name"],
                split_enabled=info_data["split_enabled"],
                split_ratio=info_data.get("split_ratio"),
                split_seed=info_data.get("split_seed"),
                split_method=info_data.get("split_method"),
                source_datasets=info_data.get("source_datasets", []),
            )
            if info_data
            else None
        )

        files = [
            FileEntry(
                relative_path=f["relative_path"],
                source_path=f["source_path"],
                sha256=f["sha256"],
            )
            for f in data.get("files", [])
        ]

        return cls(version=data.get("version", "1.0"), info=info, files=files)
