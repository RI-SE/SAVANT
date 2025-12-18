"""Data models for trainit_gui."""

from .project import Project, TrainingConfig, ProjectDefaults
from .dataset import DatasetInfo, ClassInfo
from .manifest import Manifest, ManifestInfo, FileEntry

__all__ = [
    "Project",
    "TrainingConfig",
    "ProjectDefaults",
    "DatasetInfo",
    "ClassInfo",
    "Manifest",
    "ManifestInfo",
    "FileEntry",
]
