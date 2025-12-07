"""Data models for trainit_gui."""

from .project import Project, TrainingConfig, ProjectDefaults
from .dataset import DatasetInfo, ClassInfo

__all__ = ['Project', 'TrainingConfig', 'ProjectDefaults', 'DatasetInfo', 'ClassInfo']
