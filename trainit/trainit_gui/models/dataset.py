"""Dataset metadata and statistics models."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ClassInfo:
    """Information about a single class in a dataset."""

    class_id: int
    name: str
    count: int = 0  # Total objects of this class across all images


@dataclass
class DatasetInfo:
    """Metadata and statistics for a single training dataset.

    Represents a dataset folder created by extract_yolo_training.py,
    containing dataset.yaml, images/, and labels/ directories.
    """

    # Basic info
    path: str  # Absolute path to dataset folder
    name: str  # Folder name (used as identifier)

    # From dataset.yaml
    num_classes: int = 0
    class_names: dict[int, str] = field(default_factory=dict)

    # Computed statistics
    train_images: int = 0
    val_images: int = 0

    @property
    def total_images(self) -> int:
        """Total number of images (train + val)."""
        return self.train_images + self.val_images

    # Class distribution: class_id -> object count
    class_distribution: dict[int, int] = field(default_factory=dict)

    @property
    def total_objects(self) -> int:
        """Total number of annotated objects across all classes."""
        return sum(self.class_distribution.values())

    # Sample image for thumbnail display
    sample_image_path: Optional[str] = None

    # Validation status
    is_valid: bool = False
    error_message: str = ""

    def get_class_info_list(self) -> list[ClassInfo]:
        """Get list of ClassInfo objects sorted by class_id."""
        return [
            ClassInfo(
                class_id=cid,
                name=self.class_names.get(cid, f"class_{cid}"),
                count=self.class_distribution.get(cid, 0)
            )
            for cid in sorted(self.class_names.keys())
        ]

    def has_matching_classes(self, other: 'DatasetInfo') -> bool:
        """Check if this dataset has identical class mapping to another.

        Required for combining datasets in training.
        """
        if self.num_classes != other.num_classes:
            return False
        return self.class_names == other.class_names


@dataclass
class AggregatedStats:
    """Aggregated statistics across multiple selected datasets.

    Used for the analysis panel display.
    """

    # Datasets included
    dataset_names: list[str] = field(default_factory=list)

    # Totals
    total_train_images: int = 0
    total_val_images: int = 0

    @property
    def total_images(self) -> int:
        return self.total_train_images + self.total_val_images

    # Class info (shared across all datasets - must be identical)
    num_classes: int = 0
    class_names: dict[int, str] = field(default_factory=dict)

    # Aggregated class distribution
    class_distribution: dict[int, int] = field(default_factory=dict)

    @property
    def total_objects(self) -> int:
        return sum(self.class_distribution.values())

    # Per-dataset breakdown
    per_dataset_images: dict[str, int] = field(default_factory=dict)
    per_dataset_objects: dict[str, int] = field(default_factory=dict)

    # Sample images (one per dataset)
    sample_images: dict[str, str] = field(default_factory=dict)

    # Validation
    is_valid: bool = False
    error_message: str = ""

    @classmethod
    def from_datasets(cls, datasets: list[DatasetInfo]) -> 'AggregatedStats':
        """Create aggregated stats from a list of datasets.

        All datasets must have matching class mappings.
        """
        if not datasets:
            return cls(is_valid=False, error_message="No datasets selected")

        # Check class compatibility
        first = datasets[0]
        for ds in datasets[1:]:
            if not first.has_matching_classes(ds):
                return cls(
                    is_valid=False,
                    error_message=f"Class mismatch: '{first.name}' and '{ds.name}' "
                                  f"have different class definitions"
                )

        # Aggregate stats
        stats = cls(
            dataset_names=[ds.name for ds in datasets],
            num_classes=first.num_classes,
            class_names=dict(first.class_names),
            is_valid=True
        )

        for ds in datasets:
            stats.total_train_images += ds.train_images
            stats.total_val_images += ds.val_images
            stats.per_dataset_images[ds.name] = ds.total_images
            stats.per_dataset_objects[ds.name] = ds.total_objects

            if ds.sample_image_path:
                stats.sample_images[ds.name] = ds.sample_image_path

            # Aggregate class distribution
            for cid, count in ds.class_distribution.items():
                stats.class_distribution[cid] = (
                    stats.class_distribution.get(cid, 0) + count
                )

        return stats
