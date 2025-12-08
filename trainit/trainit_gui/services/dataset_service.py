"""Service for scanning and analyzing training datasets."""

import logging
import os
from pathlib import Path
from typing import Optional

import yaml

from ..models.dataset import DatasetInfo, AggregatedStats

logger = logging.getLogger(__name__)


class DatasetService:
    """Service for discovering and analyzing YOLO training datasets.

    Datasets are expected to follow the structure created by extract_yolo_training.py:
        dataset_folder/
        ├── dataset.yaml
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/
    """

    DATASET_YAML_NAME = "dataset.yaml"
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def scan_directory(self, root_path: str) -> list[str]:
        """Scan a directory for valid dataset folders.

        Args:
            root_path: Path to directory containing dataset folders

        Returns:
            List of folder names that contain valid dataset.yaml files
        """
        root = Path(root_path)
        if not root.is_dir():
            logger.warning(f"Not a directory: {root_path}")
            return []

        valid_datasets = []
        for item in root.iterdir():
            if item.is_dir():
                yaml_path = item / self.DATASET_YAML_NAME
                if yaml_path.exists():
                    valid_datasets.append(item.name)
                    logger.debug(f"Found dataset: {item.name}")

        return sorted(valid_datasets)

    def load_dataset_info(self, dataset_path: str) -> DatasetInfo:
        """Load metadata and compute statistics for a single dataset.

        Args:
            dataset_path: Absolute path to dataset folder

        Returns:
            DatasetInfo with metadata and statistics
        """
        path = Path(dataset_path)
        name = path.name

        info = DatasetInfo(path=str(path), name=name)

        # Parse dataset.yaml
        yaml_path = path / self.DATASET_YAML_NAME
        if not yaml_path.exists():
            info.error_message = f"Missing {self.DATASET_YAML_NAME}"
            return info

        try:
            yaml_data = self._parse_yaml(yaml_path)
            if yaml_data is None:
                info.error_message = "Failed to parse dataset.yaml"
                return info

            info.num_classes = yaml_data.get('nc', 0)
            names = yaml_data.get('names', {})
            # Handle both dict and list formats
            if isinstance(names, list):
                info.class_names = {i: n for i, n in enumerate(names)}
            else:
                info.class_names = {int(k): v for k, v in names.items()}

        except Exception as e:
            info.error_message = f"Error parsing yaml: {e}"
            return info

        # Count images
        info.train_images = self._count_images(path / "images" / "train")
        info.val_images = self._count_images(path / "images" / "val")

        # Analyze labels for class distribution
        info.class_distribution = self._analyze_labels(path / "labels" / "train")
        val_dist = self._analyze_labels(path / "labels" / "val")
        for cid, count in val_dist.items():
            info.class_distribution[cid] = info.class_distribution.get(cid, 0) + count

        # Find a sample image
        info.sample_image_path = self._find_sample_image(path / "images" / "train")
        if not info.sample_image_path:
            info.sample_image_path = self._find_sample_image(path / "images" / "val")

        info.is_valid = True
        return info

    def analyze_datasets(
        self,
        root_path: str,
        dataset_names: list[str]
    ) -> AggregatedStats:
        """Analyze multiple datasets and return aggregated statistics.

        Args:
            root_path: Root directory containing datasets
            dataset_names: List of dataset folder names to analyze

        Returns:
            AggregatedStats with combined statistics
        """
        root = Path(root_path)
        datasets = []

        for name in dataset_names:
            dataset_path = root / name
            info = self.load_dataset_info(str(dataset_path))
            if info.is_valid:
                datasets.append(info)
            else:
                return AggregatedStats(
                    is_valid=False,
                    error_message=f"Invalid dataset '{name}': {info.error_message}"
                )

        return AggregatedStats.from_datasets(datasets)

    def _parse_yaml(self, yaml_path: Path) -> Optional[dict]:
        """Parse a YAML file safely."""
        try:
            with open(yaml_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to parse {yaml_path}: {e}")
            return None

    def _count_images(self, images_dir: Path) -> int:
        """Count image files in a directory."""
        if not images_dir.is_dir():
            return 0

        count = 0
        for item in images_dir.iterdir():
            if item.is_file() and item.suffix.lower() in self.IMAGE_EXTENSIONS:
                count += 1
        return count

    def _analyze_labels(self, labels_dir: Path) -> dict[int, int]:
        """Analyze label files and count objects per class.

        YOLO OBB label format: class_id x1 y1 x2 y2 x3 y3 x4 y4
        """
        distribution: dict[int, int] = {}

        if not labels_dir.is_dir():
            return distribution

        for label_file in labels_dir.iterdir():
            if label_file.suffix.lower() != '.txt':
                continue

            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if parts:
                            try:
                                class_id = int(parts[0])
                                distribution[class_id] = distribution.get(class_id, 0) + 1
                            except ValueError:
                                continue
            except Exception as e:
                logger.warning(f"Error reading label file {label_file}: {e}")

        return distribution

    def _find_sample_image(self, images_dir: Path) -> Optional[str]:
        """Find a sample image from the directory."""
        if not images_dir.is_dir():
            return None

        for item in sorted(images_dir.iterdir()):
            if item.is_file() and item.suffix.lower() in self.IMAGE_EXTENSIONS:
                return str(item)

        return None
