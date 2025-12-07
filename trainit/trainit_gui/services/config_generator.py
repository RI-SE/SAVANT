"""Service for generating training configuration files."""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from ..models.project import Project, TrainingConfig
from ..models.dataset import DatasetInfo
from .dataset_service import DatasetService

logger = logging.getLogger(__name__)


class ConfigGenerator:
    """Service for generating YOLO training configuration files.

    Generates:
    1. Merged dataset YAML - combines selected datasets into single config
    2. Training params JSON - parameters for train_yolo_obb.py --config option
    """

    def __init__(self, dataset_service: Optional[DatasetService] = None):
        self.dataset_service = dataset_service or DatasetService()

    def generate_training_files(
        self,
        project: Project,
        config: TrainingConfig,
        output_dir: str,
        copy_images: bool = False
    ) -> tuple[str, str]:
        """Generate training configuration files for a config.

        Args:
            project: The project containing dataset references
            config: The training configuration to generate files for
            output_dir: Directory to write output files
            copy_images: If True, copy images to output dir (default: use symlinks)

        Returns:
            Tuple of (dataset_yaml_path, params_json_path)

        Raises:
            ValueError: If datasets have mismatched class definitions
            FileNotFoundError: If datasets don't exist
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load and validate all selected datasets
        datasets = self._load_selected_datasets(
            project.datasets_root,
            config.selected_datasets
        )

        # Validate class compatibility
        self._validate_class_compatibility(datasets)

        # Generate merged dataset YAML
        yaml_path = output_path / f"{config.name}_dataset.yaml"
        self._generate_merged_dataset_yaml(
            datasets=datasets,
            output_path=yaml_path,
            project=project,
            config=config,
            copy_images=copy_images,
            output_dir=output_path
        )

        # Generate training params JSON
        json_path = output_path / f"{config.name}_params.json"
        self._generate_params_json(
            config=config,
            dataset_yaml_path=str(yaml_path),
            output_path=json_path
        )

        logger.info(f"Generated training files in {output_dir}")
        return str(yaml_path), str(json_path)

    def _load_selected_datasets(
        self,
        datasets_root: str,
        dataset_names: list[str]
    ) -> list[DatasetInfo]:
        """Load DatasetInfo for all selected datasets."""
        root = Path(datasets_root)
        datasets = []

        for name in dataset_names:
            dataset_path = root / name
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found: {dataset_path}")

            info = self.dataset_service.load_dataset_info(str(dataset_path))
            if not info.is_valid:
                raise ValueError(f"Invalid dataset '{name}': {info.error_message}")

            datasets.append(info)

        return datasets

    def _validate_class_compatibility(self, datasets: list[DatasetInfo]) -> None:
        """Ensure all datasets have identical class definitions."""
        if len(datasets) < 2:
            return

        first = datasets[0]
        for ds in datasets[1:]:
            if not first.has_matching_classes(ds):
                raise ValueError(
                    f"Class mismatch between '{first.name}' and '{ds.name}'. "
                    f"All datasets must have identical class definitions."
                )

    def _generate_merged_dataset_yaml(
        self,
        datasets: list[DatasetInfo],
        output_path: Path,
        project: Project,
        config: TrainingConfig,
        copy_images: bool,
        output_dir: Path
    ) -> None:
        """Generate a merged dataset.yaml file.

        Creates symlinks (or copies) to combine multiple datasets.
        """
        # Use class names from first dataset (all validated to be identical)
        first_ds = datasets[0]

        # Create images and labels directories
        images_train_dir = output_dir / "images" / "train"
        images_val_dir = output_dir / "images" / "val"
        labels_train_dir = output_dir / "labels" / "train"
        labels_val_dir = output_dir / "labels" / "val"

        for d in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Link/copy files from each dataset with prefixed names
        for ds in datasets:
            ds_path = Path(ds.path)
            prefix = ds.name + "_"

            # Link train images and labels
            self._link_files(
                src_dir=ds_path / "images" / "train",
                dst_dir=images_train_dir,
                prefix=prefix,
                copy=copy_images
            )
            self._link_files(
                src_dir=ds_path / "labels" / "train",
                dst_dir=labels_train_dir,
                prefix=prefix,
                copy=copy_images
            )

            # Link val images and labels
            self._link_files(
                src_dir=ds_path / "images" / "val",
                dst_dir=images_val_dir,
                prefix=prefix,
                copy=copy_images
            )
            self._link_files(
                src_dir=ds_path / "labels" / "val",
                dst_dir=labels_val_dir,
                prefix=prefix,
                copy=copy_images
            )

        # Generate the YAML content
        yaml_content = {
            'path': str(output_dir.resolve()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': first_ds.num_classes,
            'names': first_ds.class_names
        }

        # Add metadata as comments
        header = f"""# Generated by trainit-gui
# Project: {project.name}
# Config: {config.name}
# Generated: {datetime.now().isoformat()}
# Source datasets: {', '.join(config.selected_datasets)}

"""
        with open(output_path, 'w') as f:
            f.write(header)
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Generated dataset YAML: {output_path}")

    def _link_files(
        self,
        src_dir: Path,
        dst_dir: Path,
        prefix: str,
        copy: bool = False
    ) -> None:
        """Link or copy files from source to destination with prefix."""
        if not src_dir.exists():
            return

        for src_file in src_dir.iterdir():
            if not src_file.is_file():
                continue

            dst_file = dst_dir / (prefix + src_file.name)

            if dst_file.exists():
                dst_file.unlink()

            if copy:
                shutil.copy2(src_file, dst_file)
            else:
                # Create symlink
                dst_file.symlink_to(src_file.resolve())

    def _generate_params_json(
        self,
        config: TrainingConfig,
        dataset_yaml_path: str,
        output_path: Path
    ) -> None:
        """Generate training parameters JSON file.

        This file can be used with train_yolo_obb.py --config option.
        """
        # Start with the data path
        params = {
            'data': dataset_yaml_path
        }

        # Add all non-default parameters from config
        params.update(config.get_non_default_params())

        # Add metadata
        params['_generated_at'] = datetime.now().isoformat()
        params['_config_name'] = config.name

        with open(output_path, 'w') as f:
            json.dump(params, f, indent=2, ensure_ascii=False)

        logger.info(f"Generated params JSON: {output_path}")

    def preview_generation(
        self,
        project: Project,
        config: TrainingConfig,
        output_dir: str
    ) -> dict:
        """Preview what files would be generated without creating them.

        Returns:
            Dict with preview information
        """
        output_path = Path(output_dir)

        # Load datasets for stats
        try:
            datasets = self._load_selected_datasets(
                project.datasets_root,
                config.selected_datasets
            )
            self._validate_class_compatibility(datasets)

            total_train = sum(ds.train_images for ds in datasets)
            total_val = sum(ds.val_images for ds in datasets)
            total_objects = sum(ds.total_objects for ds in datasets)

            return {
                'valid': True,
                'output_dir': str(output_path),
                'dataset_yaml': str(output_path / f"{config.name}_dataset.yaml"),
                'params_json': str(output_path / f"{config.name}_params.json"),
                'total_train_images': total_train,
                'total_val_images': total_val,
                'total_objects': total_objects,
                'num_classes': datasets[0].num_classes if datasets else 0,
                'datasets': [ds.name for ds in datasets]
            }

        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }
