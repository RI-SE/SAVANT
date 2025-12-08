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
from ..models.manifest import ManifestInfo
from .dataset_service import DatasetService
from .manifest_service import ManifestService
from .split_service import SplitService

logger = logging.getLogger(__name__)


class ConfigGenerator:
    """Service for generating YOLO training configuration files.

    Generates:
    1. Merged dataset YAML - combines selected datasets into single config
    2. Training params JSON - parameters for train_yolo_obb.py --config option
    """

    def __init__(
        self,
        dataset_service: Optional[DatasetService] = None,
        manifest_service: Optional[ManifestService] = None,
        split_service: Optional[SplitService] = None
    ):
        self.dataset_service = dataset_service or DatasetService()
        self.manifest_service = manifest_service or ManifestService()
        self.split_service = split_service or SplitService()

    def generate_training_files(
        self,
        project: Project,
        config: TrainingConfig,
        output_dir: str,
        copy_images: bool = False,
        generate_manifest: bool = True
    ) -> tuple[str, str, Optional[str]]:
        """Generate training configuration files for a config.

        Args:
            project: The project containing dataset references
            config: The training configuration to generate files for
            output_dir: Directory to write output files
            copy_images: If True, copy images to output dir (default: use symlinks)
            generate_manifest: If True, generate manifest JSON with file hashes

        Returns:
            Tuple of (dataset_yaml_path, params_json_path, manifest_json_path or None)

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

        # Create output directories
        images_train_dir = output_path / "images" / "train"
        images_val_dir = output_path / "images" / "val"
        labels_train_dir = output_path / "labels" / "train"
        labels_val_dir = output_path / "labels" / "val"

        for d in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Track file mappings for manifest: list of (dest_path, source_path)
        file_mappings: list[tuple[Path, Path]] = []
        split_method: Optional[str] = None

        if config.split_enabled:
            # Re-map train/val split
            file_mappings, split_method = self._generate_with_split(
                datasets=datasets,
                config=config,
                output_path=output_path,
                copy_images=copy_images
            )
        else:
            # Preserve source train/val structure
            file_mappings = self._generate_preserve_split(
                datasets=datasets,
                output_path=output_path,
                copy_images=copy_images
            )

        # Generate the dataset YAML
        yaml_path = output_path / f"{config.name}_dataset.yaml"
        self._generate_dataset_yaml(
            datasets=datasets,
            output_path=yaml_path,
            output_dir=output_path,
            project=project,
            config=config,
            split_method=split_method
        )

        # Generate training params JSON
        json_path = output_path / f"{config.name}_params.json"
        self._generate_params_json(
            config=config,
            dataset_yaml_path=str(yaml_path),
            output_path=json_path
        )

        # Generate manifest if requested
        manifest_path: Optional[str] = None
        if generate_manifest and file_mappings:
            manifest_file = output_path / f"{config.name}_manifest.json"
            manifest_info = ManifestInfo(
                generated_at=datetime.now().isoformat(),
                project_name=project.name,
                config_name=config.name,
                split_enabled=config.split_enabled,
                split_ratio=config.split_ratio if config.split_enabled else None,
                split_seed=config.split_seed if config.split_enabled else None,
                split_method=split_method,
                source_datasets=config.selected_datasets
            )
            manifest = self.manifest_service.generate_manifest(
                output_dir=output_path,
                info=manifest_info,
                file_mappings=file_mappings
            )
            self.manifest_service.save_manifest(manifest, manifest_file)
            manifest_path = str(manifest_file)

        logger.info(f"Generated training files in {output_dir}")
        return str(yaml_path), str(json_path), manifest_path

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

    def _generate_preserve_split(
        self,
        datasets: list[DatasetInfo],
        output_path: Path,
        copy_images: bool
    ) -> list[tuple[Path, Path]]:
        """Generate files preserving source train/val structure.

        Args:
            datasets: List of datasets to include
            output_path: Output directory
            copy_images: If True, copy files; otherwise create symlinks

        Returns:
            List of (dest_path, source_path) tuples for manifest
        """
        file_mappings: list[tuple[Path, Path]] = []

        images_train_dir = output_path / "images" / "train"
        images_val_dir = output_path / "images" / "val"
        labels_train_dir = output_path / "labels" / "train"
        labels_val_dir = output_path / "labels" / "val"

        for ds in datasets:
            ds_path = Path(ds.path)
            prefix = ds.name + "_"

            # Link train images and labels
            file_mappings.extend(self._link_files_with_tracking(
                src_dir=ds_path / "images" / "train",
                dst_dir=images_train_dir,
                prefix=prefix,
                copy=copy_images
            ))
            file_mappings.extend(self._link_files_with_tracking(
                src_dir=ds_path / "labels" / "train",
                dst_dir=labels_train_dir,
                prefix=prefix,
                copy=copy_images
            ))

            # Link val images and labels
            file_mappings.extend(self._link_files_with_tracking(
                src_dir=ds_path / "images" / "val",
                dst_dir=images_val_dir,
                prefix=prefix,
                copy=copy_images
            ))
            file_mappings.extend(self._link_files_with_tracking(
                src_dir=ds_path / "labels" / "val",
                dst_dir=labels_val_dir,
                prefix=prefix,
                copy=copy_images
            ))

        return file_mappings

    def _generate_with_split(
        self,
        datasets: list[DatasetInfo],
        config: TrainingConfig,
        output_path: Path,
        copy_images: bool
    ) -> tuple[list[tuple[Path, Path]], str]:
        """Generate files with train/val re-mapping.

        Args:
            datasets: List of datasets to include
            config: Training config with split settings
            output_path: Output directory
            copy_images: If True, copy files; otherwise create symlinks

        Returns:
            Tuple of (file_mappings, split_method)
        """
        file_mappings: list[tuple[Path, Path]] = []

        # Collect all files from all datasets
        all_files: list[tuple[Path, str]] = []  # (file_path, dataset_name)

        for ds in datasets:
            ds_path = Path(ds.path)
            include_val = config.include_val_in_pool.get(ds.name, False)

            files = self.split_service.collect_files_from_dataset(ds_path, include_val)
            all_files.extend((f, ds.name) for f in files)

        # Perform split
        just_files = [f for f, _ in all_files]
        split_result = self.split_service.split_files(
            files=just_files,
            train_ratio=config.split_ratio,
            seed=config.split_seed
        )

        # Create file_to_dataset mapping
        file_to_ds = {f: ds for f, ds in all_files}

        images_train_dir = output_path / "images" / "train"
        images_val_dir = output_path / "images" / "val"
        labels_train_dir = output_path / "labels" / "train"
        labels_val_dir = output_path / "labels" / "val"

        # Process train files
        for src_file in split_result.train_files:
            ds_name = file_to_ds[src_file]
            prefix = ds_name + "_"

            # Link image
            img_dst = images_train_dir / (prefix + src_file.name)
            self._create_link_or_copy(src_file, img_dst, copy_images)
            file_mappings.append((img_dst, src_file))

            # Link corresponding label
            label_src = self._find_label_for_image(src_file)
            if label_src:
                label_dst = labels_train_dir / (prefix + label_src.name)
                self._create_link_or_copy(label_src, label_dst, copy_images)
                file_mappings.append((label_dst, label_src))

        # Process val files
        for src_file in split_result.val_files:
            ds_name = file_to_ds[src_file]
            prefix = ds_name + "_"

            # Link image
            img_dst = images_val_dir / (prefix + src_file.name)
            self._create_link_or_copy(src_file, img_dst, copy_images)
            file_mappings.append((img_dst, src_file))

            # Link corresponding label
            label_src = self._find_label_for_image(src_file)
            if label_src:
                label_dst = labels_val_dir / (prefix + label_src.name)
                self._create_link_or_copy(label_src, label_dst, copy_images)
                file_mappings.append((label_dst, label_src))

        logger.info(
            f"Split generated: {len(split_result.train_files)} train, "
            f"{len(split_result.val_files)} val ({split_result.method})"
        )

        return file_mappings, split_result.method

    def _find_label_for_image(self, image_path: Path) -> Optional[Path]:
        """Find label file corresponding to image.

        Checks in both train and val label directories.
        """
        label_name = image_path.stem + ".txt"

        # Determine which split the image is in
        if "train" in str(image_path):
            labels_dir = image_path.parent.parent.parent / "labels" / "train"
        else:
            labels_dir = image_path.parent.parent.parent / "labels" / "val"

        label_path = labels_dir / label_name
        if label_path.exists():
            return label_path

        # Check other folder (for pooled val case)
        other_split = "val" if "train" in str(labels_dir) else "train"
        other_dir = labels_dir.parent / other_split
        other_path = other_dir / label_name
        if other_path.exists():
            return other_path

        return None

    def _create_link_or_copy(self, src: Path, dst: Path, copy: bool) -> None:
        """Create symlink or copy file."""
        if dst.exists():
            dst.unlink()

        if copy:
            shutil.copy2(src, dst)
        else:
            dst.symlink_to(src.resolve())

    def _generate_dataset_yaml(
        self,
        datasets: list[DatasetInfo],
        output_path: Path,
        output_dir: Path,
        project: Project,
        config: TrainingConfig,
        split_method: Optional[str]
    ) -> None:
        """Generate the dataset YAML file."""
        first_ds = datasets[0]

        yaml_content = {
            'path': str(output_dir.resolve()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': first_ds.num_classes,
            'names': first_ds.class_names
        }

        # Add metadata as comments
        split_info = ""
        if config.split_enabled:
            split_info = f"\n# Split: {config.split_ratio:.0%} train ({split_method}), seed={config.split_seed}"

        header = f"""# Generated by trainit-gui
# Project: {project.name}
# Config: {config.name}
# Generated: {datetime.now().isoformat()}
# Source datasets: {', '.join(config.selected_datasets)}{split_info}

"""
        with open(output_path, 'w') as f:
            f.write(header)
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Generated dataset YAML: {output_path}")

    def _link_files_with_tracking(
        self,
        src_dir: Path,
        dst_dir: Path,
        prefix: str,
        copy: bool = False
    ) -> list[tuple[Path, Path]]:
        """Link or copy files from source to destination with prefix.

        Returns:
            List of (dest_path, source_path) tuples for manifest
        """
        mappings: list[tuple[Path, Path]] = []

        if not src_dir.exists():
            return mappings

        for src_file in src_dir.iterdir():
            if not src_file.is_file():
                continue

            dst_file = dst_dir / (prefix + src_file.name)
            self._create_link_or_copy(src_file, dst_file, copy)
            mappings.append((dst_file, src_file))

        return mappings

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
        output_dir: str,
        generate_manifest: bool = True
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

            # Calculate totals considering split settings
            if config.split_enabled:
                # Count total files that would be pooled
                total_pooled = 0
                for ds in datasets:
                    ds_path = Path(ds.path)
                    include_val = config.include_val_in_pool.get(ds.name, False)
                    files = self.split_service.collect_files_from_dataset(
                        ds_path, include_val
                    )
                    total_pooled += len(files)

                estimated_train = int(total_pooled * config.split_ratio)
                estimated_val = total_pooled - estimated_train
            else:
                estimated_train = total_train
                estimated_val = total_val

            result = {
                'valid': True,
                'output_dir': str(output_path),
                'dataset_yaml': str(output_path / f"{config.name}_dataset.yaml"),
                'params_json': str(output_path / f"{config.name}_params.json"),
                'total_train_images': estimated_train,
                'total_val_images': estimated_val,
                'total_objects': total_objects,
                'num_classes': datasets[0].num_classes if datasets else 0,
                'datasets': [ds.name for ds in datasets],
                'split_enabled': config.split_enabled,
                'split_ratio': config.split_ratio if config.split_enabled else None,
                'split_seed': config.split_seed if config.split_enabled else None,
            }

            if generate_manifest:
                result['manifest_json'] = str(
                    output_path / f"{config.name}_manifest.json"
                )

            return result

        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }
