#!/usr/bin/env python3
"""
split_train_val.py

Splits train dataset into train/val by taking a percentage of images from EACH sequence,
ensuring both datasets have diverse representation across all video sequences.

This script:
1. Identifies all sequences in the train directory
2. Calculates how many images from each sequence should go to validation
3. Randomly selects images from each sequence for validation
4. Moves selected images and their corresponding labels to val directories

Usage:
    python split_train_val.py [options]

Arguments:
    --train-ratio      Ratio of data to keep in train (default: 0.9 for 90/10 split)
    --dataset-path     Path to dataset root (default: datasets/UAV_yolo_obb)
    --seed             Random seed for reproducibility (default: 42)
    --dry-run          Show what would be moved without actually moving
    --restore          Restore val data back to train (undo split)
    --remove-orphans   Remove images without corresponding label files
    --verbose, -v      Show detailed output

Examples:
    # Default 90/10 split
    python split_train_val.py

    # 80/20 split
    python split_train_val.py --train-ratio 0.8

    # Remove orphaned images before splitting
    python split_train_val.py --remove-orphans

    # Dry run to see what would happen
    python split_train_val.py --dry-run

    # Restore val back to train
    python split_train_val.py --restore

    # Custom dataset path with 85/15 split
    python split_train_val.py --dataset-path /path/to/dataset --train-ratio 0.85
"""

import argparse
import shutil
import sys
from pathlib import Path
from collections import defaultdict
import random


class DatasetSplitter:
    """Handles splitting train/val datasets by sequence."""

    def __init__(self, dataset_path: Path, train_ratio: float = 0.9,
                 seed: int = 42, verbose: bool = False):
        """Initialize dataset splitter.

        Args:
            dataset_path: Path to dataset root directory
            train_ratio: Ratio of data to keep in train (0.0 to 1.0)
            seed: Random seed for reproducibility
            verbose: Enable verbose output
        """
        self.dataset_path = dataset_path
        self.train_ratio = train_ratio
        self.val_ratio = 1.0 - train_ratio
        self.seed = seed
        self.verbose = verbose

        # Set random seed
        random.seed(seed)

        # Define directories
        self.train_images_dir = dataset_path / "images" / "train"
        self.train_labels_dir = dataset_path / "labels" / "train"
        self.val_images_dir = dataset_path / "images" / "val"
        self.val_labels_dir = dataset_path / "labels" / "val"

    def validate_directories(self) -> bool:
        """Validate that required directories exist.

        Returns:
            True if valid, False otherwise
        """
        if not self.train_images_dir.exists():
            print(f"Error: Train images directory not found: {self.train_images_dir}")
            return False

        if not self.train_labels_dir.exists():
            print(f"Error: Train labels directory not found: {self.train_labels_dir}")
            return False

        return True

    def extract_sequence_id(self, filename: str) -> str:
        """Extract sequence ID from filename.

        Args:
            filename: Image filename (e.g., "M0101_00001.jpg")

        Returns:
            Sequence ID (e.g., "M0101")
        """
        # Assuming format: M####_#####.jpg or similar
        # Extract the part before the first underscore
        return filename.split('_')[0]

    def group_images_by_sequence(self) -> dict:
        """Group all train images by their sequence ID.

        Returns:
            Dictionary mapping sequence_id -> list of image paths
        """
        sequences = defaultdict(list)

        for img_path in sorted(self.train_images_dir.glob("*.*")):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                seq_id = self.extract_sequence_id(img_path.name)
                sequences[seq_id].append(img_path)

        return dict(sequences)

    def get_label_path(self, image_path: Path) -> Path:
        """Get corresponding label path for an image.

        Args:
            image_path: Path to image file

        Returns:
            Path to corresponding label file
        """
        label_name = image_path.stem + '.txt'
        return self.train_labels_dir / label_name

    def create_val_directories(self):
        """Create validation directories if they don't exist."""
        self.val_images_dir.mkdir(parents=True, exist_ok=True)
        self.val_labels_dir.mkdir(parents=True, exist_ok=True)

    def split_dataset(self, dry_run: bool = False) -> dict:
        """Split dataset into train/val by sampling from each sequence.

        Args:
            dry_run: If True, don't actually move files

        Returns:
            Dictionary with split statistics
        """
        if not self.validate_directories():
            return {}

        print(f"\nSplitting dataset with {self.train_ratio:.1%} train / {self.val_ratio:.1%} val")
        print(f"Dataset path: {self.dataset_path}")
        print(f"Random seed: {self.seed}")

        # Group images by sequence
        print("\nGrouping images by sequence...")
        sequences = self.group_images_by_sequence()

        if not sequences:
            print("Error: No images found in train directory")
            return {}

        print(f"Found {len(sequences)} sequences:")
        for seq_id, images in sorted(sequences.items()):
            print(f"  {seq_id}: {len(images)} images")

        # Create validation directories
        if not dry_run:
            self.create_val_directories()

        # Split each sequence
        stats = {
            'sequences': len(sequences),
            'total_images': 0,
            'train_images': 0,
            'val_images': 0,
            'moved_images': 0,
            'moved_labels': 0,
            'missing_labels': 0,
            'sequence_details': {}
        }

        print(f"\n{'Dry run - ' if dry_run else ''}Splitting sequences:")

        for seq_id, images in sorted(sequences.items()):
            num_images = len(images)
            num_val = int(num_images * self.val_ratio)
            num_train = num_images - num_val

            stats['total_images'] += num_images
            stats['train_images'] += num_train
            stats['val_images'] += num_val

            # Randomly select images for validation
            val_images = random.sample(images, num_val)

            # Move images and labels
            moved_images = 0
            moved_labels = 0
            missing_labels = 0

            for img_path in val_images:
                label_path = self.get_label_path(img_path)

                # Check if label exists
                if not label_path.exists():
                    missing_labels += 1
                    if self.verbose:
                        print(f"  Warning: Label not found for {img_path.name}")
                    continue

                if not dry_run:
                    # Move image
                    dest_img = self.val_images_dir / img_path.name
                    shutil.move(str(img_path), str(dest_img))
                    moved_images += 1

                    # Move label
                    dest_label = self.val_labels_dir / label_path.name
                    shutil.move(str(label_path), str(dest_label))
                    moved_labels += 1
                else:
                    moved_images += 1
                    moved_labels += 1

            stats['moved_images'] += moved_images
            stats['moved_labels'] += moved_labels
            stats['missing_labels'] += missing_labels

            stats['sequence_details'][seq_id] = {
                'total': num_images,
                'train': num_train,
                'val': num_val,
                'moved': moved_images
            }

            print(f"  {seq_id}: {num_train} train / {num_val} val "
                  f"({moved_images} {'would be ' if dry_run else ''}moved)")

        return stats

    def restore_dataset(self, dry_run: bool = False) -> dict:
        """Restore validation data back to training.

        Args:
            dry_run: If True, don't actually move files

        Returns:
            Dictionary with restore statistics
        """
        if not self.val_images_dir.exists() or not self.val_labels_dir.exists():
            print("Error: Validation directories not found")
            return {}

        print(f"\n{'Dry run - ' if dry_run else ''}Restoring validation data to training...")

        stats = {
            'restored_images': 0,
            'restored_labels': 0,
            'missing_labels': 0
        }

        # Restore images
        for img_path in self.val_images_dir.glob("*.*"):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                label_name = img_path.stem + '.txt'
                label_path = self.val_labels_dir / label_name

                if not label_path.exists():
                    stats['missing_labels'] += 1
                    if self.verbose:
                        print(f"  Warning: Label not found for {img_path.name}")
                    continue

                if not dry_run:
                    # Move image back to train
                    dest_img = self.train_images_dir / img_path.name
                    shutil.move(str(img_path), str(dest_img))
                    stats['restored_images'] += 1

                    # Move label back to train
                    dest_label = self.train_labels_dir / label_name
                    shutil.move(str(label_path), str(dest_label))
                    stats['restored_labels'] += 1
                else:
                    stats['restored_images'] += 1
                    stats['restored_labels'] += 1

        print(f"  Images: {stats['restored_images']} {'would be ' if dry_run else ''}restored")
        print(f"  Labels: {stats['restored_labels']} {'would be ' if dry_run else ''}restored")
        if stats['missing_labels'] > 0:
            print(f"  Missing labels: {stats['missing_labels']}")

        return stats

    def remove_orphaned_images(self, dry_run: bool = False) -> dict:
        """Remove images that don't have corresponding label files.

        Args:
            dry_run: If True, don't actually remove files

        Returns:
            Dictionary with removal statistics
        """
        print(f"\n{'Dry run - ' if dry_run else ''}Checking for orphaned images...")

        stats = {
            'train_orphans': 0,
            'val_orphans': 0,
            'total_orphans': 0,
            'orphan_files': []
        }

        # Check train directory
        if self.train_images_dir.exists():
            print(f"Checking train images directory...")
            for img_path in self.train_images_dir.glob("*.*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    label_path = self.get_label_path(img_path)
                    if not label_path.exists():
                        stats['train_orphans'] += 1
                        stats['orphan_files'].append(str(img_path))

                        if self.verbose:
                            print(f"  Orphan: {img_path.name}")

                        if not dry_run:
                            img_path.unlink()

        # Check val directory
        if self.val_images_dir.exists():
            print(f"Checking val images directory...")
            for img_path in self.val_images_dir.glob("*.*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    label_name = img_path.stem + '.txt'
                    label_path = self.val_labels_dir / label_name

                    if not label_path.exists():
                        stats['val_orphans'] += 1
                        stats['orphan_files'].append(str(img_path))

                        if self.verbose:
                            print(f"  Orphan: {img_path.name}")

                        if not dry_run:
                            img_path.unlink()

        stats['total_orphans'] = stats['train_orphans'] + stats['val_orphans']

        return stats

    def print_orphan_statistics(self, stats: dict, dry_run: bool = False):
        """Print orphan removal statistics.

        Args:
            stats: Statistics dictionary from remove_orphaned_images()
            dry_run: Whether this was a dry run
        """
        if not stats:
            return

        print("\n" + "="*60)
        print("ORPHAN REMOVAL SUMMARY")
        print("="*60)

        action = "found" if dry_run else "removed"

        if stats['total_orphans'] == 0:
            print(f"No orphaned images found!")
        else:
            print(f"Train orphans {action}: {stats['train_orphans']}")
            print(f"Val orphans {action}: {stats['val_orphans']}")
            print(f"Total orphans {action}: {stats['total_orphans']}")

            if dry_run:
                print("\nRun without --dry-run to actually remove these files")

        print("="*60)

    def print_statistics(self, stats: dict):
        """Print split statistics.

        Args:
            stats: Statistics dictionary from split_dataset()
        """
        if not stats:
            return

        print("\n" + "="*60)
        print("SPLIT SUMMARY")
        print("="*60)
        print(f"Total sequences: {stats['sequences']}")
        print(f"Total images: {stats['total_images']}")
        print(f"Train images: {stats['train_images']} ({stats['train_images']/stats['total_images']:.1%})")
        print(f"Val images: {stats['val_images']} ({stats['val_images']/stats['total_images']:.1%})")
        print(f"Moved images: {stats['moved_images']}")
        print(f"Moved labels: {stats['moved_labels']}")
        if stats['missing_labels'] > 0:
            print(f"Missing labels: {stats['missing_labels']} (skipped)")
        print("="*60)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Split train dataset into train/val by sampling from each sequence',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--train-ratio', type=float, default=0.9,
                       help='Ratio of data to keep in train (default: 0.9 for 90/10 split)')
    parser.add_argument('--dataset-path', type=str, default='datasets/UAV_yolo_obb',
                       help='Path to dataset root directory (default: datasets/UAV_yolo_obb)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be moved without actually moving')
    parser.add_argument('--restore', action='store_true',
                       help='Restore val data back to train (undo split)')
    parser.add_argument('--remove-orphans', action='store_true',
                       help='Remove images without corresponding label files')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Show verbose output')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    # Validate train ratio
    if not 0.0 < args.train_ratio < 1.0:
        print(f"Error: train-ratio must be between 0.0 and 1.0, got {args.train_ratio}")
        sys.exit(1)

    # Convert path to Path object
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path not found: {dataset_path}")
        sys.exit(1)

    # Create splitter
    splitter = DatasetSplitter(
        dataset_path=dataset_path,
        train_ratio=args.train_ratio,
        seed=args.seed,
        verbose=args.verbose
    )

    try:
        # Handle orphan removal
        if args.remove_orphans:
            orphan_stats = splitter.remove_orphaned_images(dry_run=args.dry_run)
            splitter.print_orphan_statistics(orphan_stats, dry_run=args.dry_run)

            if args.dry_run:
                print("\nDry run complete - no files were removed")

            # If only removing orphans (not splitting), exit here
            if args.restore or not args.train_ratio:
                sys.exit(0)

        if args.restore:
            # Restore validation to training
            stats = splitter.restore_dataset(dry_run=args.dry_run)
            if args.dry_run:
                print("\nDry run complete - no files were moved")
        else:
            # Split dataset
            stats = splitter.split_dataset(dry_run=args.dry_run)
            splitter.print_statistics(stats)
            if args.dry_run:
                print("\nDry run complete - no files were moved")
                print("Run without --dry-run to actually perform the split")

        sys.exit(0)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
