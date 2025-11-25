#!/usr/bin/env python3
"""
remap_classes_to_ontology.py

Script to replace existing YOLO class IDs in some dataset with SAVANT ontology UIDs in label files and .yaml.

Usage:
    python remap_classes_to_ontology.py --dry-run          # Preview changes without modifying
    python remap_classes_to_ontology.py                    # Execute changes
    python remap_classes_to_ontology.py --labels-only      # Only update label files
    python remap_classes_to_ontology.py --yaml-only        # Only update .yaml
"""

import argparse
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path to import common package
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.ontology import read_ontology_classes


class ClassRemapper:
    """Handles remapping of YOLO class IDs to SAVANT ontology UIDs."""

    def __init__(self, yaml_path: str, ontology_path: str, labels_base_dir: str):
        """Initialize the remapper.

        Args:
            yaml_path: Path to UAV.yaml file
            ontology_path: Path to SAVANT ontology .ttl file
            labels_base_dir: Base directory containing labels/train and labels/val
        """
        self.yaml_path = yaml_path
        self.ontology_path = ontology_path
        self.labels_base_dir = Path(labels_base_dir)
        self.mapping = {}
        self.ontology_classes = []
        self.yaml_data = None

    def load_yaml(self) -> Dict:
        """Load UAV.yaml file."""
        print(f"Loading {self.yaml_path}...")
        with open(self.yaml_path, 'r') as f:
            self.yaml_data = yaml.safe_load(f)
        return self.yaml_data

    def load_ontology(self) -> List[Dict]:
        """Load SAVANT ontology classes."""
        print(f"Loading ontology from {self.ontology_path}...")
        self.ontology_classes = read_ontology_classes(self.ontology_path)
        return self.ontology_classes

    def create_mapping(self) -> Dict[int, int]:
        """Create mapping from old class IDs to new ontology UIDs.

        Returns:
            Dictionary mapping old_id -> new_uid
        """
        if not self.yaml_data or not self.ontology_classes:
            raise RuntimeError("Must load yaml and ontology first")

        class_names = self.yaml_data.get('names', {})
        print(f"\nFound {len(class_names)} classes in UAV.yaml:")

        # Create ontology lookup by label (case-insensitive)
        # Only include classes with valid UIDs (not None)
        ontology_lookup = {
            c['label'].lower(): c for c in self.ontology_classes
            if c['label'] and c['uid'] is not None
        }

        # Create mapping
        mapping = {}
        for old_id, class_name in class_names.items():
            class_name_lower = class_name.lower()

            # Try exact match first
            if class_name_lower in ontology_lookup:
                ontology_class = ontology_lookup[class_name_lower]
                mapping[int(old_id)] = ontology_class['uid']
                print(f"  {old_id}: {class_name:<15} → UID {ontology_class['uid']:>3} "
                      f"({ontology_class['class_name']})")
            else:
                print(f"  {old_id}: {class_name:<15} → NOT FOUND IN ONTOLOGY")
                raise ValueError(f"No ontology match found for class '{class_name}'")

        self.mapping = mapping
        return mapping

    def get_label_files(self) -> List[Path]:
        """Get all label files in train and val directories.

        Returns:
            List of Path objects for all .txt files
        """
        label_files = []

        for split in ['train', 'val']:
            split_dir = self.labels_base_dir / split
            if split_dir.exists():
                txt_files = list(split_dir.glob('*.txt'))
                label_files.extend(txt_files)
                print(f"Found {len(txt_files)} files in {split}/")

        return label_files

    def update_label_file(self, file_path: Path, dry_run: bool = False) -> Tuple[int, Dict[int, int]]:
        """Update class IDs in a single label file.

        Args:
            file_path: Path to label file
            dry_run: If True, don't write changes

        Returns:
            Tuple of (lines_updated, class_counts)
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()

        updated_lines = []
        lines_changed = 0
        class_counts = {}

        for line in lines:
            line = line.strip()
            if not line:
                updated_lines.append(line)
                continue

            parts = line.split()
            if len(parts) < 9:  # OBB format: class_id + 8 coordinates
                updated_lines.append(line)
                continue

            try:
                old_class_id = int(parts[0])

                if old_class_id in self.mapping:
                    new_uid = self.mapping[old_class_id]
                    parts[0] = str(new_uid)
                    lines_changed += 1
                    class_counts[new_uid] = class_counts.get(new_uid, 0) + 1

                updated_lines.append(' '.join(parts))
            except ValueError:
                # If first part is not an integer, keep line as-is
                updated_lines.append(line)

        # Write back if not dry run
        if not dry_run and lines_changed > 0:
            with open(file_path, 'w') as f:
                f.write('\n'.join(updated_lines) + '\n')

        return lines_changed, class_counts

    def update_all_labels(self, dry_run: bool = False) -> None:
        """Update all label files.

        Args:
            dry_run: If True, don't write changes
        """
        label_files = self.get_label_files()
        total_files = len(label_files)

        if total_files == 0:
            print("No label files found!")
            return

        print(f"\n{'DRY RUN: ' if dry_run else ''}Updating {total_files} label files...")

        total_lines_updated = 0
        total_class_counts = {}

        for i, file_path in enumerate(label_files, 1):
            lines_updated, class_counts = self.update_label_file(file_path, dry_run)
            total_lines_updated += lines_updated

            # Merge class counts
            for uid, count in class_counts.items():
                total_class_counts[uid] = total_class_counts.get(uid, 0) + count

            # Progress indicator
            if i % 1000 == 0 or i == total_files:
                print(f"  Processed {i}/{total_files} files "
                      f"({i*100//total_files}%) - {total_lines_updated} labels updated")

        print(f"\n{'Would update' if dry_run else 'Updated'} {total_lines_updated} labels across {total_files} files")
        print("\nClass distribution:")
        for uid in sorted(total_class_counts.keys()):
            # Find class name
            class_name = next((c['label'] for c in self.ontology_classes if c['uid'] == uid), 'Unknown')
            print(f"  UID {uid} ({class_name}): {total_class_counts[uid]} instances")

    def update_yaml(self, dry_run: bool = False) -> None:
        """Update UAV.yaml with new UIDs.

        Args:
            dry_run: If True, don't write changes
        """
        if not self.yaml_data:
            raise RuntimeError("Must load yaml first")

        # Create backup
        backup_path = f"{self.yaml_path}.backup"
        if not dry_run:
            print(f"\nCreating backup: {backup_path}")
            with open(self.yaml_path, 'r') as f_in:
                with open(backup_path, 'w') as f_out:
                    f_out.write(f_in.read())

        # Update names section
        old_names = self.yaml_data.get('names', {})
        new_names = {}

        print(f"\n{'DRY RUN: ' if dry_run else ''}Updating UAV.yaml class IDs:")
        for old_id, class_name in old_names.items():
            if int(old_id) in self.mapping:
                new_uid = self.mapping[int(old_id)]
                new_names[new_uid] = class_name
                print(f"  {old_id} → {new_uid}: {class_name}")
            else:
                # Keep unmapped classes as-is
                new_names[old_id] = class_name
                print(f"  {old_id} (unchanged): {class_name}")

        self.yaml_data['names'] = new_names

        # Write updated YAML
        if not dry_run:
            with open(self.yaml_path, 'w') as f:
                yaml.dump(self.yaml_data, f, default_flow_style=False, sort_keys=False)
            print(f"\n✓ Updated {self.yaml_path}")
        else:
            print(f"\nWould update {self.yaml_path} with:")
            print(yaml.dump({'names': new_names}, default_flow_style=False, sort_keys=False))


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Remap YOLO class IDs to SAVANT ontology UIDs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview changes without modifying files
  python remap_classes_to_ontology.py --dry-run

  # Execute remapping (updates both labels and YAML)
  python remap_classes_to_ontology.py

  # Only update label files
  python remap_classes_to_ontology.py --labels-only

  # Only update UAV.yaml
  python remap_classes_to_ontology.py --yaml-only
        """
    )

    parser.add_argument('--yaml', default='UAV.yaml',
                       help='Path to UAV.yaml file (default: UAV.yaml)')
    parser.add_argument('--ontology', default='savant_ontology_1.2.0.ttl',
                       help='Path to SAVANT ontology file (default: savant_ontology_1.2.0.ttl)')
    parser.add_argument('--labels-dir', default='datasets/UAV_yolo_obb/labels',
                       help='Base directory for label files (default: datasets/UAV_yolo_obb/labels)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview changes without modifying files')
    parser.add_argument('--labels-only', action='store_true',
                       help='Only update label files, skip UAV.yaml')
    parser.add_argument('--yaml-only', action='store_true',
                       help='Only update UAV.yaml, skip label files')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    if args.labels_only and args.yaml_only:
        print("Error: Cannot specify both --labels-only and --yaml-only")
        sys.exit(1)

    print("=" * 80)
    print("SAVANT Ontology Class Remapper")
    print("=" * 80)

    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No files will be modified\n")

    try:
        # Initialize remapper
        remapper = ClassRemapper(args.yaml, args.ontology, args.labels_dir)

        # Load data
        remapper.load_yaml()
        remapper.load_ontology()

        # Create mapping
        mapping = remapper.create_mapping()

        if not mapping:
            print("\nError: No class mappings created!")
            sys.exit(1)

        print(f"\nCreated mapping for {len(mapping)} classes")
        print("=" * 80)

        # Update files based on options
        if not args.yaml_only:
            remapper.update_all_labels(dry_run=args.dry_run)

        if not args.labels_only:
            remapper.update_yaml(dry_run=args.dry_run)

        print("\n" + "=" * 80)
        if args.dry_run:
            print("✓ DRY RUN COMPLETE - No files were modified")
            print("\nRun without --dry-run to apply changes:")
            print(f"  python {sys.argv[0]}")
        else:
            print("✓ REMAPPING COMPLETE")
            print(f"\nBackup created: {args.yaml}.backup")
        print("=" * 80)

    except FileNotFoundError as e:
        print(f"\nError: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
