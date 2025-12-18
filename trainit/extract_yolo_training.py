#!/usr/bin/env python3
"""
extract_yolo_training - Extract YOLO OBB training data from video with OpenLabel annotations

This tool extracts frames from a video file at regular intervals and creates a YOLO OBB
(Oriented Bounding Box) training dataset from the corresponding OpenLabel annotations.

Usage:
    extract-yolo-training -i /path/to/folder -o /path/to/output --ontology ontology.ttl

Arguments:
    --input, -i      Input folder containing video (.mp4) and OpenLabel JSON
    --output, -o     Output folder for YOLO dataset
    --ontology       Path to SAVANT ontology (.ttl) file

Example:
    extract-yolo-training -i TestVids/Kraklanda_short -o datasets/kraklanda \\
        --ontology schema/savant_ontology_1.3.1.ttl --interval-seconds 2 --train-ratio 0.9
"""

import argparse
import json
import logging
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import yaml

from savant_common import read_ontology_classes
from savant_common.openlabel import OpenLabel, load_openlabel
from trainit import __version__

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when input validation fails."""

    pass


class ExtractionError(Exception):
    """Raised when frame extraction fails."""

    pass


class ExtractionConfig:
    """Configuration for YOLO training data extraction."""

    DEFAULT_ONTOLOGY = Path("ontology/savant.ttl")

    def __init__(self, args: argparse.Namespace):
        """Initialize and validate configuration from CLI arguments."""
        self.input_path = Path(args.input)
        self.output_path = Path(args.output)
        self.ontology_path = (
            Path(args.ontology) if args.ontology else self.DEFAULT_ONTOLOGY
        )
        self.video_name: Optional[str] = args.video
        self.openlabel_name: Optional[str] = args.openlabel
        self.interval_seconds: Optional[float] = args.interval_seconds
        self.interval_frames: Optional[int] = args.interval_frames
        self.train_ratio: Optional[float] = args.train_ratio
        self.seed: int = args.seed
        self.verbose: bool = args.verbose
        self._ontology_was_default: bool = args.ontology is None

        # Resolved paths (set during validation)
        self.video_path: Optional[Path] = None
        self.openlabel_path: Optional[Path] = None

        self.validate()

    def validate(self) -> None:
        """Validate configuration and resolve file paths."""
        # Check input folder exists
        if not self.input_path.exists():
            raise ValidationError(f"Input folder not found: {self.input_path}")
        if not self.input_path.is_dir():
            raise ValidationError(f"Input path is not a directory: {self.input_path}")

        # Check ontology exists
        if not self.ontology_path.exists():
            if self._ontology_was_default:
                raise ValidationError(
                    f"Default ontology file not found: {self.ontology_path}\n"
                    f"Use --ontology to specify a different ontology file."
                )
            else:
                raise ValidationError(f"Ontology file not found: {self.ontology_path}")

        # Validate train ratio
        if self.train_ratio is not None:
            if not (0.0 < self.train_ratio < 1.0):
                raise ValidationError(
                    f"Train ratio must be between 0 and 1: {self.train_ratio}"
                )

        # Validate interval arguments (mutually exclusive)
        if self.interval_seconds is not None and self.interval_frames is not None:
            raise ValidationError(
                "Cannot specify both --interval-seconds and --interval-frames"
            )

        # Default to 5 seconds if neither specified
        if self.interval_seconds is None and self.interval_frames is None:
            self.interval_seconds = 5.0

        # Find video file
        self._resolve_video_path()

        # Find OpenLabel file
        self._resolve_openlabel_path()

    def _resolve_video_path(self) -> None:
        """Find and validate video file in input folder."""
        video_files = list(self.input_path.glob("*.mp4"))

        if not video_files:
            raise ValidationError(f"No .mp4 video files found in: {self.input_path}")

        if self.video_name:
            # User specified video
            video_path = self.input_path / self.video_name
            if not video_path.exists():
                raise ValidationError(f"Specified video not found: {video_path}")
            self.video_path = video_path
        elif len(video_files) == 1:
            self.video_path = video_files[0]
        else:
            # Multiple videos, need --video argument
            video_names = [v.name for v in video_files]
            raise ValidationError(
                "Multiple video files found. Use --video to specify:\n  "
                + "\n  ".join(video_names)
            )

    def _resolve_openlabel_path(self) -> None:
        """Find and validate OpenLabel file in input folder."""
        if self.openlabel_name:
            # User specified OpenLabel file (skip tagged_file check)
            openlabel_path = self.input_path / self.openlabel_name
            if not openlabel_path.exists():
                raise ValidationError(
                    f"Specified OpenLabel file not found: {openlabel_path}"
                )

            # Validate it's a valid OpenLabel file
            try:
                load_openlabel(openlabel_path)
            except ValueError as e:
                raise ValidationError(f"Invalid OpenLabel file: {e}")

            self.openlabel_path = openlabel_path
            return

        # Auto-detect: find JSON with matching tagged_file
        video_filename = self.video_path.name
        json_files = list(self.input_path.glob("*.json"))

        for json_file in json_files:
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)

                if "openlabel" not in data:
                    continue

                tagged_file = (
                    data.get("openlabel", {}).get("metadata", {}).get("tagged_file")
                )
                if tagged_file == video_filename:
                    self.openlabel_path = json_file
                    return

            except (json.JSONDecodeError, KeyError):
                continue

        raise ValidationError(
            f"No OpenLabel file found with tagged_file matching '{video_filename}'.\n"
            f"Use --openlabel to force a specific file."
        )


class YOLODatasetExtractor:
    """Extracts YOLO OBB training data from video and OpenLabel annotations."""

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.openlabel: Optional[OpenLabel] = None
        self.ontology_classes: List[Dict] = []
        self.class_map: Dict[str, int] = {}  # type_name -> yolo_class_id
        self.class_names: Dict[int, str] = {}  # yolo_class_id -> type_name
        self.video_cap: Optional[cv2.VideoCapture] = None
        self.fps: float = 0.0
        self.frame_count: int = 0
        self.frame_width: int = 0
        self.frame_height: int = 0

    def run(self) -> None:
        """Execute the extraction pipeline."""
        try:
            self._load_data()
            self._calculate_frame_indices()
            self._build_class_map()
            self._setup_output_directories()
            self._extract_frames()
            self._apply_train_val_split()
            self._generate_yaml_config()
            self._print_summary()
        finally:
            if self.video_cap is not None:
                self.video_cap.release()

    def _load_data(self) -> None:
        """Load OpenLabel, ontology, and video."""
        logger.info(f"Loading OpenLabel: {self.config.openlabel_path}")
        self.openlabel = load_openlabel(self.config.openlabel_path)

        logger.info(f"Loading ontology: {self.config.ontology_path}")
        self.ontology_classes = read_ontology_classes(str(self.config.ontology_path))

        logger.info(f"Opening video: {self.config.video_path}")
        self.video_cap = cv2.VideoCapture(str(self.config.video_path))
        if not self.video_cap.isOpened():
            raise ExtractionError(f"Failed to open video: {self.config.video_path}")

        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            f"Video: {self.frame_width}x{self.frame_height}, "
            f"{self.fps:.2f} fps, {self.frame_count} frames"
        )

    def _calculate_frame_indices(self) -> None:
        """Calculate which frame indices to extract."""
        if self.config.interval_seconds is not None:
            frame_interval = max(1, int(self.fps * self.config.interval_seconds))
            logger.info(
                f"Extracting every {self.config.interval_seconds}s "
                f"(~{frame_interval} frames)"
            )
        else:
            frame_interval = self.config.interval_frames
            logger.info(f"Extracting every {frame_interval} frames")

        self.frame_indices = list(range(0, self.frame_count, frame_interval))
        logger.info(f"Will extract {len(self.frame_indices)} frames")

    def _build_class_map(self) -> None:
        """Build mapping from OpenLabel types to YOLO class IDs using ontology UIDs.

        Uses ontology UIDs directly as YOLO class IDs. Only includes UIDs that form
        a consecutive sequence starting from 0. Objects with non-consecutive UIDs
        are skipped during extraction.
        """
        # Build UID -> label mapping from ontology and find consecutive UIDs
        all_uids = {}  # uid -> label
        for ont_class in self.ontology_classes:
            uid = ont_class.get("uid")
            label = ont_class.get("label")
            if uid is not None and label is not None:
                all_uids[uid] = label

        # Find consecutive sequence starting from 0
        consecutive_uids = set()
        uid = 0
        while uid in all_uids:
            consecutive_uids.add(uid)
            uid += 1

        if not consecutive_uids:
            raise ExtractionError(
                "No consecutive UIDs starting from 0 found in ontology."
            )

        max_consecutive_uid = max(consecutive_uids)
        logger.info(
            f"Ontology has {len(all_uids)} classes with UIDs, "
            f"using consecutive UIDs 0-{max_consecutive_uid} ({len(consecutive_uids)} classes)"
        )

        # Log any non-consecutive UIDs that will be excluded
        non_consecutive = sorted(set(all_uids.keys()) - consecutive_uids)
        if non_consecutive:
            logger.info(
                f"Excluding {len(non_consecutive)} non-consecutive UIDs: "
                f"{non_consecutive[:10]}{'...' if len(non_consecutive) > 10 else ''}"
            )

        # Build class_names with ALL consecutive UIDs (for YAML)
        for uid in sorted(consecutive_uids):
            self.class_names[uid] = all_uids[uid]

        # Build label -> uid mapping for consecutive UIDs only (case-insensitive)
        label_to_uid = {}
        for uid in consecutive_uids:
            label = all_uids[uid]
            label_to_uid[label.lower()] = uid

        # Collect unique types from dataset and map to UIDs
        unique_types = set()
        for frame_idx in self.frame_indices:
            boxes = self.openlabel.get_boxes_with_ids_for_frame(frame_idx)
            for _, obj_type, *_ in boxes:
                unique_types.add(obj_type)

        logger.info(f"Found {len(unique_types)} unique object types in annotations")

        # Build class_map for types that match consecutive UIDs
        matched_count = 0
        skipped_types = []

        for obj_type in sorted(unique_types):
            uid = label_to_uid.get(obj_type.lower())
            if uid is not None:
                self.class_map[obj_type] = uid
                matched_count += 1
            else:
                skipped_types.append(obj_type)
                logger.warning(
                    f"Object type '{obj_type}' not in consecutive UID range - will be skipped"
                )

        if not self.class_map:
            raise ExtractionError(
                "No object types matched consecutive ontology UIDs. Cannot create dataset."
            )

        logger.info(
            f"Class mapping: {matched_count} types matched consecutive UIDs, "
            f"{len(skipped_types)} skipped"
        )
        for obj_type, uid in sorted(self.class_map.items(), key=lambda x: x[1]):
            logger.debug(f"  {obj_type} -> UID {uid}")

    def _setup_output_directories(self) -> None:
        """Create output directory structure."""
        self.config.output_path.mkdir(parents=True, exist_ok=True)

        # Initial structure (before split)
        (self.config.output_path / "images").mkdir(exist_ok=True)
        (self.config.output_path / "labels").mkdir(exist_ok=True)

    def _extract_frames(self) -> None:
        """Extract frames and create label files."""
        images_dir = self.config.output_path / "images"
        labels_dir = self.config.output_path / "labels"

        total_frames = len(self.frame_indices)
        extracted = 0
        skipped_annotations = 0

        for i, frame_idx in enumerate(self.frame_indices):
            if (i + 1) % 100 == 0 or i == 0:
                logger.info(f"Processing frame {i + 1}/{total_frames}")

            # Seek to frame
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.video_cap.read()

            if not ret:
                logger.warning(f"Failed to read frame {frame_idx}")
                continue

            # Generate filename
            frame_name = f"frame_{frame_idx:06d}"

            # Save image
            image_path = images_dir / f"{frame_name}.jpg"
            cv2.imwrite(str(image_path), frame)

            # Get annotations and convert to YOLO OBB format
            boxes = self.openlabel.get_boxes_with_ids_for_frame(frame_idx)
            label_lines = []

            for obj_id, obj_type, x_center, y_center, width, height, rotation in boxes:
                if obj_type not in self.class_map:
                    skipped_annotations += 1
                    continue

                class_id = self.class_map[obj_type]

                # Convert to YOLO OBB format (normalized 4 corners)
                corners = self._bbox_to_corners(
                    x_center,
                    y_center,
                    width,
                    height,
                    rotation,
                    self.frame_width,
                    self.frame_height,
                )

                if corners is not None:
                    # Format: class_id x1 y1 x2 y2 x3 y3 x4 y4
                    line = f"{class_id} " + " ".join(f"{c:.6f}" for c in corners)
                    label_lines.append(line)

            # Save label file (even if empty)
            label_path = labels_dir / f"{frame_name}.txt"
            with open(label_path, "w") as f:
                f.write("\n".join(label_lines))

            extracted += 1

        logger.info(
            f"Extracted {extracted} frames, skipped {skipped_annotations} annotations"
        )

    def _bbox_to_corners(
        self,
        x_center: float,
        y_center: float,
        width: float,
        height: float,
        rotation: float,
        img_width: int,
        img_height: int,
    ) -> Optional[List[float]]:
        """Convert center/size/rotation bbox to normalized corner points.

        Args:
            x_center, y_center: Center in pixels
            width, height: Size in pixels
            rotation: Rotation in radians
            img_width, img_height: Image dimensions for normalization

        Returns:
            List of 8 floats [x1,y1,x2,y2,x3,y3,x4,y4] normalized to [0,1],
            or None if conversion fails
        """
        try:
            # Half dimensions
            hw = width / 2
            hh = height / 2

            # Corner offsets (before rotation)
            # Order: top-left, top-right, bottom-right, bottom-left
            corners_local = [
                (-hw, -hh),  # top-left
                (hw, -hh),  # top-right
                (hw, hh),  # bottom-right
                (-hw, hh),  # bottom-left
            ]

            # Rotation matrix components
            cos_r = math.cos(rotation)
            sin_r = math.sin(rotation)

            # Apply rotation and translate to center
            corners = []
            for dx, dy in corners_local:
                # Rotate
                rx = dx * cos_r - dy * sin_r
                ry = dx * sin_r + dy * cos_r
                # Translate to center and normalize
                x_norm = (x_center + rx) / img_width
                y_norm = (y_center + ry) / img_height
                # Clamp to [0, 1]
                x_norm = max(0.0, min(1.0, x_norm))
                y_norm = max(0.0, min(1.0, y_norm))
                corners.extend([x_norm, y_norm])

            return corners

        except Exception as e:
            logger.warning(f"Failed to convert bbox: {e}")
            return None

    def _apply_train_val_split(self) -> None:
        """Split extracted data into train/val sets if requested."""
        if self.config.train_ratio is None:
            # No split - just create train directories and move files
            images_dir = self.config.output_path / "images"
            labels_dir = self.config.output_path / "labels"

            train_images = self.config.output_path / "images" / "train"
            train_labels = self.config.output_path / "labels" / "train"
            train_images.mkdir(exist_ok=True)
            train_labels.mkdir(exist_ok=True)

            # Move all files to train
            for img_file in images_dir.glob("*.jpg"):
                shutil.move(str(img_file), str(train_images / img_file.name))

            for lbl_file in labels_dir.glob("*.txt"):
                shutil.move(str(lbl_file), str(train_labels / lbl_file.name))

            self.train_count = len(list(train_images.glob("*.jpg")))
            self.val_count = 0
            return

        # Apply train/val split
        images_dir = self.config.output_path / "images"
        labels_dir = self.config.output_path / "labels"

        # Get all image files
        image_files = sorted(images_dir.glob("*.jpg"))

        # Shuffle with seed
        random.seed(self.config.seed)
        shuffled = list(image_files)
        random.shuffle(shuffled)

        # Split
        split_idx = int(len(shuffled) * self.config.train_ratio)
        train_files = shuffled[:split_idx]
        val_files = shuffled[split_idx:]

        # Create subdirectories
        train_images = images_dir / "train"
        val_images = images_dir / "val"
        train_labels = labels_dir / "train"
        val_labels = labels_dir / "val"

        for d in [train_images, val_images, train_labels, val_labels]:
            d.mkdir(exist_ok=True)

        # Move files
        for img_file in train_files:
            label_file = labels_dir / f"{img_file.stem}.txt"
            shutil.move(str(img_file), str(train_images / img_file.name))
            if label_file.exists():
                shutil.move(str(label_file), str(train_labels / label_file.name))

        for img_file in val_files:
            label_file = labels_dir / f"{img_file.stem}.txt"
            shutil.move(str(img_file), str(val_images / img_file.name))
            if label_file.exists():
                shutil.move(str(label_file), str(val_labels / label_file.name))

        self.train_count = len(train_files)
        self.val_count = len(val_files)

        logger.info(f"Split: {self.train_count} train, {self.val_count} val")

    def _generate_yaml_config(self) -> None:
        """Generate YOLO dataset YAML configuration."""
        yaml_path = self.config.output_path / "dataset.yaml"

        config = {
            "path": str(self.config.output_path.absolute()),
            "train": "images/train",
            "val": "images/val" if self.config.train_ratio else "images/train",
            "nc": len(self.class_names),
            "names": self.class_names,
        }

        with open(yaml_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"YAML config saved: {yaml_path}")

    def _print_summary(self) -> None:
        """Print extraction summary."""
        print("\n" + "=" * 60)
        print(
            f"EXTRACTION COMPLETE (SAVANT trainit - extract_yolo_training v{__version__})"
        )
        print("=" * 60)
        print(f"Output:     {self.config.output_path}")
        print(f"Classes:    {len(self.class_names)}")
        print(f"Train:      {self.train_count} images")
        if self.config.train_ratio:
            print(f"Val:        {self.val_count} images")
        print(f"Config:     {self.config.output_path / 'dataset.yaml'}")
        print("=" * 60)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract YOLO OBB training data from video with OpenLabel annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic extraction with default ontology and 5-second interval
  extract-yolo-training -i TestVids/video_folder -o datasets/output

  # Extract every 2 seconds with 90/10 train/val split
  extract-yolo-training -i input -o output --interval-seconds 2 --train-ratio 0.9

  # Extract every 30 frames with custom ontology
  extract-yolo-training -i input -o output --ontology custom_ontology.ttl --interval-frames 30

  # Specify video and OpenLabel files explicitly
  extract-yolo-training -i folder -o out --video clip.mp4 --openlabel annotations.json
""",
    )

    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    # Core arguments
    core = parser.add_argument_group("Core Arguments")
    core.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Input folder containing video (.mp4) and OpenLabel JSON",
    )
    core.add_argument(
        "-o", "--output", type=str, required=True, help="Output folder for YOLO dataset"
    )
    core.add_argument(
        "--ontology",
        type=str,
        default=None,
        help="Path to SAVANT ontology (.ttl) file (default: ontology/savant.ttl)",
    )

    # File selection
    files = parser.add_argument_group("File Selection")
    files.add_argument(
        "--video",
        type=str,
        default=None,
        help="Video filename (required if multiple .mp4 files in folder)",
    )
    files.add_argument(
        "--openlabel",
        type=str,
        default=None,
        help="Force specific OpenLabel JSON file (skips tagged_file check)",
    )

    # Frame extraction
    extraction = parser.add_argument_group("Frame Extraction")
    extraction.add_argument(
        "--interval-seconds",
        type=float,
        default=None,
        help="Extract every N seconds (default: 5.0)",
    )
    extraction.add_argument(
        "--interval-frames",
        type=int,
        default=None,
        help="Extract every N frames (mutually exclusive with --interval-seconds)",
    )

    # Dataset options
    dataset = parser.add_argument_group("Dataset Options")
    dataset.add_argument(
        "--train-ratio",
        type=float,
        default=None,
        help="Train/val split ratio (e.g., 0.9 for 90%% train). If not set, all goes to train.",
    )
    dataset.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split (default: 42)",
    )

    # Output options
    output = parser.add_argument_group("Output Options")
    output.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    try:
        args = parse_arguments()
        setup_logging(args.verbose)

        config = ExtractionConfig(args)
        extractor = YOLODatasetExtractor(config)
        extractor.run()

        sys.exit(0)

    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)
    except ExtractionError as e:
        logger.error(f"Extraction error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if logging.getLogger().level == logging.DEBUG:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
