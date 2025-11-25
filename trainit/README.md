# trainit

Training tools for YOLO OBB (Oriented Bounding Box) models.

## Overview

This directory contains tools for training and preparing datasets for YOLO OBB models:

- **train_yolo_obb.py** - Train YOLO OBB models for UAV object detection
- **split_train_val.py** - Split YOLO datasets into train/val by sequence

## Installation

These tools are installed as part of the SAVANT package:

```bash
# From repository root
uv sync --dev

# The scripts are available as CLI commands
train-yolo-obb --help
split-train-val --help
```

## Scripts

### train_yolo_obb.py

Trains YOLO OBB (Oriented Bounding Box) models for UAV object detection with advanced configuration options.

**Usage:**

```bash
# Basic training with YOLO defaults
train-yolo-obb --data dataset.yaml

# Custom model and training parameters
train-yolo-obb --data dataset.yaml --model yolo11m-obb.pt --epochs 100 --batch 16

# Custom learning rate and optimizer
train-yolo-obb --data dataset.yaml --lr0 0.001 --optimizer Adam

# Training with caching for speed
train-yolo-obb --data dataset.yaml --cache ram --workers 12

# Custom augmentation
train-yolo-obb --data dataset.yaml --degrees 15 --scale 0.8 --mosaic 0.8

# Resume training
train-yolo-obb --data dataset.yaml --resume
```

See `train-yolo-obb --help` for complete list of options.

---

### split_train_val.py

Splits YOLO train datasets into train/val sets by sampling from each video sequence, ensuring both datasets have diverse representation across all sequences.

**Usage:**

```bash
# Default 90/10 split
split-train-val --dataset-path datasets/UAV_yolo_obb

# Custom 80/20 split
split-train-val --dataset-path datasets/UAV_yolo_obb --train-ratio 0.8

# Dry run to preview the split
split-train-val --dataset-path datasets/UAV_yolo_obb --dry-run

# Remove orphaned images (images without labels) before splitting
split-train-val --dataset-path datasets/UAV_yolo_obb --remove-orphans

# Restore validation data back to training
split-train-val --dataset-path datasets/UAV_yolo_obb --restore

# Verbose output
split-train-val --dataset-path datasets/UAV_yolo_obb --verbose
```

**Arguments:**

- `--dataset-path` (required) - Path to dataset root directory
- `--train-ratio` - Ratio of data to keep in train (default: 0.9 for 90/10 split)
- `--seed` - Random seed for reproducibility (default: 42)
- `--dry-run` - Show what would be moved without actually moving files
- `--restore` - Restore val data back to train (undo split)
- `--remove-orphans` - Remove images without corresponding label files
- `--verbose`, `-v` - Show detailed output

**Dataset Structure Expected:**

```
dataset_root/
├── images/
│   └── train/
│       ├── M0101_00001.jpg
│       ├── M0101_00002.jpg
│       ├── M0201_00001.jpg
│       └── ...
└── labels/
    └── train/
        ├── M0101_00001.txt
        ├── M0101_00002.txt
        ├── M0201_00001.txt
        └── ...
```

After splitting, it creates:

```
dataset_root/
├── images/
│   ├── train/    # 90% of images from each sequence
│   └── val/      # 10% of images from each sequence
└── labels/
    ├── train/    # Corresponding labels
    └── val/      # Corresponding labels
```

**What it does:**

1. Groups images by sequence ID (extracted from filename, e.g., "M0101" from "M0101_00001.jpg")
2. Calculates how many images from each sequence should go to validation
3. Randomly selects images from each sequence for validation (ensuring even distribution)
4. Moves selected images and their labels to val directories
5. Preserves sequence diversity in both train and val sets

**Example Workflow:**

```bash
# Step 1: Preview the split
split-train-val --dataset-path datasets/UAV_yolo_obb --dry-run

# Step 2: Check for and remove orphaned images
split-train-val --dataset-path datasets/UAV_yolo_obb --remove-orphans --dry-run
split-train-val --dataset-path datasets/UAV_yolo_obb --remove-orphans

# Step 3: Perform the actual split
split-train-val --dataset-path datasets/UAV_yolo_obb

# Step 4: Train the model
train-yolo-obb --data datasets/UAV_yolo_obb/dataset.yaml --epochs 50

# If you need to undo the split
split-train-val --dataset-path datasets/UAV_yolo_obb --restore
```

**Output Example:**

```
Splitting dataset with 90.0% train / 10.0% val
Dataset path: datasets/UAV_yolo_obb
Random seed: 42

Grouping images by sequence...
Found 10 sequences:
  M0101: 1500 images
  M0201: 1200 images
  M0301: 800 images
  ...

Splitting sequences:
  M0101: 1350 train / 150 val (150 moved)
  M0201: 1080 train / 120 val (120 moved)
  M0301: 720 train / 80 val (80 moved)
  ...

============================================================
SPLIT SUMMARY
============================================================
Total sequences: 10
Total images: 10000
Train images: 9000 (90.0%)
Val images: 1000 (10.0%)
Moved images: 1000
Moved labels: 1000
============================================================
```

## Running with uv

You can run these scripts directly with uv without activating the virtual environment:

```bash
# From repository root
uv run split-train-val --dataset-path datasets/UAV_yolo_obb
uv run train-yolo-obb --data dataset.yaml --epochs 50
```

## Complete Training Workflow

Here's a typical workflow for training a YOLO OBB model:

```bash
# 1. Split your dataset
split-train-val --dataset-path datasets/UAV_yolo_obb --dry-run  # Preview
split-train-val --dataset-path datasets/UAV_yolo_obb            # Execute

# 2. Train the model
train-yolo-obb --data datasets/UAV_yolo_obb/dataset.yaml \
    --model yolo11s-obb.pt \
    --epochs 50 \
    --batch 30 \
    --imgsz 640

# 3. If needed, adjust and continue training
train-yolo-obb --data datasets/UAV_yolo_obb/dataset.yaml \
    --resume \
    --epochs 100
```

## Notes

- **Always use `--dry-run` first** with `split-train-val` to preview changes
- The splitter preserves sequence diversity by sampling from each video sequence
- Label files should be in YOLO format: `<class_id> <x> <y> <w> <h> <angle>`
- Training results are saved to `runs/obb/train/` by default
- Use `--cache ram` for faster training if you have sufficient memory
