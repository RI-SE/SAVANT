# trainit

Training tools for YOLO OBB (Oriented Bounding Box) models.

## Overview

This directory contains tools for training and preparing datasets for YOLO OBB models:

- **train_yolo_obb.py** - Train YOLO OBB models for UAV object detection
- **split_train_val.py** - Split YOLO datasets into train/val by sequence

## Contents

- [Installation](#installation)
- [Scripts](#scripts)
  - [train_yolo_obb.py](#train_yolo_obbpy)
    - [Quick Start](#quick-start)
    - [Core Arguments](#core-arguments)
    - [Advanced Training Parameters](#advanced-training-parameters)
    - [Augmentation Parameters](#augmentation-parameters)
    - [Usage Strategies and Tradeoffs](#usage-strategies-and-tradeoffs)
    - [Examples](#examples)
  - [split_train_val.py](#split_train_valpy)
- [Running with uv](#running-with-uv)
- [Complete Training Workflow](#complete-training-workflow)
- [Notes](#notes)

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

#### Quick Start

```bash
# Basic training with YOLO defaults
train-yolo-obb --data dataset.yaml

# Custom model and epochs
train-yolo-obb --data dataset.yaml --model yolo11m-obb.pt --epochs 100
```

#### Core Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--data` | `-d` | (required) | Dataset configuration YAML file |
| `--model` | `-m` | `yolo11s-obb.pt` | Pre-trained model path or YOLO model name |
| `--epochs` | `-e` | 50 | Number of training epochs |
| `--imgsz` | `-s` | 640 | Image size for training (pixels) |
| `--batch` | `-b` | 30 | Batch size |
| `--device` | | auto | Device: `cuda`, `mps`, or `cpu` |
| `--project` | | `runs/obb` | Project directory for results |
| `--name` | `-n` | `train` | Experiment name |
| `--resume` | | False | Resume training from last checkpoint |
| `--verbose` | `-v` | False | Enable verbose YOLO output |
| `--provenance` | | None | Path to W3C PROV-JSON provenance chain file |

#### Advanced Training Parameters

These parameters are optional. If not specified, YOLO defaults are used.

| Argument | YOLO Default | Description |
|----------|--------------|-------------|
| `--lr0` | 0.01 | Initial learning rate. Lower (0.001) for fine-tuning, higher (0.01-0.02) for training from scratch. |
| `--lrf` | 0.01 | Final learning rate as fraction of lr0. Controls learning rate decay schedule. |
| `--optimizer` | auto | Optimizer: `SGD` (stable, good for large batches), `Adam` (fast convergence), `AdamW` (with weight decay), `NAdam`, `RAdam`, `RMSProp` |
| `--warmup-epochs` | 3.0 | Warmup epochs. Gradually increases LR to prevent early instability. Increase for large datasets. |
| `--warmup-momentum` | 0.8 | Initial momentum during warmup. |
| `--patience` | 50 | Early stopping patience (epochs without improvement). Lower (10-20) for quick iteration. |
| `--save-period` | -1 | Save checkpoint every N epochs. Set to 5-10 for long training runs. |
| `--cache` | false | Image caching: `ram` (fastest, needs memory), `disk` (moderate), `false` (slowest) |
| `--workers` | 8 | Dataloader worker threads. Increase for faster data loading if CPU permits. |
| `--close-mosaic` | 10 | Disable mosaic augmentation in final N epochs for fine-tuning on real images. |
| `--freeze` | None | Freeze first N layers for transfer learning. Use 10-15 for backbone freeze. |
| `--box` | 7.5 | Box loss weight. Increase (10+) if bounding box accuracy is poor. |
| `--cls` | 0.5 | Classification loss weight. Increase if class predictions are poor. |
| `--dfl` | 1.5 | Distribution focal loss weight. Affects bounding box regression. |

#### Augmentation Parameters

Data augmentation improves model generalization. Optional - uses YOLO defaults if not specified.

| Argument | YOLO Default | Description |
|----------|--------------|-------------|
| `--hsv-h` | 0.015 | HSV-Hue augmentation range (0-1). Color variation. |
| `--hsv-s` | 0.7 | HSV-Saturation augmentation range. |
| `--hsv-v` | 0.4 | HSV-Value (brightness) augmentation range. |
| `--degrees` | 0.0 | Rotation augmentation range in degrees. Important for objects at various orientations. |
| `--translate` | 0.1 | Translation augmentation as fraction of image size. |
| `--scale` | 0.5 | Scaling augmentation gain (0.5 = 50% to 150% scale). |
| `--shear` | 0.0 | Shear augmentation in degrees. |
| `--perspective` | 0.0 | Perspective augmentation. Use small values (0.0001) for aerial views. |
| `--fliplr` | 0.5 | Horizontal flip probability. |
| `--mosaic` | 1.0 | Mosaic augmentation probability. Combines 4 images. |
| `--mixup` | 0.0 | Mixup augmentation probability. Blends images for regularization. |

#### Usage Strategies and Tradeoffs

##### Fast Iteration vs Production Training

**Fast iteration** (quick experiments, hyperparameter tuning):
```bash
train-yolo-obb --data dataset.yaml \
    --epochs 20 \
    --patience 10 \
    --batch 16
```
- Fewer epochs for quick feedback
- Lower patience for early stopping
- Smaller batch if GPU limited

**Production training** (best results):
```bash
train-yolo-obb --data dataset.yaml \
    --epochs 100 \
    --cache ram \
    --workers 12 \
    --patience 30 \
    --save-period 10
```
- More epochs for convergence
- Caching for speed
- Checkpointing for safety
- Higher patience to avoid premature stopping

##### Transfer Learning

Transfer learning uses pre-trained weights and is effective when:
- You have a small dataset (<1000 images)
- Your domain is similar to COCO (vehicles, people, objects)

**Freeze backbone** (small datasets, similar domain):
```bash
train-yolo-obb --data dataset.yaml \
    --model yolo11s-obb.pt \
    --freeze 10 \
    --lr0 0.001 \
    --epochs 50
```
- `--freeze 10` freezes the backbone, only training detection head
- Lower learning rate prevents destroying pre-trained features

**Full fine-tuning** (larger datasets, different domain):
```bash
train-yolo-obb --data dataset.yaml \
    --model yolo11s-obb.pt \
    --lr0 0.001 \
    --warmup-epochs 5
```
- No freezing, all layers trained
- Longer warmup for stability

##### Memory Optimization

| GPU Memory | Recommended Batch | Cache |
|------------|-------------------|-------|
| 8GB | 8-16 | disk or false |
| 16GB | 16-32 | ram |
| 24GB+ | 32-64 | ram |

```bash
# Low memory (8GB GPU)
train-yolo-obb --data dataset.yaml --batch 8 --imgsz 512

# High memory (24GB+ GPU)
train-yolo-obb --data dataset.yaml --batch 48 --cache ram --workers 16
```

##### Augmentation Strategies

**Conservative** (small datasets <1000 images):
```bash
train-yolo-obb --data dataset.yaml \
    --mosaic 0.5 \
    --scale 0.3 \
    --fliplr 0.5
```
- Lower augmentation to avoid overfitting to augmented patterns
- Keep some augmentation for generalization

**Aggressive** (large datasets >10000 images):
```bash
train-yolo-obb --data dataset.yaml \
    --mosaic 1.0 \
    --mixup 0.15 \
    --degrees 15 \
    --scale 0.8 \
    --perspective 0.0001
```
- Full mosaic and mixup for regularization
- Rotation for orientation invariance
- Perspective for aerial view variation

**UAV/Aerial-specific**:
```bash
train-yolo-obb --data dataset.yaml \
    --degrees 180 \
    --perspective 0.0002 \
    --scale 0.7 \
    --fliplr 0.5
```
- Full rotation (objects can be at any orientation from above)
- Perspective augmentation for altitude variation
- High scale variation for zoom differences

##### Learning Rate Tuning

**When to adjust learning rate:**
- **Large batch sizes (>32)**: Increase `--lr0` proportionally (e.g., batch 64 → lr0 0.02)
- **Transfer learning**: Decrease `--lr0` to 0.001-0.005
- **Training instability**: Decrease `--lr0` and increase `--warmup-epochs`

**Optimizer selection:**
| Optimizer | Best For |
|-----------|----------|
| SGD | Large datasets, stable training, production |
| Adam | Quick convergence, small datasets |
| AdamW | When you need weight decay regularization |

```bash
# SGD for production
train-yolo-obb --data dataset.yaml --optimizer SGD --lr0 0.01

# Adam for quick experiments
train-yolo-obb --data dataset.yaml --optimizer Adam --lr0 0.001
```

##### Early Stopping and Checkpointing

**For long training runs:**
```bash
train-yolo-obb --data dataset.yaml \
    --epochs 200 \
    --patience 30 \
    --save-period 10
```
- `--patience 30`: Stop if no improvement for 30 epochs
- `--save-period 10`: Save checkpoint every 10 epochs (recovery from crashes)

**For quick iteration:**
```bash
train-yolo-obb --data dataset.yaml \
    --epochs 50 \
    --patience 10
```
- Lower patience for faster stopping on bad configurations

#### Examples

```bash
# 1. Basic training
train-yolo-obb --data UAV.yaml

# 2. Larger model with more epochs
train-yolo-obb --data UAV.yaml --model yolo11m-obb.pt --epochs 100 --batch 16

# 3. Fast training with caching
train-yolo-obb --data UAV.yaml --cache ram --workers 12

# 4. Transfer learning with frozen backbone
train-yolo-obb --data UAV.yaml --freeze 10 --lr0 0.001 --epochs 50

# 5. Custom augmentation for aerial imagery
train-yolo-obb --data UAV.yaml --degrees 180 --perspective 0.0001 --scale 0.7

# 6. Low memory configuration
train-yolo-obb --data UAV.yaml --batch 8 --imgsz 512 --cache disk

# 7. Production training with checkpoints
train-yolo-obb --data UAV.yaml --epochs 150 --patience 30 --save-period 10 --cache ram

# 8. Resume interrupted training
train-yolo-obb --data UAV.yaml --resume

# 9. Custom optimizer and learning rate
train-yolo-obb --data UAV.yaml --optimizer AdamW --lr0 0.001 --warmup-epochs 5

# 10. Training with provenance tracking
train-yolo-obb --data UAV.yaml --epochs 100 --provenance training_chain.json

# 11. Full configuration example
train-yolo-obb --data UAV.yaml \
    --model yolo11m-obb.pt \
    --epochs 100 \
    --batch 24 \
    --imgsz 640 \
    --optimizer SGD \
    --lr0 0.01 \
    --patience 25 \
    --save-period 10 \
    --cache ram \
    --workers 12 \
    --degrees 15 \
    --scale 0.6 \
    --mosaic 0.9 \
    --provenance chain.json
```

See `train-yolo-obb --help` for the complete categorized argument list.

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
