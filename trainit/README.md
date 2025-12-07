# trainit

Training tools for YOLO OBB (Oriented Bounding Box) models.

## Overview

This directory contains tools for training and preparing datasets for YOLO OBB models:

- **extract_yolo_training.py** - Extract YOLO OBB training data from video with OpenLabel annotations
- **train_yolo_obb.py** - Train YOLO OBB models for UAV object detection
- **split_train_val.py** - Split YOLO datasets into train/val by sequence

> [!NOTE]
> The simple split_train_val tool will be replaced by a more comprehensive dataset management tool in future releases.


## Contents

- [Installation](#installation)
- [Scripts](#scripts)
  - [extract_yolo_training.py](#extract_yolo_trainingpy)
  - [train_yolo_obb.py](#train_yolo_obbpy)
    - [Quick Start](#quick-start)
    - [Core Arguments](#core-arguments)
    - [Advanced Training Parameters](#advanced-training-parameters)
    - [Augmentation Parameters](#augmentation-parameters)
    - [Usage Strategies and Tradeoffs](#usage-strategies-and-tradeoffs)
    - [Examples](#examples)
    - [In-Depth Parameter Guide](#in-depth-parameter-guide)
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
extract-yolo-training --help
train-yolo-obb --help
split-train-val --help
```

## Scripts

### extract_yolo_training.py

Extracts YOLO OBB training data from video files with OpenLabel annotations. This tool samples frames from a video at regular intervals and creates a complete YOLO OBB dataset with images, labels, and configuration.

**Key Features:**
- Extracts frames at configurable intervals (by seconds or frame count)
- Converts OpenLabel annotations to YOLO OBB format (oriented bounding boxes)
- Uses ontology UIDs directly as YOLO class IDs (consecutive UIDs 0-N only)
- Optional train/val split
- Generates ready-to-use dataset.yaml configuration

**Quick Start:**

```bash
# Basic extraction with default settings (every 5 seconds)
extract-yolo-training -i path/to/video_folder -o path/to/output_dataset

# Extract every 2 seconds with 90/10 train/val split
extract-yolo-training -i input_folder -o output_dataset --interval-seconds 2 --train-ratio 0.9

# Extract every 30 frames
extract-yolo-training -i input_folder -o output_dataset --interval-frames 30
```

**Arguments:**

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input` | `-i` | (required) | Input folder containing video (.mp4) and OpenLabel JSON |
| `--output` | `-o` | (required) | Output folder for YOLO dataset |
| `--ontology` | | `ontology/savant.ttl` | Path to SAVANT ontology (.ttl) file |
| `--video` | | auto-detect | Video filename (if multiple .mp4 in folder) |
| `--openlabel` | | auto-detect | Force specific OpenLabel JSON (skips tagged_file check) |
| `--interval-seconds` | | 5.0 | Extract every N seconds |
| `--interval-frames` | | | Extract every N frames (mutually exclusive with seconds) |
| `--train-ratio` | | None | Train/val split ratio (e.g., 0.9 for 90% train) |
| `--seed` | | 42 | Random seed for train/val split |
| `--verbose` | `-v` | False | Enable verbose logging |
| `--version` | | | Show version and exit |

**Input Requirements:**

The input folder must contain:
- A video file (.mp4)
- An OpenLabel JSON file with `metadata.tagged_file` matching the video filename

**Output Structure:**

```
output_dataset/
├── dataset.yaml          # YOLO configuration file
├── images/
│   ├── train/           # Training images
│   │   ├── frame_000000.jpg
│   │   └── ...
│   └── val/             # Validation images (if --train-ratio used)
└── labels/
    ├── train/           # Training labels (YOLO OBB format)
    │   ├── frame_000000.txt
    │   └── ...
    └── val/             # Validation labels (if --train-ratio used)
```

**Class ID Mapping:**

The tool uses ontology UIDs directly as YOLO class IDs:
- Only consecutive UIDs starting from 0 are included (e.g., 0, 1, 2, ..., 24)
- Non-consecutive UIDs (e.g., 200, 300) are excluded
- All consecutive UIDs appear in the YAML file, even if not present in the dataset
- Objects with non-consecutive UIDs are skipped during extraction

**Example Workflow:**

```bash
# 1. Extract training data from annotated video
extract-yolo-training -i annotated_videos/scene1 -o datasets/scene1 --train-ratio 0.9

# 2. Train the model
train-yolo-obb --data datasets/scene1/dataset.yaml --epochs 50

# 3. Or combine multiple extractions and use split-train-val
extract-yolo-training -i videos/scene1 -o datasets/combined
extract-yolo-training -i videos/scene2 -o datasets/combined  # Append to same dataset
split-train-val --dataset-path datasets/combined
```

---

### train_yolo_obb.py

Trains YOLO OBB (Oriented Bounding Box) models for UAV object detection.

An example dataset configuration file is provided in `examples/dataset.yaml`.

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
| `--flipud` | 0.0 | Vertical flip probability. Useful for aerial nadir views. |
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
- More epochs for convergence (not always needed)
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
train-yolo-obb --data dataset.yaml

# 2. Larger model with more epochs
train-yolo-obb --data dataset.yaml --model yolo11m-obb.pt --epochs 100 --batch 16

# 3. Fast training with caching
train-yolo-obb --data dataset.yaml --cache ram --workers 12

# 4. Transfer learning with frozen backbone
train-yolo-obb --data dataset.yaml --freeze 10 --lr0 0.001 --epochs 50

# 5. Custom augmentation for aerial imagery
train-yolo-obb --data dataset.yaml --degrees 180 --perspective 0.0001 --scale 0.7

# 6. Low memory configuration
train-yolo-obb --data dataset.yaml --batch 8 --imgsz 512 --cache disk

# 7. Production training with checkpoints
train-yolo-obb --data dataset.yaml --epochs 150 --patience 30 --save-period 10 --cache ram

# 8. Resume interrupted training
train-yolo-obb --data dataset.yaml --resume

# 9. Custom optimizer and learning rate
train-yolo-obb --data dataset.yaml --optimizer AdamW --lr0 0.001 --warmup-epochs 5

# 10. Training with provenance tracking
train-yolo-obb --data dataset.yaml --epochs 100 --provenance training_chain.json

# 11. Full configuration example
train-yolo-obb --data dataset.yaml \
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

#### In-Depth Parameter Guide

This section provides detailed explanations of training parameters for users who want to understand the underlying concepts and tune their training for specific use cases.

##### Image Size (imgsz)

The image size parameter controls the resolution at which images are processed during training.

| imgsz | Best For | Trade-offs |
|-------|----------|------------|
| 320-512 | Fast experiments, large objects | May miss small objects |
| 640 | General purpose (default) | Good balance of speed and accuracy |
| 1024-1280 | Small object detection, UAV imagery | Higher memory, slower training |

**Key considerations:**
- Image size must be divisible by 32
- Larger imgsz requires more GPU memory: `memory ∝ batch × imgsz²`
- Model size (n/s/m/l/x) doesn't affect recommended imgsz - choose based on object size in your images
- For UAV imagery with small objects, consider 1024 or higher

##### Batch Size

Batch size determines how many images are processed together in each training step.

| GPU VRAM | imgsz 640 | imgsz 1024 |
|----------|-----------|------------|
| 8GB | 8-16 | 2-4 |
| 16GB | 16-32 | 8-12 |
| 24GB+ | 32-64 | 16-24 |

**Tips:**
- Use `batch=-1` for automatic batch size (targets 60% GPU utilization)
- Larger batches give more stable gradients but may need higher learning rate
- YOLO is batch-size agnostic - smaller GPUs can achieve similar results with smaller batches

##### Learning Rate (lr0, lrf)

Learning rate controls how much the model weights are updated during training.

- **lr0** (default 0.01): Initial learning rate
  - SGD typically uses 0.01
  - Adam/AdamW typically uses 0.001
  - Reduce to 0.001-0.005 for transfer learning
  - Increase proportionally for batch sizes >32

- **lrf** (default 0.01): Final LR as fraction of lr0
  - Training uses cosine annealing from lr0 to lr0×lrf
  - Lower values (0.001) give slower decay for fine-grained learning at end

##### Optimizer

| Optimizer | Best For | Characteristics |
|-----------|----------|-----------------|
| auto | Most cases | Selects based on configuration |
| SGD | Production, large datasets | Stable, well-tested, good generalization |
| Adam | Quick experiments, small datasets | Faster convergence |
| AdamW | When overfitting is a concern | Adam with decoupled weight decay |

##### Warmup Parameters

Warmup gradually increases the learning rate at the start of training to prevent instability.

- **warmup_epochs** (default 3.0): Number of epochs to ramp up learning rate
  - Increase to 5+ for large datasets or if training is unstable
- **warmup_momentum** (default 0.8): Starting momentum, ramps up to 0.937

##### Close Mosaic (close_mosaic)

Disables mosaic augmentation in the final N epochs (default 10).

**Why it matters:** Mosaic combines 4 images, which is great for learning but creates unrealistic scenes. Disabling it in final epochs lets the model fine-tune on realistic single images.

##### Freeze Layers

Freezes the first N backbone layers during training for transfer learning.

- Use `--freeze 10` to freeze backbone, training only detection head
- Best for small datasets (<1000 images) where you want to preserve pre-trained features
- Combine with lower learning rate (0.001)

##### Loss Weights (box, cls, dfl)

These control the relative importance of different loss components:

- **box** (default 7.5): Bounding box localization loss
  - Increase to 10+ if bounding boxes are inaccurate
  - OBB default (7.5) is higher than regular detection
- **cls** (default 0.5): Classification loss
  - Increase if class predictions are poor
- **dfl** (default 1.5): Distribution focal loss for bbox regression
  - Fine-tunes bounding box quality

##### Augmentation Parameters In-Depth

**Rotation (degrees)**
- Range: 0-180 (default 0.0)
- **Aerial imagery:** Use 180 - objects can appear at any orientation from above
- **Ground photography:** Use 10-30 - natural camera tilt only
- **OBB note:** Rotation is especially important since OBB captures object orientation

**Translation (translate)**
- Range: 0.0-1.0 (default 0.1)
- Shifts images by fraction of width/height
- Helps detect partially visible objects at image edges

**Scale**
- Range: ≥0.0 (default 0.5)
- Value of 0.5 means objects appear at 50%-150% of original size
- **UAV imagery:** Use 0.5-0.8 (altitude varies significantly)
- **Ground imagery:** Use 0.3-0.5

**Shear**
- Range: 0-90 degrees (default 0.0)
- Skews image diagonally, simulating perspective effects
- **Ground imagery:** 2-5°
- **Aerial at angle:** 5-15°

**Perspective**
- Range: 0.0-0.001 (default 0.0)
- Applies perspective distortion
- **Aerial with tilt variation:** 0.0001-0.0003
- **Nadir (straight-down):** Keep at 0

**Horizontal Flip (fliplr)**
- Range: 0.0-1.0 (default 0.5)
- Safe for most scenarios
- Disable (0.0) for asymmetric objects like text or directional signs

**Vertical Flip (flipud)**
- Range: 0.0-1.0 (default 0.0)
- **Aerial nadir:** Use 0.5 - objects can appear inverted
- **Ground photography:** Keep 0.0 - sky shouldn't appear below ground
- **Aerial at angle:** Keep 0.0 - horizon orientation matters

**Mosaic**
- Range: 0.0-1.0 (default 1.0)
- Combines 4 images into one during training
- Excellent for small object detection and varied backgrounds
- Increases effective batch size and context diversity

**Mixup**
- Range: 0.0-1.0 (default 0.0)
- Blends two images together with their labels
- Use 0.1-0.3 for large datasets as regularization
- Keep 0.0 for small datasets or when precise boundaries matter

##### Use-Case Recommendations

**Aerial Photo - Nadir (Straight Down)**
```bash
train-yolo-obb --data dataset.yaml \
    --degrees 180 --flipud 0.5 --fliplr 0.5 --scale 0.7 --perspective 0
```
Objects can appear at any orientation; no horizon to preserve.

**Aerial Photo - Oblique (Angled)**
```bash
train-yolo-obb --data dataset.yaml \
    --degrees 15 --flipud 0 --fliplr 0.5 --scale 0.6 --perspective 0.0002 --shear 5
```
Preserve horizon orientation; add perspective for altitude variation.

**Ground-Level Photography**
```bash
train-yolo-obb --data dataset.yaml \
    --degrees 10 --flipud 0 --fliplr 0.5 --scale 0.4 --perspective 0.0001
```
Natural camera tilt only; preserve up/down orientation.

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
uv run extract-yolo-training -i videos/scene1 -o datasets/scene1
uv run split-train-val --dataset-path datasets/UAV_yolo_obb
uv run train-yolo-obb --data dataset.yaml --epochs 50
```

## Complete Training Workflow

Here's a typical workflow for training a YOLO OBB model:

```bash
# 1. Extract training data from annotated videos
extract-yolo-training -i annotated_videos/scene1 -o datasets/my_dataset
extract-yolo-training -i annotated_videos/scene2 -o datasets/my_dataset  # Add more scenes

# 2. Split the dataset (if not using --train-ratio during extraction)
split-train-val --dataset-path datasets/my_dataset --dry-run  # Preview
split-train-val --dataset-path datasets/my_dataset            # Execute

# 3. Train the model
train-yolo-obb --data datasets/my_dataset/dataset.yaml \
    --model yolo11s-obb.pt \
    --epochs 50 \
    --batch 30 \
    --imgsz 640

# 4. If needed, adjust and continue training
train-yolo-obb --data datasets/my_dataset/dataset.yaml \
    --resume \
    --epochs 100
```

**Alternative: Single video with built-in split:**

```bash
# Extract with automatic train/val split
extract-yolo-training -i annotated_video -o datasets/my_dataset --train-ratio 0.9

# Train directly
train-yolo-obb --data datasets/my_dataset/dataset.yaml --epochs 50
```

## Notes

- **Always use `--dry-run` first** with `split-train-val` to preview changes
- The splitter preserves sequence diversity by sampling from each video sequence
- Label files should be in YOLO format: `<class_id> <x> <y> <w> <h> <angle>`
- Training results are saved to `runs/obb/train/` by default
- Use `--cache ram` for faster training if you have sufficient memory
