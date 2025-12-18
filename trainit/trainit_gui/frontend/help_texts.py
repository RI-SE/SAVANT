"""Centralized help texts for trainit-gui parameters.

This module contains all parameter documentation, group descriptions,
and augmentation presets used throughout the GUI.
"""

from typing import Dict, Any


# Parameter help texts - displayed when clicking info buttons
PARAMETER_HELP: Dict[str, Dict[str, str]] = {
    # Core Parameters
    "model": {
        "title": "Model",
        "text": (
            "Pre-trained YOLO model to use as starting point.\n\n"
            "Available sizes:\n"
            "  - yolo11n-obb (nano) - Fastest, least accurate\n"
            "  - yolo11s-obb (small) - Good balance\n"
            "  - yolo11m-obb (medium) - More accurate\n"
            "  - yolo11l-obb (large) - High accuracy\n"
            "  - yolo11x-obb (extra-large) - Best accuracy, slowest\n\n"
            "Larger models need more GPU memory and train slower."
        ),
        "tooltip": "YOLO model variant (n/s/m/l/x)",
    },
    "epochs": {
        "title": "Epochs",
        "text": (
            "Number of complete passes through the training dataset.\n\n"
            "More epochs = longer training but potentially better results.\n"
            "Typical values: 50-200 depending on dataset size.\n\n"
            "Use 'patience' parameter for early stopping to prevent\n"
            "overfitting and save time."
        ),
        "tooltip": "Range: 1-1000, Default: 50",
    },
    "imgsz": {
        "title": "Image Size",
        "text": (
            "Training image size in pixels (must be divisible by 32).\n\n"
            "Recommendations:\n"
            "  - 320-512: Fast experiments, large objects\n"
            "  - 640: General purpose (default)\n"
            "  - 1024-1280: Small object detection, UAV imagery\n\n"
            "Larger sizes improve accuracy for small objects but\n"
            "require more GPU memory: memory ~ batch x imgsz^2"
        ),
        "tooltip": "Range: 32-2048, Default: 640",
    },
    "batch": {
        "title": "Batch Size",
        "text": (
            "Number of images processed together per training step.\n\n"
            "GPU Memory Recommendations:\n"
            "  - 8GB VRAM: batch 8-16 (imgsz 640)\n"
            "  - 16GB VRAM: batch 16-32 (imgsz 640)\n"
            "  - 24GB+ VRAM: batch 32-64 (imgsz 640)\n\n"
            "Larger batches give more stable gradients but need\n"
            "proportionally higher learning rate.\n\n"
            "Use batch=-1 for automatic sizing (60% GPU utilization)."
        ),
        "tooltip": "Range: 1-256, Default: 30",
    },
    "device": {
        "title": "Device",
        "text": (
            "Hardware device for training.\n\n"
            "Options:\n"
            "  - auto: Auto-detect best available\n"
            "  - cuda: NVIDIA GPU (fastest)\n"
            "  - mps: Apple Silicon GPU\n"
            "  - cpu: CPU only (slowest)"
        ),
        "tooltip": "auto/cuda/mps/cpu",
    },
    "project": {
        "title": "Project Directory",
        "text": (
            "Output directory for training results.\n\n"
            "Contains:\n"
            "  - weights/best.pt: Best model weights\n"
            "  - weights/last.pt: Final model weights\n"
            "  - Training metrics and plots"
        ),
        "tooltip": "Default: runs/obb",
    },
    # Advanced Training Parameters
    "lr0": {
        "title": "Initial Learning Rate",
        "text": (
            "Starting learning rate for training.\n\n"
            "Recommendations:\n"
            "  - SGD optimizer: 0.01 (default)\n"
            "  - Adam/AdamW: 0.001\n"
            "  - Transfer learning: 0.001-0.005\n"
            "  - Large batches (>32): increase proportionally\n\n"
            "Higher values = faster learning but may be unstable.\n"
            "Lower values = slower but more stable convergence."
        ),
        "tooltip": "Range: 0.0001-1.0, Default: 0.01",
    },
    "lrf": {
        "title": "Final Learning Rate",
        "text": (
            "Final learning rate as a fraction of lr0.\n\n"
            "Training uses cosine annealing from lr0 to (lr0 x lrf).\n\n"
            "Example: lr0=0.01, lrf=0.01 means LR decays from\n"
            "0.01 to 0.0001 over the training run.\n\n"
            "Lower values (0.001) give slower decay for\n"
            "fine-grained learning at the end."
        ),
        "tooltip": "Range: 0.0-1.0, Default: 0.01",
    },
    "optimizer": {
        "title": "Optimizer",
        "text": (
            "Optimization algorithm for training.\n\n"
            "Options:\n"
            "  - auto: Automatic selection (recommended)\n"
            "  - SGD: Stable, good for production and large datasets\n"
            "  - Adam: Faster convergence, good for small datasets\n"
            "  - AdamW: Adam with weight decay, prevents overfitting\n"
            "  - NAdam/RAdam: Adam variants with momentum improvements"
        ),
        "tooltip": "Default: auto",
    },
    "warmup_epochs": {
        "title": "Warmup Epochs",
        "text": (
            "Number of epochs to gradually increase learning rate.\n\n"
            "Prevents training instability in early epochs by\n"
            "ramping up LR from near-zero to lr0.\n\n"
            "Increase to 5+ for large datasets or if training\n"
            "is unstable at the start."
        ),
        "tooltip": "Range: 0-10, Default: 3.0",
    },
    "warmup_momentum": {
        "title": "Warmup Momentum",
        "text": (
            "Starting momentum during warmup phase.\n\n"
            "Ramps up from this value to 0.937 (training momentum)\n"
            "during the warmup epochs."
        ),
        "tooltip": "Range: 0-1, Default: 0.8",
    },
    "patience": {
        "title": "Early Stopping Patience",
        "text": (
            "Stop training after N epochs without improvement.\n\n"
            "Prevents overfitting and saves training time.\n\n"
            "Recommendations:\n"
            "  - Quick experiments: 10-20 epochs\n"
            "  - Production training: 30-50 epochs"
        ),
        "tooltip": "Range: 1-500, Default: 50",
    },
    "save_period": {
        "title": "Save Period",
        "text": (
            "Save a checkpoint every N epochs.\n\n"
            "Set to -1 to disable periodic saving (only best/last).\n"
            "Set to 5-10 for long training runs to enable recovery\n"
            "from crashes."
        ),
        "tooltip": "Range: -1 to 100, Default: -1",
    },
    "cache": {
        "title": "Image Caching",
        "text": (
            "Cache images for faster training.\n\n"
            "Options:\n"
            "  - false: No caching (slowest, least memory)\n"
            "  - true/ram: Cache in RAM (fastest, most memory)\n"
            "  - disk: Cache on disk (moderate speed/memory)\n\n"
            "RAM caching can significantly speed up training\n"
            "if you have sufficient memory."
        ),
        "tooltip": "false/true/ram/disk",
    },
    "workers": {
        "title": "Data Loading Workers",
        "text": (
            "Number of CPU threads for loading training data.\n\n"
            "More workers = faster data loading if CPU permits.\n"
            "Set to 0 for debugging (single-threaded)."
        ),
        "tooltip": "Range: 0-32, Default: 8",
    },
    "close_mosaic": {
        "title": "Close Mosaic",
        "text": (
            "Disable mosaic augmentation in final N epochs.\n\n"
            "Mosaic combines 4 images which is great for learning\n"
            "but creates unrealistic scenes. Disabling it in the\n"
            "final epochs lets the model fine-tune on realistic\n"
            "single images.\n\n"
            "Increase if model performs well on augmented data\n"
            "but poorly on real images."
        ),
        "tooltip": "Range: 0-100, Default: 10",
    },
    "freeze": {
        "title": "Freeze Layers",
        "text": (
            "Freeze first N backbone layers during training.\n\n"
            "Used for transfer learning to preserve pre-trained\n"
            "feature extraction, only training the detection head.\n\n"
            "Recommendations:\n"
            "  - Small datasets (<1000 images): freeze 10-15 layers\n"
            "  - Large datasets: no freezing (train all layers)\n\n"
            "Combine with lower learning rate (0.001) when freezing."
        ),
        "tooltip": "Range: 0-100, Default: 0",
    },
    # Loss Weights
    "box": {
        "title": "Box Loss Weight",
        "text": (
            "Weight for bounding box localization loss.\n\n"
            "Controls how much emphasis is placed on accurate\n"
            "bounding box coordinates.\n\n"
            "Increase to 10+ if bounding boxes are inaccurate.\n"
            "OBB default (7.5) is higher than regular detection."
        ),
        "tooltip": "Range: 0.1-20, Default: 7.5",
    },
    "cls": {
        "title": "Classification Loss Weight",
        "text": (
            "Weight for object classification loss.\n\n"
            "Controls emphasis on correct class predictions.\n\n"
            "Increase if class predictions are poor.\n"
            "Decrease if focusing primarily on localization."
        ),
        "tooltip": "Range: 0.1-5, Default: 0.5",
    },
    "dfl": {
        "title": "Distribution Focal Loss Weight",
        "text": (
            "Weight for distribution focal loss (bbox regression).\n\n"
            "Fine-tunes bounding box quality by learning a\n"
            "distribution over possible box coordinates.\n\n"
            "Higher values emphasize precise localization."
        ),
        "tooltip": "Range: 0.1-5, Default: 1.5",
    },
    # Augmentation Parameters
    "hsv_h": {
        "title": "HSV Hue Augmentation",
        "text": (
            "Range of random hue variation (0-1).\n\n"
            "Adds color diversity to training images.\n"
            "Helps model generalize across different lighting."
        ),
        "tooltip": "Range: 0-1, Default: 0.015",
    },
    "hsv_s": {
        "title": "HSV Saturation Augmentation",
        "text": (
            "Range of random saturation variation (0-1).\n\n"
            "Varies color intensity in training images.\n"
            "Helps handle washed-out or oversaturated images."
        ),
        "tooltip": "Range: 0-1, Default: 0.7",
    },
    "hsv_v": {
        "title": "HSV Value Augmentation",
        "text": (
            "Range of random brightness variation (0-1).\n\n"
            "Simulates different lighting conditions.\n"
            "Helps model work in bright and dark environments."
        ),
        "tooltip": "Range: 0-1, Default: 0.4",
    },
    "degrees": {
        "title": "Rotation Augmentation",
        "text": (
            "Maximum rotation angle in degrees (0-180).\n\n"
            "Recommendations:\n"
            "  - Aerial nadir (straight down): 180 degrees\n"
            "    Objects can appear at any orientation.\n"
            "  - Ground photography: 10-30 degrees\n"
            "    Natural camera tilt only.\n\n"
            "Critical for OBB since it captures object orientation."
        ),
        "tooltip": "Range: 0-180, Default: 0.0",
    },
    "translate": {
        "title": "Translation Augmentation",
        "text": (
            "Maximum shift as fraction of image size (0-1).\n\n"
            "Randomly shifts images horizontally and vertically.\n"
            "Helps detect partially visible objects at edges.\n\n"
            "0.1 = up to 10% shift in each direction."
        ),
        "tooltip": "Range: 0-1, Default: 0.1",
    },
    "scale": {
        "title": "Scale Augmentation",
        "text": (
            "Scale variation factor.\n\n"
            "Value of 0.5 means objects appear at 50%-150%\n"
            "of their original size.\n\n"
            "Recommendations:\n"
            "  - UAV imagery: 0.5-0.8 (altitude varies)\n"
            "  - Ground imagery: 0.3-0.5"
        ),
        "tooltip": "Range: 0-2, Default: 0.5",
    },
    "shear": {
        "title": "Shear Augmentation",
        "text": (
            "Shear angle in degrees.\n\n"
            "Skews images diagonally, simulating perspective.\n\n"
            "Recommendations:\n"
            "  - Ground imagery: 2-5 degrees\n"
            "  - Aerial oblique: 5-15 degrees"
        ),
        "tooltip": "Range: 0-90, Default: 0.0",
    },
    "perspective": {
        "title": "Perspective Augmentation",
        "text": (
            "Perspective distortion factor (0-0.001).\n\n"
            "Simulates different viewing angles and depths.\n\n"
            "Recommendations:\n"
            "  - Aerial with tilt variation: 0.0001-0.0003\n"
            "  - Nadir (straight down): 0 (no distortion needed)"
        ),
        "tooltip": "Range: 0-0.01, Default: 0.0",
    },
    "fliplr": {
        "title": "Horizontal Flip",
        "text": (
            "Probability of horizontal flip (left-right).\n\n"
            "Safe for most scenarios. Set to 0.5 by default.\n\n"
            "Disable (0.0) only for asymmetric objects like\n"
            "text or directional signs."
        ),
        "tooltip": "Range: 0-1, Default: 0.5",
    },
    "flipud": {
        "title": "Vertical Flip",
        "text": (
            "Probability of vertical flip (up-down).\n\n"
            "Recommendations:\n"
            "  - Aerial nadir: 0.5 (objects can appear inverted)\n"
            "  - Ground photography: 0.0 (sky shouldn't be below)\n"
            "  - Aerial oblique: 0.0 (horizon orientation matters)"
        ),
        "tooltip": "Range: 0-1, Default: 0.0",
    },
    "mosaic": {
        "title": "Mosaic Augmentation",
        "text": (
            "Probability of mosaic augmentation.\n\n"
            "Combines 4 training images into one, which is\n"
            "excellent for small object detection and provides\n"
            "varied backgrounds.\n\n"
            "Increases effective batch size and context diversity.\n"
            "Disable (0.0) for very specific, consistent scenes."
        ),
        "tooltip": "Range: 0-1, Default: 1.0",
    },
    "mixup": {
        "title": "Mixup Augmentation",
        "text": (
            "Probability of mixup augmentation.\n\n"
            "Blends two images together with their labels.\n"
            "Acts as regularization to prevent overfitting.\n\n"
            "Recommendations:\n"
            "  - Large datasets: 0.1-0.3\n"
            "  - Small datasets: 0.0 (may hurt performance)"
        ),
        "tooltip": "Range: 0-1, Default: 0.0",
    },
    # Train/Val Split Parameters
    "split_enabled": {
        "title": "Enable Train/Val Split",
        "text": (
            "Re-map train/val split during file generation.\n\n"
            "When enabled, source files from selected datasets are\n"
            "pooled together and redistributed into new train/val\n"
            "splits based on the configured ratio.\n\n"
            "Uses stratified sampling by sequence ID when possible\n"
            "(extracts prefix before underscore from filenames).\n"
            "Falls back to random split if sequences aren't detected."
        ),
        "tooltip": "Enable train/val re-splitting",
    },
    "split_ratio": {
        "title": "Train Ratio",
        "text": (
            "Fraction of data to use for training.\n\n"
            "Example: 0.9 means 90% train, 10% validation.\n\n"
            "Recommendations:\n"
            "  - Small datasets (<500 images): 0.8\n"
            "  - Medium datasets: 0.85-0.9\n"
            "  - Large datasets (>5000 images): 0.9-0.95"
        ),
        "tooltip": "Range: 0.5-0.99, Default: 0.9",
    },
    "split_seed": {
        "title": "Random Seed",
        "text": (
            "Seed for reproducible train/val splits.\n\n"
            "Using the same seed produces identical splits,\n"
            "useful for reproducible experiments.\n\n"
            "Change the seed to get a different random split."
        ),
        "tooltip": "Range: 0-999999, Default: 42",
    },
}

# Group descriptions - displayed when clicking group info buttons
GROUP_HELP: Dict[str, Dict[str, str]] = {
    "core": {
        "title": "Core Parameters",
        "text": (
            "Essential training settings that you'll typically adjust.\n\n"
            "These control the basic training setup: model architecture,\n"
            "training duration, image resolution, and hardware usage.\n\n"
            "Start with defaults and adjust based on your dataset size\n"
            "and available GPU memory."
        ),
    },
    "advanced": {
        "title": "Advanced Training Parameters",
        "text": (
            "Fine-tuning options for experienced users.\n\n"
            "These control the optimization process: learning rate,\n"
            "optimizer choice, warmup, early stopping, and caching.\n\n"
            "Usually the defaults work well. Adjust if:\n"
            "  - Training is unstable (reduce lr0, increase warmup)\n"
            "  - Training is too slow (enable caching, add workers)\n"
            "  - Model overfits (use freezing, lower patience)"
        ),
    },
    "loss": {
        "title": "Loss Weights",
        "text": (
            "Control the relative importance of loss components.\n\n"
            "The total loss is: box_loss + cls_loss + dfl_loss\n"
            "Each weight scales its component's contribution.\n\n"
            "Adjust if:\n"
            "  - Boxes are inaccurate: increase 'box'\n"
            "  - Classes are wrong: increase 'cls'\n"
            "  - Boxes are imprecise: increase 'dfl'"
        ),
    },
    "augmentation": {
        "title": "Augmentation Parameters",
        "text": (
            "Data augmentation improves model generalization.\n\n"
            "Augmentation randomly modifies training images to\n"
            "simulate variations the model might encounter.\n\n"
            "Key parameters for aerial/UAV imagery:\n"
            "  - degrees: Set to 180 for nadir views\n"
            "  - flipud: Set to 0.5 for nadir, 0 for oblique\n"
            "  - scale: 0.5-0.8 for altitude variation\n\n"
            "Use the 'Preset' dropdown for recommended settings."
        ),
    },
    "split": {
        "title": "Train/Val Split Defaults",
        "text": (
            "Default values for train/val split parameters.\n\n"
            "These defaults are used when you enable splitting in\n"
            "the 'Generate Training Files' dialog.\n\n"
            "Train Ratio: Fraction of data for training (e.g. 0.9).\n"
            "Seed: Random seed for reproducible splits."
        ),
    },
}

# Augmentation presets for common use cases
AUGMENTATION_PRESETS: Dict[str, Dict[str, Any]] = {
    "Custom": {
        "description": "Custom settings (no changes)",
        "values": {},  # Empty = no changes
    },
    "Aerial - Nadir": {
        "description": "Straight-down aerial view (drone, satellite)",
        "values": {
            "degrees": 180.0,
            "flipud": 0.5,
            "fliplr": 0.5,
            "scale": 0.7,
            "perspective": 0.0,
            "shear": 0.0,
        },
    },
    "Aerial - Oblique": {
        "description": "Angled aerial view with visible horizon",
        "values": {
            "degrees": 180.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "scale": 0.6,
            "perspective": 0.0002,
            "shear": 5.0,
        },
    },
    "Ground-Level": {
        "description": "Standard ground photography",
        "values": {
            "degrees": 10.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "scale": 0.4,
            "perspective": 0.0001,
            "shear": 0.0,
        },
    },
}
