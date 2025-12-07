"""Project and training configuration models."""

from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Single training configuration within a project.

    Contains all parameters supported by train_yolo_obb.py.
    Parameters set to None use YOLO's internal defaults.
    """

    name: str = Field(..., description="Configuration name")
    description: str = Field(default="", description="Optional description")

    # Dataset selection (folder names relative to datasets_root)
    selected_datasets: list[str] = Field(
        default_factory=list,
        description="List of dataset folder names to include"
    )

    # Core training parameters
    model: str = Field(
        default="yolo11s-obb.pt",
        description="Pre-trained model path or YOLO model name"
    )
    epochs: int = Field(default=50, ge=1, description="Number of training epochs")
    imgsz: int = Field(
        default=640, ge=32, le=2048,
        description="Image size for training"
    )
    batch: int = Field(default=30, ge=1, description="Batch size")
    device: str = Field(
        default="auto",
        description="Device: 'cuda', 'mps', 'cpu', or 'auto'"
    )
    project: str = Field(
        default="runs/obb",
        description="Project directory for saving results"
    )

    # Advanced training parameters (None = use YOLO defaults)
    lr0: Optional[float] = Field(
        default=None, gt=0,
        description="Initial learning rate"
    )
    lrf: Optional[float] = Field(
        default=None, ge=0, le=1,
        description="Final learning rate as fraction of lr0"
    )
    optimizer: Optional[Literal["SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"]] = Field(
        default=None,
        description="Optimizer choice"
    )
    warmup_epochs: Optional[float] = Field(
        default=None, ge=0,
        description="Number of warmup epochs"
    )
    warmup_momentum: Optional[float] = Field(
        default=None, ge=0, le=1,
        description="Initial momentum during warmup"
    )
    patience: Optional[int] = Field(
        default=None, ge=1,
        description="Early stopping patience"
    )
    save_period: Optional[int] = Field(
        default=None,
        description="Save checkpoint every N epochs (-1 to disable)"
    )
    cache: Optional[Literal["true", "false", "ram", "disk"]] = Field(
        default=None,
        description="Image caching strategy"
    )
    workers: Optional[int] = Field(
        default=None, ge=0,
        description="Number of dataloader workers"
    )
    close_mosaic: Optional[int] = Field(
        default=None, ge=0,
        description="Disable mosaic in final N epochs"
    )
    freeze: Optional[int] = Field(
        default=None, ge=0,
        description="Freeze first N layers"
    )

    # Loss weights
    box: Optional[float] = Field(
        default=None, gt=0,
        description="Box loss gain weight"
    )
    cls: Optional[float] = Field(
        default=None, gt=0,
        description="Classification loss gain"
    )
    dfl: Optional[float] = Field(
        default=None, gt=0,
        description="Distribution focal loss gain"
    )

    # Augmentation parameters
    hsv_h: Optional[float] = Field(
        default=None, ge=0, le=1,
        description="HSV Hue augmentation range"
    )
    hsv_s: Optional[float] = Field(
        default=None, ge=0, le=1,
        description="HSV Saturation augmentation range"
    )
    hsv_v: Optional[float] = Field(
        default=None, ge=0, le=1,
        description="HSV Value augmentation range"
    )
    degrees: Optional[float] = Field(
        default=None, ge=0,
        description="Rotation augmentation in degrees"
    )
    translate: Optional[float] = Field(
        default=None, ge=0, le=1,
        description="Translation augmentation fraction"
    )
    scale: Optional[float] = Field(
        default=None, gt=0,
        description="Scaling augmentation gain"
    )
    shear: Optional[float] = Field(
        default=None, ge=0,
        description="Shear augmentation in degrees"
    )
    perspective: Optional[float] = Field(
        default=None, ge=0,
        description="Perspective augmentation"
    )
    fliplr: Optional[float] = Field(
        default=None, ge=0, le=1,
        description="Horizontal flip probability"
    )
    flipud: Optional[float] = Field(
        default=None, ge=0, le=1,
        description="Vertical flip probability"
    )
    mosaic: Optional[float] = Field(
        default=None, ge=0, le=1,
        description="Mosaic augmentation probability"
    )
    mixup: Optional[float] = Field(
        default=None, ge=0, le=1,
        description="Mixup augmentation probability"
    )

    # Train/Val split configuration (used during generation, not training)
    split_enabled: bool = Field(
        default=False,
        description="Enable train/val re-mapping during generation"
    )
    split_ratio: float = Field(
        default=0.9, ge=0.5, le=0.99,
        description="Ratio of data to keep in train (e.g., 0.9 = 90% train)"
    )
    split_seed: int = Field(
        default=42, ge=0,
        description="Random seed for reproducible splits"
    )
    include_val_in_pool: dict[str, bool] = Field(
        default_factory=dict,
        description="Per-dataset toggle for including val folder in split pool"
    )

    def get_non_default_params(self) -> dict:
        """Return dict of parameters that are not None (non-default).

        Useful for generating CLI arguments or config files.
        Excludes metadata and split fields which are not YOLO training params.
        """
        # Fields to skip (not passed to YOLO training)
        skip_fields = {
            'name', 'description', 'selected_datasets',
            'split_enabled', 'split_ratio', 'split_seed', 'include_val_in_pool'
        }
        result = {}
        for field_name, field_info in self.model_fields.items():
            value = getattr(self, field_name)
            if field_name in skip_fields:
                continue
            if value is not None:
                result[field_name] = value
        return result


class ProjectDefaults(BaseModel):
    """Default training parameters for new configurations in a project.

    Only non-None values override the trainit defaults.
    """

    # Core params (None = use trainit defaults)
    model: Optional[str] = None
    epochs: Optional[int] = Field(default=None, ge=1)
    imgsz: Optional[int] = Field(default=None, ge=32, le=2048)
    batch: Optional[int] = Field(default=None, ge=1)
    device: Optional[str] = None
    project: Optional[str] = None

    # Advanced params
    lr0: Optional[float] = None
    lrf: Optional[float] = None
    optimizer: Optional[str] = None
    warmup_epochs: Optional[float] = None
    warmup_momentum: Optional[float] = None
    patience: Optional[int] = None
    save_period: Optional[int] = None
    cache: Optional[str] = None
    workers: Optional[int] = None
    close_mosaic: Optional[int] = None
    freeze: Optional[int] = None
    box: Optional[float] = None
    cls: Optional[float] = None
    dfl: Optional[float] = None

    # Augmentation params
    hsv_h: Optional[float] = None
    hsv_s: Optional[float] = None
    hsv_v: Optional[float] = None
    degrees: Optional[float] = None
    translate: Optional[float] = None
    scale: Optional[float] = None
    shear: Optional[float] = None
    perspective: Optional[float] = None
    fliplr: Optional[float] = None
    flipud: Optional[float] = None
    mosaic: Optional[float] = None
    mixup: Optional[float] = None

    # Train/Val split defaults
    split_enabled: Optional[bool] = None
    split_ratio: Optional[float] = Field(default=None, ge=0.5, le=0.99)
    split_seed: Optional[int] = Field(default=None, ge=0)

    def apply_to_config(self, config: 'TrainingConfig') -> 'TrainingConfig':
        """Apply these defaults to a config, only overriding trainit defaults."""
        data = config.model_dump()
        for field_name in self.model_fields:
            value = getattr(self, field_name)
            if value is not None:
                data[field_name] = value
        return TrainingConfig(**data)


class Project(BaseModel):
    """Project containing dataset references and training configurations.

    A project defines:
    - A root directory containing training datasets
    - Available datasets discovered in that directory
    - Multiple training configurations that can reference those datasets
    - Optional default parameters for new configurations
    """

    name: str = Field(..., description="Project name")
    version: str = Field(default="1.0", description="Project file version")
    description: str = Field(default="", description="Optional project description")

    # Dataset location
    datasets_root: str = Field(
        ...,
        description="Absolute path to root directory containing datasets"
    )

    # Discovered datasets (populated when scanning datasets_root)
    available_datasets: list[str] = Field(
        default_factory=list,
        description="List of valid dataset folder names in datasets_root"
    )

    # Default parameters for new configurations
    default_config: Optional[ProjectDefaults] = Field(
        default=None,
        description="Default training parameters for new configurations"
    )

    # Training configurations
    configs: list[TrainingConfig] = Field(
        default_factory=list,
        description="List of training configurations"
    )

    # Metadata
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Creation timestamp"
    )
    modified_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Last modification timestamp"
    )

    def update_modified(self) -> None:
        """Update the modified_at timestamp."""
        self.modified_at = datetime.now().isoformat()

    def get_config_by_name(self, name: str) -> Optional[TrainingConfig]:
        """Find a configuration by name."""
        for config in self.configs:
            if config.name == name:
                return config
        return None

    def add_config(self, config: TrainingConfig) -> None:
        """Add a new configuration, updating modified timestamp."""
        self.configs.append(config)
        self.update_modified()

    def remove_config(self, name: str) -> bool:
        """Remove a configuration by name. Returns True if found and removed."""
        for i, config in enumerate(self.configs):
            if config.name == name:
                del self.configs[i]
                self.update_modified()
                return True
        return False
