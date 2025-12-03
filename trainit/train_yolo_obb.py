#!/usr/bin/env python3
"""
train_yolo_obb.py

YOLO Oriented Bounding Box (OBB) model training tool for UAV object detection.
Trains YOLO models on datasets with oriented bounding boxes.

Usage:
    python train_yolo_obb.py --data path/to/data.yaml [options]

Core Arguments:
    --data, -d         Dataset configuration file (YAML) (required)
    --model, -m        Pre-trained model path (default: yolo11s-obb.pt)
    --epochs, -e       Number of training epochs (default: 50)
    --imgsz, -s        Image size for training (default: 640)
    --batch, -b        Batch size (default: 30)
    --device           Device to use (default: auto-detect)
    --project          Project directory for results (default: runs/obb)
    --name, -n         Experiment name (default: train)
    --resume           Resume training from last checkpoint
    --verbose, -v      Enable verbose logging
    --provenance       Path to provenance chain file for W3C PROV-JSON tracking

Advanced Training Parameters (optional - uses YOLO defaults if not specified):
    --lr0              Initial learning rate (default: 0.01)
    --lrf              Final learning rate as fraction of lr0 (default: 0.01)
    --optimizer        Optimizer: SGD, Adam, AdamW, NAdam, RAdam, RMSProp (default: auto)
    --warmup-epochs    Number of warmup epochs (default: 3.0)
    --warmup-momentum  Warmup initial momentum (default: 0.8)
    --patience         Early stopping patience in epochs (default: 50)
    --save-period      Save checkpoint every N epochs (default: -1, disabled)
    --cache            Cache images: true, false, ram, disk (default: false)
    --workers          Number of dataloader workers (default: 8)
    --close-mosaic     Disable mosaic in final N epochs (default: 10)
    --freeze           Freeze first N layers for transfer learning (default: None)
    --box              Box loss gain weight (default: 7.5 for OBB)
    --cls              Class loss gain weight (default: 0.5)
    --dfl              Distribution focal loss gain (default: 1.5)

Augmentation Parameters (optional - uses YOLO defaults if not specified):
    --hsv-h            HSV-Hue augmentation range (default: 0.015)
    --hsv-s            HSV-Saturation augmentation range (default: 0.7)
    --hsv-v            HSV-Value augmentation range (default: 0.4)
    --degrees          Rotation augmentation in degrees (default: 0.0)
    --translate        Translation augmentation as fraction (default: 0.1)
    --scale            Scaling augmentation gain (default: 0.5)
    --shear            Shear augmentation in degrees (default: 0.0)
    --perspective      Perspective augmentation (default: 0.0)
    --fliplr           Horizontal flip probability (default: 0.5)
    --mosaic           Mosaic augmentation probability (default: 1.0)
    --mixup            Mixup augmentation probability (default: 0.0)

Examples:
    # Basic training with YOLO defaults (see examples/dataset.yaml)
    python train_yolo_obb.py --data dataset.yaml

    # Custom learning rate and optimizer
    python train_yolo_obb.py --data dataset.yaml --lr0 0.001 --optimizer Adam

    # Training with caching and more workers
    python train_yolo_obb.py --data dataset.yaml --cache ram --workers 12

    # Custom augmentation settings
    python train_yolo_obb.py --data dataset.yaml --degrees 10 --scale 0.8 --mosaic 0.8

    # Advanced augmentation with perspective and shear
    python train_yolo_obb.py --data dataset.yaml --perspective 0.0001 --shear 2.0 --degrees 15

    # Custom warmup settings
    python train_yolo_obb.py --data dataset.yaml --warmup-epochs 5.0 --warmup-momentum 0.9

    # Save checkpoint every 5 epochs
    python train_yolo_obb.py --data dataset.yaml --save-period 5

    # Transfer learning - freeze backbone layers
    python train_yolo_obb.py --data dataset.yaml --freeze 10

    # Resume training
    python train_yolo_obb.py --data dataset.yaml --resume
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Optional
import yaml
import hashlib
import uuid as uuid_module

# Configure logging
logger = logging.getLogger(__name__)

# Import version from package
from trainit import __version__


def setup_ultralytics_settings(project_dir: str = "runs/obb") -> Path:
    """Setup local Ultralytics settings to avoid using default ~/.config/Ultralytics.

    Creates or updates a local settings.yaml file in the current working directory,
    preserving UUID if it already exists.

    Args:
        project_dir: Directory for training runs (from --project argument)

    Returns:
        Path to the settings file
    """
    # Get the current working directory
    cwd = Path.cwd().resolve()
    settings_file = cwd / "settings.yaml"

    # Set environment variable to redirect Ultralytics config to local directory
    os.environ["YOLO_CONFIG_DIR"] = str(cwd)

    # Default paths relative to current working directory
    datasets_dir = cwd / "datasets"
    runs_dir = cwd / project_dir
    weights_dir = cwd / "weights"

    # Check if settings file already exists
    existing_uuid = None
    if settings_file.exists():
        try:
            with open(settings_file, 'r') as f:
                existing_settings = yaml.safe_load(f)
                if existing_settings and 'uuid' in existing_settings:
                    existing_uuid = existing_settings['uuid']
                    logger.info(f"Found existing settings file with UUID: {existing_uuid[:8]}...")
        except Exception as e:
            logger.warning(f"Could not read existing settings file: {e}")

    # Generate or reuse UUID
    if existing_uuid:
        settings_uuid = existing_uuid
    else:
        # Generate a new UUID and hash it (same way Ultralytics does)
        new_uuid = uuid_module.uuid4()
        settings_uuid = hashlib.sha256(str(new_uuid).encode()).hexdigest()
        logger.info(f"Generated new UUID: {settings_uuid[:8]}...")

    # Create settings dictionary with required paths
    settings = {
        'settings_version': '0.0.4',
        'datasets_dir': str(datasets_dir),
        'weights_dir': str(weights_dir),
        'runs_dir': str(runs_dir),
        'uuid': settings_uuid,
        'sync': True,
        'api_key': '',
        'openai_api_key': '',
        'clearml': True,
        'comet': True,
        'dvc': True,
        'hub': True,
        'mlflow': True,
        'neptune': True,
        'raytune': True,
        'tensorboard': True,
        'wandb': True
    }

    # Write settings to local file
    try:
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        with open(settings_file, 'w') as f:
            yaml.dump(settings, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Ultralytics settings written to: {settings_file}")
        logger.info(f"  Datasets dir: {datasets_dir}")
        logger.info(f"  Weights dir: {weights_dir}")
        logger.info(f"  Runs dir: {runs_dir}")
    except Exception as e:
        logger.error(f"Failed to write settings file: {e}")
        raise

    return settings_file


class ValidationError(Exception):
    """Raised when training parameter validation fails."""
    pass


class TrainingError(Exception):
    """Raised when model training fails."""
    pass


class YOLOTrainingConfig:
    """Configuration class for YOLO OBB training."""

    def __init__(self, args: argparse.Namespace):
        """Initialize configuration from command line arguments.

        Args:
            args: Parsed command line arguments
        """
        # Required/core parameters
        self.data_path = Path(args.data)
        self.model_path = args.model
        self.epochs = args.epochs
        self.imgsz = args.imgsz
        self.batch = args.batch
        self.device = args.device
        self.project = args.project
        self.name = args.name
        self.resume = args.resume
        self.verbose = args.verbose

        # Optional advanced parameters (None = use YOLO defaults)
        # Note: argparse converts hyphens to underscores automatically
        self.lr0 = args.lr0
        self.lrf = args.lrf
        self.optimizer = args.optimizer
        self.warmup_epochs = args.warmup_epochs
        self.warmup_momentum = args.warmup_momentum
        self.patience = args.patience
        self.save_period = args.save_period
        self.cache = self._parse_cache(args.cache)
        self.workers = args.workers
        self.close_mosaic = args.close_mosaic
        self.freeze = args.freeze
        self.box = args.box
        self.cls = args.cls
        self.dfl = args.dfl
        self.hsv_h = args.hsv_h
        self.hsv_s = args.hsv_s
        self.hsv_v = args.hsv_v
        self.degrees = args.degrees
        self.translate = args.translate
        self.scale = args.scale
        self.shear = args.shear
        self.perspective = args.perspective
        self.fliplr = args.fliplr
        self.mosaic = args.mosaic
        self.mixup = args.mixup

        self.validate()

    def _parse_cache(self, cache_value: Optional[str]) -> Optional[bool]:
        """Parse cache argument to boolean or string.

        Args:
            cache_value: Cache argument value

        Returns:
            Parsed cache value (True, False, 'ram', 'disk', or None)
        """
        if cache_value is None:
            return None
        if cache_value.lower() == 'true':
            return True
        elif cache_value.lower() == 'false':
            return False
        else:
            return cache_value  # 'ram' or 'disk'
    
    def validate(self) -> None:
        """Validate training configuration.
        
        Raises:
            ValidationError: If validation fails
        """
        # Check data file exists
        if not self.data_path.exists():
            raise ValidationError(f"Dataset configuration file not found: {self.data_path}")
        
        # Validate data file extension
        if self.data_path.suffix.lower() not in ['.yaml', '.yml']:
            raise ValidationError(f"Dataset configuration must be a YAML file, got: {self.data_path}")
        
        # Validate epochs
        if self.epochs < 1:
            raise ValidationError(f"Epochs must be >= 1, got: {self.epochs}")
        
        # Validate image size
        if self.imgsz < 32 or self.imgsz > 2048:
            raise ValidationError(f"Image size must be between 32 and 2048, got: {self.imgsz}")
        
        # Validate batch size
        if self.batch < 1:
            raise ValidationError(f"Batch size must be >= 1, got: {self.batch}")
        
        # Check if model file exists (if not a YOLO model name)
        if not self.model_path.startswith('yolo'):
            model_path = Path(self.model_path)
            if not model_path.exists():
                logger.warning(f"Model file not found: {model_path}. Will attempt to download if it's a valid YOLO model name.")


class YOLOTrainer:
    """Handles YOLO OBB model training operations."""

    def __init__(self, config: YOLOTrainingConfig, yolo_class):
        """Initialize YOLO trainer.

        Args:
            config: Training configuration
            yolo_class: The YOLO class to use for model loading
        """
        self.config = config
        self.model = None
        self.YOLO = yolo_class

    def load_model(self) -> None:
        """Load YOLO model for training.

        Raises:
            TrainingError: If model loading fails
        """
        try:
            logger.info(f"Loading model: {self.config.model_path}")
            self.model = self.YOLO(self.config.model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            raise TrainingError(f"Failed to load model '{self.config.model_path}': {e}")
    
    def print_training_info(self) -> None:
        """Print training configuration information."""
        print("\n" + "="*60)
        print("YOLO OBB TRAINING CONFIGURATION")
        print("="*60)
        print(f"Dataset: {self.config.data_path}")
        print(f"Model: {self.config.model_path}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Image size: {self.config.imgsz}")
        print(f"Batch size: {self.config.batch}")
        print(f"Device: {self.config.device}")
        print(f"Project: {self.config.project}")
        print(f"Name: {self.config.name}")
        print(f"Resume: {self.config.resume}")

        # Print optional advanced parameters if set
        advanced_params = []
        if self.config.lr0 is not None:
            advanced_params.append(f"Learning rate (lr0): {self.config.lr0}")
        if self.config.lrf is not None:
            advanced_params.append(f"Final LR fraction (lrf): {self.config.lrf}")
        if self.config.optimizer is not None:
            advanced_params.append(f"Optimizer: {self.config.optimizer}")
        if self.config.warmup_epochs is not None:
            advanced_params.append(f"Warmup epochs: {self.config.warmup_epochs}")
        if self.config.warmup_momentum is not None:
            advanced_params.append(f"Warmup momentum: {self.config.warmup_momentum}")
        if self.config.patience is not None:
            advanced_params.append(f"Patience: {self.config.patience}")
        if self.config.save_period is not None:
            advanced_params.append(f"Save period: {self.config.save_period}")
        if self.config.cache is not None:
            advanced_params.append(f"Cache: {self.config.cache}")
        if self.config.workers is not None:
            advanced_params.append(f"Workers: {self.config.workers}")
        if self.config.close_mosaic is not None:
            advanced_params.append(f"Close mosaic: {self.config.close_mosaic}")
        if self.config.freeze is not None:
            advanced_params.append(f"Freeze layers: {self.config.freeze}")
        if self.config.box is not None:
            advanced_params.append(f"Box loss: {self.config.box}")
        if self.config.cls is not None:
            advanced_params.append(f"Class loss: {self.config.cls}")
        if self.config.dfl is not None:
            advanced_params.append(f"DFL loss: {self.config.dfl}")

        # Augmentation parameters
        aug_params = []
        if self.config.hsv_h is not None:
            aug_params.append(f"HSV-H: {self.config.hsv_h}")
        if self.config.hsv_s is not None:
            aug_params.append(f"HSV-S: {self.config.hsv_s}")
        if self.config.hsv_v is not None:
            aug_params.append(f"HSV-V: {self.config.hsv_v}")
        if self.config.degrees is not None:
            aug_params.append(f"Degrees: {self.config.degrees}")
        if self.config.translate is not None:
            aug_params.append(f"Translate: {self.config.translate}")
        if self.config.scale is not None:
            aug_params.append(f"Scale: {self.config.scale}")
        if self.config.shear is not None:
            aug_params.append(f"Shear: {self.config.shear}")
        if self.config.perspective is not None:
            aug_params.append(f"Perspective: {self.config.perspective}")
        if self.config.fliplr is not None:
            aug_params.append(f"FlipLR: {self.config.fliplr}")
        if self.config.mosaic is not None:
            aug_params.append(f"Mosaic: {self.config.mosaic}")
        if self.config.mixup is not None:
            aug_params.append(f"Mixup: {self.config.mixup}")

        if advanced_params:
            print("\nAdvanced Parameters:")
            for param in advanced_params:
                print(f"  {param}")

        if aug_params:
            print("\nAugmentation Parameters:")
            for param in aug_params:
                print(f"  {param}")

        print("="*60)
    
    def train(self) -> object:
        """Execute model training.

        Returns:
            Training results object from YOLO (contains save_dir, metrics, etc.)

        Raises:
            TrainingError: If training fails
        """
        if self.model is None:
            raise TrainingError("Model not loaded. Call load_model() first.")
        
        try:
            logger.info("Starting training...")
            self.print_training_info()
            
            # Prepare training arguments (core parameters)
            train_args = {
                'data': str(self.config.data_path),
                'epochs': self.config.epochs,
                'imgsz': self.config.imgsz,
                'batch': self.config.batch,
                'device': self.config.device,
                'project': self.config.project,
                'name': self.config.name,
                'verbose': self.config.verbose,  # Control YOLO's verbose output
            }

            # Add resume if specified
            if self.config.resume:
                train_args['resume'] = True
                logger.info("Resuming training from last checkpoint...")

            # Add optional advanced parameters (only if explicitly set by user)
            optional_params = {
                'lr0': self.config.lr0,
                'lrf': self.config.lrf,
                'optimizer': self.config.optimizer,
                'warmup_epochs': self.config.warmup_epochs,
                'warmup_momentum': self.config.warmup_momentum,
                'patience': self.config.patience,
                'save_period': self.config.save_period,
                'cache': self.config.cache,
                'workers': self.config.workers,
                'close_mosaic': self.config.close_mosaic,
                'freeze': self.config.freeze,
                'box': self.config.box,
                'cls': self.config.cls,
                'dfl': self.config.dfl,
                'hsv_h': self.config.hsv_h,
                'hsv_s': self.config.hsv_s,
                'hsv_v': self.config.hsv_v,
                'degrees': self.config.degrees,
                'translate': self.config.translate,
                'scale': self.config.scale,
                'shear': self.config.shear,
                'perspective': self.config.perspective,
                'fliplr': self.config.fliplr,
                'mosaic': self.config.mosaic,
                'mixup': self.config.mixup,
            }

            # Only include optional parameters that were explicitly set (not None)
            for param_name, param_value in optional_params.items():
                if param_value is not None:
                    train_args[param_name] = param_value
                    logger.debug(f"Using custom {param_name}: {param_value}")

            # Start training
            results = self.model.train(**train_args)
            
            # Print training results summary
            self.print_training_results(results)

            logger.info("Training completed successfully!")

            return results

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            raise
        except Exception as e:
            raise TrainingError(f"Training failed: {e}")
    
    def print_training_results(self, results) -> None:
        """Print training results summary.
        
        Args:
            results: Training results object from YOLO
        """
        try:
            print("\n" + "="*60)
            print("TRAINING RESULTS SUMMARY")
            print("="*60)
            
            if hasattr(results, 'save_dir'):
                print(f"Results saved to: {results.save_dir}")
            
            if hasattr(results, 'best_fitness'):
                print(f"Best fitness: {results.best_fitness:.4f}")
            
            # Try to access metrics if available
            try:
                if hasattr(results, 'results_dict'):
                    metrics = results.results_dict
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"{key}: {value:.4f}")
            except AttributeError:
                pass
            
            print("="*60)
            
        except Exception as e:
            logger.warning(f"Could not print training results summary: {e}")


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration for the script.

    Args:
        verbose: Enable debug level logging
    """
    # Configure script logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )


def get_default_device() -> str:
    """Get default device for training.
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
    except ImportError:
        pass
    return 'cpu'


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Train YOLO OBB models for oriented bounding box object detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with YOLO defaults (see examples/dataset.yaml)
  python train_yolo_obb.py --data dataset.yaml

  # Custom model and training parameters
  python train_yolo_obb.py --data dataset.yaml --model yolo11m-obb.pt --epochs 100 --batch 16

  # Custom learning rate and optimizer
  python train_yolo_obb.py --data dataset.yaml --lr0 0.001 --optimizer Adam

  # Training with caching and more workers for speed
  python train_yolo_obb.py --data dataset.yaml --cache ram --workers 12

  # Adjust loss weights for better detection
  python train_yolo_obb.py --data dataset.yaml --box 10.0 --cls 0.7

  # Custom augmentation for better generalization
  python train_yolo_obb.py --data dataset.yaml --degrees 15 --scale 0.8 --mosaic 0.8 --mixup 0.1

  # Advanced augmentation with perspective and shear
  python train_yolo_obb.py --data dataset.yaml --perspective 0.0001 --shear 2.0 --degrees 15

  # Custom warmup for smoother training start
  python train_yolo_obb.py --data dataset.yaml --warmup-epochs 5.0 --warmup-momentum 0.9

  # Save checkpoint every 5 epochs
  python train_yolo_obb.py --data dataset.yaml --save-period 5

  # Transfer learning - freeze backbone layers
  python train_yolo_obb.py --data dataset.yaml --freeze 10

  # Resume previous training
  python train_yolo_obb.py --data dataset.yaml --resume

  # Custom project and experiment name
  python train_yolo_obb.py --data dataset.yaml --project my_experiments --name uav_detection
        """
    )

    parser.add_argument('--version', action='version',
                        version=f'SAVANT trainit v{__version__}')

    # Core Arguments
    core = parser.add_argument_group('Core Arguments')
    core.add_argument("--data", "-d", required=True,
                      help="Path to dataset configuration YAML file")
    core.add_argument("--model", "-m", default="yolo11s-obb.pt",
                      help="Pre-trained model path or name (default: yolo11s-obb.pt)")
    core.add_argument("--epochs", "-e", type=int, default=50,
                      help="Number of training epochs (default: 50)")
    core.add_argument("--imgsz", "-s", type=int, default=640,
                      help="Image size for training (default: 640)")
    core.add_argument("--batch", "-b", type=int, default=30,
                      help="Batch size (default: 30)")
    core.add_argument("--device", default=get_default_device(),
                      help=f"Device to use for training (default: {get_default_device()})")
    core.add_argument("--project", default="runs/obb",
                      help="Project directory for saving results (default: runs/obb)")
    core.add_argument("--name", "-n", default="train",
                      help="Experiment name (default: train)")
    core.add_argument("--resume", action="store_true",
                      help="Resume training from last checkpoint")
    core.add_argument("--verbose", "-v", action="store_true",
                      help="Enable verbose logging")
    core.add_argument("--provenance",
                      help="Path to provenance chain file for W3C PROV-JSON tracking (created if not exists)")

    # Advanced Training Parameters
    advanced = parser.add_argument_group(
        'Advanced Training Parameters',
        'Optional parameters - uses YOLO defaults if not specified')
    advanced.add_argument("--lr0", type=float, default=None,
                          help="Initial learning rate (YOLO default: 0.01)")
    advanced.add_argument("--lrf", type=float, default=None,
                          help="Final learning rate as fraction of lr0 (YOLO default: 0.01)")
    advanced.add_argument("--optimizer", type=str, default=None,
                          choices=['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp'],
                          help="Optimizer choice (YOLO default: auto)")
    advanced.add_argument("--warmup-epochs", type=float, default=None,
                          help="Number of warmup epochs (YOLO default: 3.0)")
    advanced.add_argument("--warmup-momentum", type=float, default=None,
                          help="Warmup initial momentum (YOLO default: 0.8)")
    advanced.add_argument("--patience", type=int, default=None,
                          help="Early stopping patience in epochs (YOLO default: 50)")
    advanced.add_argument("--save-period", type=int, default=None,
                          help="Save checkpoint every N epochs (YOLO default: -1, disabled)")
    advanced.add_argument("--cache", type=str, default=None,
                          choices=['true', 'false', 'ram', 'disk'],
                          help="Cache images to RAM/disk for faster training (YOLO default: false)")
    advanced.add_argument("--workers", type=int, default=None,
                          help="Number of dataloader worker threads (YOLO default: 8)")
    advanced.add_argument("--close-mosaic", type=int, default=None,
                          help="Disable mosaic augmentation in final N epochs (YOLO default: 10)")
    advanced.add_argument("--freeze", type=int, default=None,
                          help="Freeze first N layers for transfer learning (YOLO default: None)")
    advanced.add_argument("--box", type=float, default=None,
                          help="Box loss gain weight (YOLO default: 7.5 for OBB)")
    advanced.add_argument("--cls", type=float, default=None,
                          help="Class loss gain weight (YOLO default: 0.5)")
    advanced.add_argument("--dfl", type=float, default=None,
                          help="Distribution focal loss gain (YOLO default: 1.5)")

    # Augmentation Parameters
    augmentation = parser.add_argument_group(
        'Augmentation Parameters',
        'Optional - uses YOLO defaults if not specified')
    augmentation.add_argument("--hsv-h", type=float, default=None,
                              help="HSV-Hue augmentation range (YOLO default: 0.015)")
    augmentation.add_argument("--hsv-s", type=float, default=None,
                              help="HSV-Saturation augmentation range (YOLO default: 0.7)")
    augmentation.add_argument("--hsv-v", type=float, default=None,
                              help="HSV-Value augmentation range (YOLO default: 0.4)")
    augmentation.add_argument("--degrees", type=float, default=None,
                              help="Rotation augmentation range in degrees (YOLO default: 0.0)")
    augmentation.add_argument("--translate", type=float, default=None,
                              help="Translation augmentation as fraction (YOLO default: 0.1)")
    augmentation.add_argument("--scale", type=float, default=None,
                              help="Scaling augmentation gain (YOLO default: 0.5)")
    augmentation.add_argument("--shear", type=float, default=None,
                              help="Shear augmentation in degrees (YOLO default: 0.0)")
    augmentation.add_argument("--perspective", type=float, default=None,
                              help="Perspective augmentation (YOLO default: 0.0)")
    augmentation.add_argument("--fliplr", type=float, default=None,
                              help="Horizontal flip probability (YOLO default: 0.5)")
    augmentation.add_argument("--mosaic", type=float, default=None,
                              help="Mosaic augmentation probability (YOLO default: 1.0)")
    augmentation.add_argument("--mixup", type=float, default=None,
                              help="Mixup augmentation probability (YOLO default: 0.0)")

    return parser.parse_args()


def build_arguments_string(args: argparse.Namespace) -> str:
    """Build a string representation of CLI arguments for provenance tracking.

    Args:
        args: Parsed command line arguments

    Returns:
        Space-separated string of CLI arguments used
    """
    parts = [
        f"--data {args.data}",
        f"--model {args.model}",
        f"--epochs {args.epochs}",
        f"--imgsz {args.imgsz}",
        f"--batch {args.batch}",
        f"--device {args.device}",
        f"--project {args.project}",
        f"--name {args.name}",
    ]

    if args.resume:
        parts.append("--resume")
    if args.verbose:
        parts.append("--verbose")

    # Advanced training parameters (only if explicitly set)
    if args.lr0 is not None:
        parts.append(f"--lr0 {args.lr0}")
    if args.lrf is not None:
        parts.append(f"--lrf {args.lrf}")
    if args.optimizer is not None:
        parts.append(f"--optimizer {args.optimizer}")
    if getattr(args, 'warmup_epochs', None) is not None:
        parts.append(f"--warmup-epochs {args.warmup_epochs}")
    if getattr(args, 'warmup_momentum', None) is not None:
        parts.append(f"--warmup-momentum {args.warmup_momentum}")
    if args.patience is not None:
        parts.append(f"--patience {args.patience}")
    if getattr(args, 'save_period', None) is not None:
        parts.append(f"--save-period {args.save_period}")
    if args.cache is not None:
        parts.append(f"--cache {args.cache}")
    if args.workers is not None:
        parts.append(f"--workers {args.workers}")
    if getattr(args, 'close_mosaic', None) is not None:
        parts.append(f"--close-mosaic {args.close_mosaic}")
    if args.freeze is not None:
        parts.append(f"--freeze {args.freeze}")
    if args.box is not None:
        parts.append(f"--box {args.box}")
    if args.cls is not None:
        parts.append(f"--cls {args.cls}")
    if args.dfl is not None:
        parts.append(f"--dfl {args.dfl}")

    # Augmentation parameters (only if explicitly set)
    if getattr(args, 'hsv_h', None) is not None:
        parts.append(f"--hsv-h {args.hsv_h}")
    if getattr(args, 'hsv_s', None) is not None:
        parts.append(f"--hsv-s {args.hsv_s}")
    if getattr(args, 'hsv_v', None) is not None:
        parts.append(f"--hsv-v {args.hsv_v}")
    if args.degrees is not None:
        parts.append(f"--degrees {args.degrees}")
    if args.translate is not None:
        parts.append(f"--translate {args.translate}")
    if args.scale is not None:
        parts.append(f"--scale {args.scale}")
    if args.shear is not None:
        parts.append(f"--shear {args.shear}")
    if args.perspective is not None:
        parts.append(f"--perspective {args.perspective}")
    if args.fliplr is not None:
        parts.append(f"--fliplr {args.fliplr}")
    if args.mosaic is not None:
        parts.append(f"--mosaic {args.mosaic}")
    if args.mixup is not None:
        parts.append(f"--mixup {args.mixup}")

    return " ".join(parts)


def main():
    """Main function to execute YOLO OBB training."""
    try:
        # Parse arguments
        args = parse_arguments()

        # Capture start time for provenance tracking
        start_time = None
        if args.provenance:
            from datetime import datetime, timezone
            start_time = datetime.now(timezone.utc)

        # Setup logging
        setup_logging(args.verbose)

        logger.info(f"SAVANT trainit v{__version__}")

        # Setup local Ultralytics settings before importing YOLO
        # This must be done before any ultralytics import
        setup_ultralytics_settings(project_dir=args.project)

        # Now import ultralytics (after settings are configured)
        try:
            from ultralytics import YOLO
            import ultralytics.utils as ultralytics_utils
        except ImportError:
            logger.error("Error: ultralytics package not found. Please install it:")
            logger.error("pip install ultralytics")
            sys.exit(1)

        # Configure Ultralytics/YOLO logging
        if not args.verbose:
            ultralytics_utils.LOGGER.setLevel(logging.WARNING)
            logger.info("YOLO output suppressed (use --verbose to enable)")
        else:
            ultralytics_utils.LOGGER.setLevel(logging.INFO)
            logger.info("YOLO verbose output enabled")

        # Create configuration
        config = YOLOTrainingConfig(args)
        logger.info("Training configuration validated successfully")

        # Create trainer and execute training
        trainer = YOLOTrainer(config, YOLO)
        trainer.load_model()
        results = trainer.train()

        # Record provenance if enabled
        if args.provenance and results is not None:
            from datetime import datetime, timezone
            from dataprov import ProvenanceChain

            end_time = datetime.now(timezone.utc)

            chain = ProvenanceChain.load_or_create(
                args.provenance,
                entity_id="savant_trainit_output",
                initial_source=str(args.data),
                description="SAVANT trainit YOLO OBB model training"
            )

            # Build arguments string
            arguments = build_arguments_string(args)

            # Inputs: dataset YAML and pre-trained model
            inputs = [str(args.data), args.model]
            input_formats = ["YAML", "PT"]

            # Outputs: best.pt and last.pt from training results
            weights_dir = Path(results.save_dir) / "weights"
            outputs = [str(weights_dir / "best.pt"), str(weights_dir / "last.pt")]
            output_formats = ["PT", "PT"]

            chain.add(
                started_at=start_time.isoformat().replace("+00:00", "Z"),
                ended_at=end_time.isoformat().replace("+00:00", "Z"),
                tool_name="train_yolo_obb",
                tool_version=__version__,
                operation="YOLO OBB model training",
                inputs=inputs,
                input_formats=input_formats,
                outputs=outputs,
                output_formats=output_formats,
                arguments=arguments,
                capture_agent=True,
                agent_type="automated",
                capture_environment=True
            )

            chain.save(args.provenance)
            logger.info(f"Provenance recorded to {args.provenance}")

        logger.info("Training workflow completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except (ValidationError, TrainingError) as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if logger.isEnabledFor(logging.DEBUG):
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()