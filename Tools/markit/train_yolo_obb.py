"""
train_yolo_obb.py

YOLO Oriented Bounding Box (OBB) model training tool for UAV object detection.
Trains YOLO models on datasets with oriented bounding boxes.

Usage:
    python train_yolo_obb.py --data path/to/data.yaml [options]

Arguments:
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

Examples:
    # Basic training
    python train_yolo_obb.py --data UAV.yaml
    
    # Custom parameters
    python train_yolo_obb.py --data UAV.yaml --model yolo11m-obb.pt --epochs 100 --batch 16
    
    # Resume training
    python train_yolo_obb.py --data UAV.yaml --resume
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Optional

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics package not found. Please install it:")
    print("pip install ultralytics")
    sys.exit(1)

# Configure logging
logger = logging.getLogger(__name__)

__version__ = '2.0.0'


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
        
        self.validate()
    
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
    
    def __init__(self, config: YOLOTrainingConfig):
        """Initialize YOLO trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.model = None
    
    def load_model(self) -> None:
        """Load YOLO model for training.
        
        Raises:
            TrainingError: If model loading fails
        """
        try:
            logger.info(f"Loading model: {self.config.model_path}")
            self.model = YOLO(self.config.model_path)
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
        print("="*60)
    
    def train(self) -> None:
        """Execute model training.
        
        Raises:
            TrainingError: If training fails
        """
        if self.model is None:
            raise TrainingError("Model not loaded. Call load_model() first.")
        
        try:
            logger.info("Starting training...")
            self.print_training_info()
            
            # Prepare training arguments
            train_args = {
                'data': str(self.config.data_path),
                'epochs': self.config.epochs,
                'imgsz': self.config.imgsz,
                'batch': self.config.batch,
                'device': self.config.device,
                'project': self.config.project,
                'name': self.config.name,
            }
            
            # Add resume if specified
            if self.config.resume:
                train_args['resume'] = True
                logger.info("Resuming training from last checkpoint...")
            
            # Start training
            results = self.model.train(**train_args)
            
            # Print training results summary
            self.print_training_results(results)
            
            logger.info("Training completed successfully!")
            
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
    """Setup logging configuration.
    
    Args:
        verbose: Enable debug level logging
    """
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
  # Basic training with default parameters
  python train_yolo_obb.py --data UAV.yaml
  
  # Custom model and training parameters
  python train_yolo_obb.py --data UAV.yaml --model yolo11m-obb.pt --epochs 100 --batch 16
  
  # Training with specific image size and device
  python train_yolo_obb.py --data UAV.yaml --imgsz 832 --device cuda:0
  
  # Resume previous training
  python train_yolo_obb.py --data UAV.yaml --resume
  
  # Custom project and experiment name
  python train_yolo_obb.py --data UAV.yaml --project my_experiments --name uav_detection
        """
    )
    
    # Required arguments
    parser.add_argument("--data", "-d", required=True,
                       help="Path to dataset configuration YAML file")
    
    # Model arguments
    parser.add_argument("--model", "-m", default="yolo11s-obb.pt",
                       help="Pre-trained model path or name (default: yolo11s-obb.pt)")
    
    # Training parameters
    parser.add_argument("--epochs", "-e", type=int, default=50,
                       help="Number of training epochs (default: 50)")
    parser.add_argument("--imgsz", "-s", type=int, default=640,
                       help="Image size for training (default: 640)")
    parser.add_argument("--batch", "-b", type=int, default=30,
                       help="Batch size (default: 30)")
    parser.add_argument("--device", default=get_default_device(),
                       help=f"Device to use for training (default: {get_default_device()})")
    
    # Output arguments
    parser.add_argument("--project", default="runs/obb",
                       help="Project directory for saving results (default: runs/obb)")
    parser.add_argument("--name", "-n", default="train",
                       help="Experiment name (default: train)")
    
    # Training options
    parser.add_argument("--resume", action="store_true",
                       help="Resume training from last checkpoint")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    return parser.parse_args()


def main():
    """Main function to execute YOLO OBB training."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logging(args.verbose)
        
        logger.info(f"YOLO OBB Training Tool v{__version__}")
        
        # Create configuration
        config = YOLOTrainingConfig(args)
        logger.info(f"Training configuration validated successfully")
        
        # Create trainer and execute training
        trainer = YOLOTrainer(config)
        trainer.load_model()
        trainer.train()
        
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