"""
pipeline - Postprocessing pipeline orchestration

Manages and executes postprocessing passes in sequence on OpenLabel data.
"""

import logging
from typing import Any, Dict

from .base import PostprocessingPass

logger = logging.getLogger(__name__)


class PostprocessingPipeline:
    """Manages and executes postprocessing passes on OpenLabel data."""

    def __init__(self):
        self.passes = []
        self.frame_width = None
        self.frame_height = None
        self.fps = None

    def set_video_properties(self, frame_width: int, frame_height: int, fps: float) -> None:
        """Set video properties for the pipeline.

        Args:
            frame_width: Video frame width in pixels
            frame_height: Video frame height in pixels
            fps: Video frames per second
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps

    def set_ontology_path(self, ontology_path: str) -> None:
        """Set ontology file path for the pipeline.

        Args:
            ontology_path: Path to SAVANT ontology TTL file
        """
        self.ontology_path = ontology_path

    def add_pass(self, postprocessing_pass: PostprocessingPass) -> None:
        """Add a postprocessing pass to the pipeline.

        Args:
            postprocessing_pass: Postprocessing pass instance
        """
        self.passes.append(postprocessing_pass)

    def execute(self, openlabel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all postprocessing passes in sequence.

        Args:
            openlabel_data: Complete OpenLabel data structure

        Returns:
            Processed OpenLabel data structure
        """
        if not self.passes:
            logger.info("No postprocessing passes configured")
            return openlabel_data

        logger.info(f"Running {len(self.passes)} postprocessing pass(es)...")

        processed_data = openlabel_data

        for i, pass_instance in enumerate(self.passes, 1):
            pass_name = pass_instance.__class__.__name__
            logger.info(f"  Pass {i}/{len(self.passes)}: {pass_name}")

            try:
                if self.frame_width and self.frame_height and self.fps:
                    pass_instance.set_video_properties(self.frame_width, self.frame_height, self.fps)

                if hasattr(self, 'ontology_path') and self.ontology_path:
                    pass_instance.set_ontology_path(self.ontology_path)

                processed_data = pass_instance.process(processed_data)
                stats = pass_instance.get_statistics()
                logger.info(f"    Statistics: {stats}")
            except Exception as e:
                logger.error(f"    Error in {pass_name}: {e}")
                raise

        logger.info("Postprocessing completed")
        return processed_data
