"""
base - Base classes for postprocessing passes

Contains abstract base class for all postprocessing operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class PostprocessingPass(ABC):
    """Abstract base class for postprocessing passes."""

    def set_video_properties(
        self, frame_width: int, frame_height: int, fps: float
    ) -> None:
        """Set video properties for passes that need them.

        Args:
            frame_width: Video frame width in pixels
            frame_height: Video frame height in pixels
            fps: Video frames per second
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps

    def set_ontology_path(self, ontology_path: str) -> None:
        """Set ontology file path for passes that need it.

        Args:
            ontology_path: Path to SAVANT ontology TTL file
        """
        self.ontology_path = ontology_path

    @abstractmethod
    def process(self, openlabel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process OpenLabel data and return modified version.

        Args:
            openlabel_data: Complete OpenLabel data structure

        Returns:
            Modified OpenLabel data structure
        """
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about what this pass did.

        Returns:
            Dictionary with processing statistics
        """
        pass
