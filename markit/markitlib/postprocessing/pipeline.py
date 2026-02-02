"""
pipeline - Postprocessing pipeline orchestration

Manages and executes postprocessing passes in sequence on OpenLabel data.
"""

import logging
import re
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

    def set_video_properties(
        self, frame_width: int, frame_height: int, fps: float
    ) -> None:
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

    def _count_objects_by_engine(
        self, openlabel_data: Dict[str, Any]
    ) -> Dict[str, int]:
        """Count objects grouped by source detection engine.

        Inspects the annotator string in each object's first frame to determine
        the source engine (yolo, oflow, aruco, or unknown).

        Args:
            openlabel_data: Complete OpenLabel data structure

        Returns:
            Dict mapping engine name to object count.
        """
        counts = {"yolo": 0, "oflow": 0, "aruco": 0, "unknown": 0}
        objects = openlabel_data.get("openlabel", {}).get("objects", {})
        frames = openlabel_data.get("openlabel", {}).get("frames", {})

        for obj_id in objects:
            engine = "unknown"
            # Find the first frame containing this object
            for frame_data in frames.values():
                frame_objects = frame_data.get("objects", {})
                if obj_id not in frame_objects:
                    continue
                try:
                    vec_list = frame_objects[obj_id]["object_data"]["vec"]
                    for vec_item in vec_list:
                        if vec_item.get("name") == "annotator":
                            for ann in reversed(vec_item.get("val", [])):
                                if "markit_housekeeping" in ann:
                                    match = re.match(
                                        r"markit_housekeeping\(([^)]*)\)", ann
                                    )
                                    if match and match.group(1) == "gap":
                                        continue
                                    continue
                                if "yolo" in ann.lower():
                                    engine = "yolo"
                                elif "oflow" in ann.lower() or "optical_flow" in ann.lower():
                                    engine = "oflow"
                                elif "aruco" in ann.lower():
                                    engine = "aruco"
                                break
                            break
                except (KeyError, IndexError):
                    pass
                break
            counts[engine] = counts.get(engine, 0) + 1

        return counts

    @staticmethod
    def _format_inventory(counts: Dict[str, int]) -> str:
        """Format an engine count dict as a concise log string."""
        total = sum(counts.values())
        parts = []
        for engine in ("yolo", "oflow", "aruco", "unknown"):
            c = counts.get(engine, 0)
            if c > 0:
                parts.append(f"{c} {engine}")
        return f"{', '.join(parts)} ({total} total)"

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

            before_counts = self._count_objects_by_engine(processed_data)
            logger.info(f"    Before: {self._format_inventory(before_counts)}")

            try:
                if self.frame_width and self.frame_height and self.fps:
                    pass_instance.set_video_properties(
                        self.frame_width, self.frame_height, self.fps
                    )

                if hasattr(self, "ontology_path") and self.ontology_path:
                    pass_instance.set_ontology_path(self.ontology_path)

                processed_data = pass_instance.process(processed_data)
                stats = pass_instance.get_statistics()
                logger.info(f"    Statistics: {stats}")
            except Exception as e:
                logger.error(f"    Error in {pass_name}: {e}")
                raise

            after_counts = self._count_objects_by_engine(processed_data)
            # Log after-counts with delta summary for any changes
            deltas = []
            for engine in ("yolo", "oflow", "aruco", "unknown"):
                diff = after_counts.get(engine, 0) - before_counts.get(engine, 0)
                if diff != 0:
                    deltas.append(f"{'lost' if diff < 0 else 'gained'} {abs(diff)} {engine}")
            after_str = self._format_inventory(after_counts)
            if deltas:
                after_str += f"  [{', '.join(deltas)}]"
            logger.info(f"    After:  {after_str}")

        logger.info("Postprocessing completed")
        return processed_data
