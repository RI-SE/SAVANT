"""
vlm_pass - VLM Analysis Postprocessing Pass

Provides the main postprocessing pass that analyzes video frames with a VLM
and adds scenario tagging contexts and tags to the OpenLABEL structure.
"""

import base64
import logging
from typing import Any, Dict, List, Optional

import cv2

from ..postprocessing.base import PostprocessingPass
from .config import VLMConfig, SamplingStrategy
from .client import create_vlm_client, VLMClient
from .sampling import create_sampler
from .prompts import load_prompts
from .response_parser import VLMResponseParser, SCENARIO_ONTOLOGY_URI

logger = logging.getLogger(__name__)


class VLMAnalysisPass(PostprocessingPass):
    """Analyze video frames with Vision-Language Model for scenario tagging.

    This pass samples frames from the video, sends them to a VLM for analysis,
    and adds the results as contexts and tags in the OpenLABEL structure.
    """

    def __init__(self, config: VLMConfig):
        """Initialize VLM analysis pass.

        Args:
            config: VLM configuration
        """
        self.config = config
        self.client: Optional[VLMClient] = None
        self.video_path: Optional[str] = None
        self.prompt_loader = None

        # Statistics
        self.frames_analyzed = 0
        self.successful_analyses = 0
        self.failed_analyses = 0
        self.contexts_added = 0
        self.tags_added = 0

    def set_video_path(self, video_path: str) -> None:
        """Set video path for frame extraction.

        Args:
            video_path: Path to the source video file
        """
        self.video_path = video_path

    def process(self, openlabel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sampled frames and add contexts/tags to OpenLabel.

        Args:
            openlabel_data: Complete OpenLabel data structure

        Returns:
            Modified OpenLabel data with VLM analysis results
        """
        if not self.config.enabled:
            logger.info("VLM analysis disabled, skipping")
            return openlabel_data

        if not self.video_path:
            logger.error("VLM analysis: video_path not set")
            return openlabel_data

        # Initialize VLM client
        self.client = create_vlm_client(self.config)

        # Health check
        if not self.client.health_check():
            logger.error(f"VLM service unavailable at {self.config.base_url}")
            logger.info("Skipping VLM analysis - service not available")
            return openlabel_data

        # Load prompts
        self.prompt_loader = load_prompts(self.config.prompts_file)
        logger.info(
            f"VLM analysis starting with {self.config.provider.value} "
            f"model: {self.config.model_name}"
        )

        # Select frames to analyze
        sampler = create_sampler(
            self.config.sampling_strategy,
            interval=self.config.sample_interval
            if self.config.sampling_strategy == SamplingStrategy.UNIFORM
            else 30,
        )

        frame_indices = sampler.select_frames(
            self.video_path, openlabel_data, self.config.max_samples
        )

        if not frame_indices:
            logger.warning("No frames selected for VLM analysis")
            return openlabel_data

        logger.info(f"Selected {len(frame_indices)} frames for VLM analysis")

        # Analyze frames
        analysis_results = self._analyze_frames(frame_indices)

        if not analysis_results:
            logger.warning("No successful VLM analyses, skipping context/tag creation")
            return openlabel_data

        # Add scenario ontology to ontologies section
        self._add_scenario_ontology(openlabel_data)

        # Determine frame intervals for contexts
        frames = openlabel_data.get("openlabel", {}).get("frames", {})
        if frames:
            frame_indices_all = sorted(int(f) for f in frames.keys())
            frame_intervals = [
                {
                    "frame_start": frame_indices_all[0],
                    "frame_end": frame_indices_all[-1],
                }
            ]
        else:
            # Use analyzed frame range
            frame_intervals = [
                {
                    "frame_start": min(frame_indices),
                    "frame_end": max(frame_indices),
                }
            ]

        # Convert to OpenLABEL contexts
        contexts = VLMResponseParser.to_openlabel_contexts(
            analysis_results, frame_intervals
        )

        # Add contexts to OpenLABEL
        if "contexts" not in openlabel_data["openlabel"]:
            openlabel_data["openlabel"]["contexts"] = {}

        openlabel_data["openlabel"]["contexts"].update(contexts)
        self.contexts_added = len(contexts)

        # Convert to OpenLABEL tags
        tags = VLMResponseParser.to_openlabel_tags(
            analysis_results,
            model_name=self.config.model_name,
            frames_analyzed=self.successful_analyses,
        )

        # Add tags to OpenLABEL
        if "tags" not in openlabel_data["openlabel"]:
            openlabel_data["openlabel"]["tags"] = {}

        openlabel_data["openlabel"]["tags"].update(tags)
        self.tags_added = len(tags)

        logger.info(
            f"VLM analysis complete: {self.successful_analyses}/{self.frames_analyzed} "
            f"frames analyzed, {self.contexts_added} contexts and {self.tags_added} tags added"
        )

        return openlabel_data

    def _analyze_frames(self, frame_indices: List[int]) -> List[Dict[str, Any]]:
        """Analyze selected frames with VLM.

        Args:
            frame_indices: List of frame indices to analyze

        Returns:
            List of parsed analysis results
        """
        analysis_results = []

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {self.video_path}")
            return analysis_results

        user_prompt = self.prompt_loader.get_user_prompt("comprehensive")

        for frame_idx in frame_indices:
            self.frames_analyzed += 1

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                logger.warning(f"Failed to read frame {frame_idx}")
                self.failed_analyses += 1
                continue

            # Encode frame as base64 JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
            _, buffer = cv2.imencode(".jpg", frame, encode_params)
            image_base64 = base64.b64encode(buffer).decode("utf-8")

            # Send to VLM
            try:
                logger.debug(f"Analyzing frame {frame_idx}...")
                response = self.client.analyze_image(image_base64, user_prompt)
                parsed = VLMResponseParser.parse_comprehensive_response(response)

                if parsed:
                    parsed["_frame_idx"] = frame_idx
                    analysis_results.append(parsed)
                    self.successful_analyses += 1
                    logger.debug(f"Frame {frame_idx}: Analysis successful")
                else:
                    self.failed_analyses += 1
                    logger.warning(f"Frame {frame_idx}: Failed to parse VLM response")

            except Exception as e:
                self.failed_analyses += 1
                logger.error(f"Frame {frame_idx}: VLM error - {e}")

        cap.release()
        return analysis_results

    def _add_scenario_ontology(self, openlabel_data: Dict[str, Any]) -> None:
        """Add scenario ontology reference to OpenLABEL ontologies section.

        Args:
            openlabel_data: OpenLABEL data to modify
        """
        ontologies = openlabel_data.get("openlabel", {}).get("ontologies", {})

        # Find next available ontology UID
        existing_uids = [int(uid) for uid in ontologies.keys() if uid.isdigit()]
        next_uid = max(existing_uids) + 1 if existing_uids else 1

        # Use UID "1" if available, otherwise use next available
        scenario_uid = "1" if "1" not in ontologies else str(next_uid)

        # Update the constant in response_parser if needed
        # For now, we use "1" as planned

        if "1" not in ontologies:
            openlabel_data["openlabel"]["ontologies"]["1"] = SCENARIO_ONTOLOGY_URI
            logger.debug(f"Added scenario ontology as ontology UID 1")

    def get_statistics(self) -> Dict[str, Any]:
        """Get VLM analysis statistics.

        Returns:
            Dictionary with analysis statistics
        """
        return {
            "frames_analyzed": self.frames_analyzed,
            "successful_analyses": self.successful_analyses,
            "failed_analyses": self.failed_analyses,
            "contexts_added": self.contexts_added,
            "tags_added": self.tags_added,
        }
