"""
sampling - Frame sampling strategies for VLM analysis

Provides strategies for selecting which frames to analyze with the VLM.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import cv2

from .config import SamplingStrategy

logger = logging.getLogger(__name__)


class FrameSampler(ABC):
    """Abstract base for frame sampling strategies."""

    @abstractmethod
    def select_frames(
        self, video_path: str, openlabel_data: Dict[str, Any], max_samples: int
    ) -> List[int]:
        """Select frame indices for VLM analysis.

        Args:
            video_path: Path to the video file
            openlabel_data: OpenLABEL data with frame information
            max_samples: Maximum number of frames to select

        Returns:
            List of frame indices to analyze
        """
        pass


class UniformSampler(FrameSampler):
    """Sample frames at uniform intervals."""

    def __init__(self, interval: int = 30):
        """Initialize uniform sampler.

        Args:
            interval: Number of frames between samples
        """
        self.interval = interval

    def select_frames(
        self, video_path: str, openlabel_data: Dict[str, Any], max_samples: int
    ) -> List[int]:
        """Select every Nth frame.

        Args:
            video_path: Path to the video file
            openlabel_data: OpenLABEL data with frame information
            max_samples: Maximum number of frames to select

        Returns:
            List of frame indices at uniform intervals
        """
        frames = openlabel_data.get("openlabel", {}).get("frames", {})
        frame_indices = sorted(int(f) for f in frames.keys())

        if not frame_indices:
            # Fall back to video properties if no frames in OpenLABEL
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                frame_indices = list(range(total_frames))
            else:
                logger.warning(f"Could not open video: {video_path}")
                return []

        if not frame_indices:
            return []

        first_frame = frame_indices[0]
        last_frame = frame_indices[-1]

        # Generate uniform samples
        selected = list(range(first_frame, last_frame + 1, self.interval))

        # Limit to max_samples while keeping uniform distribution
        if len(selected) > max_samples:
            step = len(selected) // max_samples
            selected = selected[::step][:max_samples]

        # Always include first and last frame if not already present
        if selected and selected[0] != first_frame:
            selected = [first_frame] + selected[1:]
        if selected and selected[-1] != last_frame and len(selected) < max_samples:
            selected.append(last_frame)

        return selected[:max_samples]


class SceneChangeSampler(FrameSampler):
    """Sample frames at scene transitions using histogram difference.

    Note: This is a placeholder for future implementation.
    """

    def __init__(self, threshold: float = 0.5):
        """Initialize scene change sampler.

        Args:
            threshold: Histogram correlation threshold for scene change detection
        """
        self.threshold = threshold

    def select_frames(
        self, video_path: str, openlabel_data: Dict[str, Any], max_samples: int
    ) -> List[int]:
        """Detect scene changes and sample representative frames.

        Args:
            video_path: Path to the video file
            openlabel_data: OpenLABEL data with frame information
            max_samples: Maximum number of frames to select

        Returns:
            List of frame indices at scene transitions
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Could not open video: {video_path}")
            return []

        frames = openlabel_data.get("openlabel", {}).get("frames", {})
        frame_indices = sorted(int(f) for f in frames.keys())

        if not frame_indices:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = list(range(total_frames))

        if not frame_indices:
            cap.release()
            return []

        selected = [frame_indices[0]]  # Always include first frame
        prev_hist = None

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Calculate histogram in HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            if prev_hist is not None:
                # Compare histograms using correlation
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                if diff < self.threshold:  # Scene change detected
                    selected.append(frame_idx)
                    logger.debug(f"Scene change detected at frame {frame_idx}")

            prev_hist = hist

            if len(selected) >= max_samples:
                break

        cap.release()

        # Always include last frame if not already selected
        if frame_indices[-1] not in selected and len(selected) < max_samples:
            selected.append(frame_indices[-1])

        return selected[:max_samples]


class KeyframeSampler(FrameSampler):
    """Sample frames based on detection events (high activity).

    Selects frames with the most detected objects, ensuring temporal distribution.
    """

    def select_frames(
        self, video_path: str, openlabel_data: Dict[str, Any], max_samples: int
    ) -> List[int]:
        """Select frames with significant detection events.

        Args:
            video_path: Path to the video file
            openlabel_data: OpenLABEL data with frame information
            max_samples: Maximum number of frames to select

        Returns:
            List of frame indices with high detection activity
        """
        frames = openlabel_data.get("openlabel", {}).get("frames", {})

        if not frames:
            # Fall back to uniform sampling if no detection data
            return UniformSampler(interval=30).select_frames(
                video_path, openlabel_data, max_samples
            )

        # Calculate detection count per frame
        frame_scores = []
        for frame_idx_str, frame_data in frames.items():
            frame_idx = int(frame_idx_str)
            obj_count = len(frame_data.get("objects", {}))
            frame_scores.append((frame_idx, obj_count))

        # Sort by object count (descending) to prioritize active frames
        frame_scores.sort(key=lambda x: x[1], reverse=True)

        # Select frames with most activity, ensuring temporal distribution
        selected = []
        min_gap = len(frames) // max_samples if max_samples > 0 else 1
        min_gap = max(1, min_gap)

        for frame_idx, score in frame_scores:
            if all(abs(frame_idx - s) >= min_gap for s in selected):
                selected.append(frame_idx)
            if len(selected) >= max_samples:
                break

        return sorted(selected)


def create_sampler(strategy: SamplingStrategy, **kwargs) -> FrameSampler:
    """Factory function for frame samplers.

    Args:
        strategy: Sampling strategy to use
        **kwargs: Strategy-specific parameters

    Returns:
        Configured frame sampler instance

    Raises:
        ValueError: If strategy is not supported
    """
    samplers = {
        SamplingStrategy.UNIFORM: UniformSampler,
        SamplingStrategy.SCENE_CHANGE: SceneChangeSampler,
        SamplingStrategy.KEYFRAMES: KeyframeSampler,
    }

    if strategy not in samplers:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    return samplers[strategy](**kwargs)
