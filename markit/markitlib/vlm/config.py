"""
config - Configuration classes for VLM analysis

Provides dataclasses for configuring VLM provider, sampling strategy, and analysis options.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class VLMProvider(Enum):
    """Supported VLM providers."""

    VLLM = "vllm"


class SamplingStrategy(Enum):
    """Frame sampling strategies for VLM analysis."""

    UNIFORM = "uniform"  # Every Nth frame
    SCENE_CHANGE = "scene_change"  # Detect scene transitions (future)
    KEYFRAMES = "keyframes"  # Frames with high detection activity (future)


@dataclass
class VLMConfig:
    """Configuration for VLM analysis.

    Attributes:
        enabled: Whether VLM analysis is enabled
        provider: VLM provider to use (currently only vLLM)
        model_name: Name of the model on the VLM server
        base_url: Base URL for the VLM API
        api_key: Optional API key for authentication
        timeout: Request timeout in seconds
        sampling_strategy: How to select frames for analysis
        sample_interval: Frame interval for uniform sampling
        max_samples: Maximum number of frames to analyze
        prompts_file: Path to custom prompts JSON file (optional)
    """

    enabled: bool = False
    provider: VLMProvider = VLMProvider.VLLM
    model_name: str = "llama-3.2-11b-vision-instruct"

    # Connection settings
    base_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    timeout: int = 120

    # Sampling settings
    sampling_strategy: SamplingStrategy = SamplingStrategy.UNIFORM
    sample_interval: int = 30
    max_samples: int = 20

    # Prompt configuration
    prompts_file: Optional[str] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.sample_interval < 1:
            raise ValueError("sample_interval must be >= 1")
        if self.max_samples < 1:
            raise ValueError("max_samples must be >= 1")
        if self.timeout < 1:
            raise ValueError("timeout must be >= 1")
