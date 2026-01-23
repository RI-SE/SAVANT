"""
vlm - Vision-Language Model integration for scenario tagging

This package provides VLM-based scene analysis for enriching OpenLABEL
annotations with contextual information like weather, road type, and traffic conditions.
"""

from .config import VLMConfig, VLMProvider, SamplingStrategy
from .client import VLMClient, VLLMClient, create_vlm_client
from .vlm_pass import VLMAnalysisPass

__all__ = [
    "VLMConfig",
    "VLMProvider",
    "SamplingStrategy",
    "VLMClient",
    "VLLMClient",
    "create_vlm_client",
    "VLMAnalysisPass",
]
