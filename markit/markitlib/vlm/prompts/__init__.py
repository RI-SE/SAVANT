"""
prompts - VLM prompt templates for scene analysis

Provides loading and management of prompt templates from JSON files.
"""

from .loader import PromptLoader, load_prompts

__all__ = ["PromptLoader", "load_prompts"]
