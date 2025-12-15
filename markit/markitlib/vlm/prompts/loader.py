"""
loader - Prompt template loading and management

Provides loading and validation of prompt templates from JSON files.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Path to default prompts bundled with the package
DEFAULT_PROMPTS_PATH = Path(__file__).parent / "default_prompts.json"


class PromptLoader:
    """Load and manage VLM prompt templates.

    Supports loading from default bundled prompts or custom user-provided files.
    """

    def __init__(self, prompts_file: Optional[str] = None):
        """Initialize prompt loader.

        Args:
            prompts_file: Path to custom prompts JSON file. If None, uses default prompts.
        """
        self.prompts_file = Path(prompts_file) if prompts_file else DEFAULT_PROMPTS_PATH
        self.prompts: Dict[str, Any] = {}
        self._load_prompts()

    def _load_prompts(self) -> None:
        """Load prompts from JSON file."""
        if not self.prompts_file.exists():
            raise FileNotFoundError(f"Prompts file not found: {self.prompts_file}")

        try:
            with open(self.prompts_file, "r", encoding="utf-8") as f:
                self.prompts = json.load(f)
            logger.info(f"Loaded prompts from {self.prompts_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in prompts file: {e}")

        # Validate required structure
        if "comprehensive" not in self.prompts:
            raise ValueError("Prompts file must contain 'comprehensive' prompt")

    def get_prompt(self, prompt_name: str = "comprehensive") -> Dict[str, Any]:
        """Get a prompt template by name.

        Args:
            prompt_name: Name of the prompt template

        Returns:
            Prompt template dict with 'system', 'user_template', and optionally 'response_schema'

        Raises:
            KeyError: If prompt name not found
        """
        if prompt_name not in self.prompts:
            raise KeyError(f"Prompt '{prompt_name}' not found. Available: {list(self.prompts.keys())}")
        return self.prompts[prompt_name]

    def get_user_prompt(self, prompt_name: str = "comprehensive") -> str:
        """Get the user prompt template text.

        Args:
            prompt_name: Name of the prompt template

        Returns:
            User prompt template string
        """
        prompt = self.get_prompt(prompt_name)
        return prompt.get("user_template", "")

    def get_system_prompt(self, prompt_name: str = "comprehensive") -> str:
        """Get the system prompt text.

        Args:
            prompt_name: Name of the prompt template

        Returns:
            System prompt string
        """
        prompt = self.get_prompt(prompt_name)
        return prompt.get("system", "")

    def get_response_schema(self, prompt_name: str = "comprehensive") -> Optional[Dict[str, Any]]:
        """Get the expected response schema for validation.

        Args:
            prompt_name: Name of the prompt template

        Returns:
            JSON schema dict or None if not defined
        """
        prompt = self.get_prompt(prompt_name)
        return prompt.get("response_schema")

    @property
    def version(self) -> str:
        """Get the prompts file version."""
        return self.prompts.get("version", "unknown")

    @property
    def available_prompts(self) -> list:
        """Get list of available prompt names."""
        # Exclude metadata keys
        return [k for k in self.prompts.keys() if k not in ("version", "description")]


def load_prompts(prompts_file: Optional[str] = None) -> PromptLoader:
    """Convenience function to load prompts.

    Args:
        prompts_file: Path to custom prompts file, or None for defaults

    Returns:
        Configured PromptLoader instance
    """
    return PromptLoader(prompts_file)
