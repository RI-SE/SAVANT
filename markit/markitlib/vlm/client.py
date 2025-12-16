"""
client - VLM API client implementations

Provides abstract base and concrete implementations for VLM API clients.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import httpx

from .config import VLMConfig, VLMProvider

logger = logging.getLogger(__name__)


class VLMClient(ABC):
    """Abstract base class for VLM clients."""

    @abstractmethod
    def analyze_image(self, image_base64: str, prompt: str) -> str:
        """Send image to VLM and return text response.

        Args:
            image_base64: Base64-encoded image data
            prompt: Text prompt for the VLM

        Returns:
            Text response from the VLM
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if VLM service is available.

        Returns:
            True if service is healthy, False otherwise
        """
        pass


class VLLMClient(VLMClient):
    """vLLM OpenAI-compatible API client.

    vLLM exposes an OpenAI-compatible API that supports vision models.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: Optional[str] = None,
        timeout: int = 120,
    ):
        """Initialize vLLM client.

        Args:
            base_url: Base URL for vLLM API (e.g., http://localhost:8000)
            model_name: Name of the model loaded in vLLM
            api_key: Optional API key (vLLM default is "EMPTY")
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key or "EMPTY"
        self.timeout = timeout

    def analyze_image(self, image_base64: str, prompt: str) -> str:
        """Send image to vLLM for analysis.

        Args:
            image_base64: Base64-encoded JPEG image
            prompt: Analysis prompt

        Returns:
            VLM response text

        Raises:
            httpx.HTTPStatusError: If request fails
            httpx.TimeoutException: If request times out
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.1,  # Low temperature for consistent structured output
        }

        response = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code != 200:
            logger.error(f"VLM request failed: {response.status_code} - {response.text}")
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    def health_check(self) -> bool:
        """Check if vLLM service is available.

        Returns:
            True if service responds to health check
        """
        try:
            # vLLM provides a /health endpoint
            response = httpx.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                return True

            # Fallback: try /v1/models endpoint
            response = httpx.get(
                f"{self.base_url}/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5,
            )
            return response.status_code == 200
        except (httpx.RequestError, httpx.TimeoutException) as e:
            logger.debug(f"Health check failed: {e}")
            return False


def create_vlm_client(config: VLMConfig) -> VLMClient:
    """Factory function to create appropriate VLM client.

    Args:
        config: VLM configuration

    Returns:
        Configured VLM client instance

    Raises:
        ValueError: If provider is not supported
    """
    if config.provider == VLMProvider.VLLM:
        return VLLMClient(
            base_url=config.base_url,
            model_name=config.model_name,
            api_key=config.api_key,
            timeout=config.timeout,
        )
    else:
        raise ValueError(f"Unsupported VLM provider: {config.provider}")
