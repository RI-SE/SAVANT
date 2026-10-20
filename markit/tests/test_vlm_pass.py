"""Tests for VLMAnalysisPass wiring."""

import pytest

from markit.markitlib.vlm import SamplingStrategy, VLMAnalysisPass, VLMConfig
from markit.markitlib.vlm.sampling import FrameSampler


@pytest.mark.parametrize("strategy", list(SamplingStrategy))
def test_every_sampling_strategy_creates_a_sampler(strategy):
    vlm_pass = VLMAnalysisPass(VLMConfig(sampling_strategy=strategy, sample_interval=7))
    sampler = vlm_pass._create_sampler()
    assert isinstance(sampler, FrameSampler)
    if strategy == SamplingStrategy.UNIFORM:
        assert sampler.interval == 7


def test_api_key_falls_back_to_environment(monkeypatch):
    from markit.markitlib.vlm.client import VLM_API_KEY_ENV, VLLMClient

    monkeypatch.setenv(VLM_API_KEY_ENV, "from-env")
    assert VLLMClient("http://localhost:8000", "m").api_key == "from-env"
    assert VLLMClient("http://localhost:8000", "m", api_key="from-cli").api_key == "from-cli"
    monkeypatch.delenv(VLM_API_KEY_ENV)
    assert VLLMClient("http://localhost:8000", "m").api_key == "EMPTY"


@pytest.mark.parametrize("url, warns", [
    ("http://vlm.example.org:8000", True),
    ("https://vlm.example.org", False),
    ("http://localhost:8000", False),
])
def test_warns_when_key_would_be_sent_unencrypted(caplog, url, warns):
    from markit.markitlib.vlm.client import VLLMClient

    with caplog.at_level("WARNING"):
        VLLMClient(url, "m", api_key="secret")
    assert ("unencrypted" in caplog.text) == warns
    assert "secret" not in caplog.text
