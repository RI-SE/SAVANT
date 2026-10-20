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


def test_scene_change_sampler_finds_cut(tmp_path):
    import cv2
    import numpy as np

    from markit.markitlib.vlm.sampling import SceneChangeSampler

    path = str(tmp_path / "cut.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for i in range(40):
        color = (255, 0, 0) if i < 25 else (0, 0, 255)
        writer.write(np.full((48, 64, 3), color, np.uint8))
    writer.release()

    frames = {"openlabel": {"frames": {str(i): {} for i in range(0, 40, 2)}}}
    selected = SceneChangeSampler(threshold=0.5).select_frames(path, frames, max_samples=10)

    assert selected[0] == 0 and 26 in selected and selected[-1] == 38
