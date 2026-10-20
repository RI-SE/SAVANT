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
