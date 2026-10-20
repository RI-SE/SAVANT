"""Tests for savant_common resource resolution."""

import pytest

from savant_common import resources


def test_missing_weights_raise_instead_of_downloading(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("must not access the network")

    monkeypatch.setattr("urllib.request.urlopen", no_network)
    with pytest.raises(FileNotFoundError, match="--weights"):
        resources.get_weights_path("does_not_exist.pt")


def test_bundled_weights_found_in_repo():
    assert resources.get_weights_path().endswith("markit_yolo.pt")
