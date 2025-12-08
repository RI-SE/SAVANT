"""
Pytest configuration and shared fixtures for markit tests.
"""

import sys
from pathlib import Path

import pytest
import numpy as np

from markit.markitlib import DetectionResult


@pytest.fixture
def test_fixtures_dir():
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def test_video_path(test_fixtures_dir):
    """Path to test video file."""
    return str(test_fixtures_dir / "Kraklanda_short.mp4")


@pytest.fixture
def test_model_path(test_fixtures_dir):
    """Path to test YOLO model."""
    return str(test_fixtures_dir / "best.pt")


@pytest.fixture
def schema_path():
    """Path to OpenLabel JSON schema."""
    return str(Path(__file__).parent.parent.parent / "schema" / "savant_openlabel_subset.schema.json")


@pytest.fixture
def ontology_path():
    """Path to SAVANT ontology file."""
    return str(Path(__file__).parent.parent.parent / "ontology" / "savant.ttl")


@pytest.fixture
def sample_class_map():
    """Sample class map for testing."""
    return {
        0: "vehicle",
        1: "car",
        2: "truck",
        3: "bus"
    }


@pytest.fixture
def sample_detection():
    """Create a sample detection result for testing."""
    bbox = np.array([
        [100.0, 100.0],
        [200.0, 100.0],
        [200.0, 150.0],
        [100.0, 150.0]
    ])
    return DetectionResult(
        object_id=1,
        class_id=1,
        confidence=0.95,
        oriented_bbox=bbox,
        center=(150.0, 125.0),
        angle=0.0,
        source_engine="yolo",
        width=100.0,
        height=50.0
    )


@pytest.fixture
def sample_obb_bbox():
    """Create a sample oriented bounding box."""
    return np.array([
        [100.0, 100.0],
        [200.0, 100.0],
        [200.0, 150.0],
        [100.0, 150.0]
    ])


@pytest.fixture
def sample_rotated_bbox():
    """Create a sample rotated bounding box (45 degrees)."""
    center_x, center_y = 150.0, 125.0
    width, height = 100.0, 50.0
    angle = np.pi / 4  # 45 degrees

    # Calculate corner points for rotated rectangle
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    # Half dimensions
    hw = width / 2
    hh = height / 2

    # Calculate corners relative to center, then rotate
    corners = np.array([
        [-hw, -hh],
        [hw, -hh],
        [hw, hh],
        [-hw, hh]
    ])

    # Apply rotation
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])

    rotated_corners = corners @ rotation_matrix.T

    # Translate to actual center
    rotated_corners[:, 0] += center_x
    rotated_corners[:, 1] += center_y

    return rotated_corners.astype(np.float32)
