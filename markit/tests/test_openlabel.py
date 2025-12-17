"""
Unit tests for OpenLabel module - JSON generation and schema validation.
"""

import json
import tempfile
import os
import pytest
import numpy as np

try:
    import jsonschema

    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False

from markit.markitlib.openlabel import OpenLabelHandler
from markit.markitlib import DetectionResult


class TestOpenLabelHandler:
    """Tests for OpenLabelHandler class."""

    def test_initialization(self, schema_path):
        """Test OpenLabel handler initialization."""
        handler = OpenLabelHandler(schema_path)
        assert handler.schema_path == schema_path
        assert isinstance(handler.openlabel_data, dict)
        assert "openlabel" in handler.openlabel_data

    def test_initialization_with_verbose(self, schema_path):
        """Test OpenLabel handler initialization with verbose mode."""
        handler = OpenLabelHandler(schema_path, verbose=True)
        assert handler.verbose is True
        assert isinstance(handler.debug_data, dict)

    def test_initialization_invalid_schema(self):
        """Test initialization fails with invalid schema path."""
        with pytest.raises(Exception):
            OpenLabelHandler("/nonexistent/schema.json")

    def test_add_metadata(self, schema_path, test_video_path):
        """Test adding metadata to OpenLabel structure."""
        handler = OpenLabelHandler(schema_path)
        handler.add_metadata(test_video_path)

        # Check metadata exists
        metadata = handler.openlabel_data.get("openlabel", {}).get("metadata", {})
        assert metadata is not None
        assert "schema_version" in metadata

    def test_add_frame_objects(self, schema_path, sample_detection, sample_class_map):
        """Test adding frame objects to OpenLabel structure."""
        handler = OpenLabelHandler(schema_path)
        handler.add_metadata("test_video.mp4")

        detections = [sample_detection]
        handler.add_frame_objects(0, detections, sample_class_map)

        # Check frame was added
        frames = handler.openlabel_data.get("openlabel", {}).get("frames", {})
        assert "0" in frames

    def test_add_multiple_frames(self, schema_path, sample_detection, sample_class_map):
        """Test adding multiple frames with objects."""
        handler = OpenLabelHandler(schema_path)
        handler.add_metadata("test_video.mp4")

        # Add first frame
        handler.add_frame_objects(0, [sample_detection], sample_class_map)

        # Create second detection with different position
        detection2 = DetectionResult(
            object_id=1,
            class_id=1,
            confidence=0.9,
            oriented_bbox=np.array(
                [[110.0, 110.0], [210.0, 110.0], [210.0, 160.0], [110.0, 160.0]]
            ),
            center=(160.0, 135.0),
            angle=0.1,
            source_engine="yolo",
            width=100.0,
            height=50.0,
        )

        # Add second frame
        handler.add_frame_objects(1, [detection2], sample_class_map)

        # Check both frames exist
        frames = handler.openlabel_data.get("openlabel", {}).get("frames", {})
        assert "0" in frames
        assert "1" in frames

    def test_save_to_file(self, schema_path, sample_detection, sample_class_map):
        """Test saving OpenLabel data to file."""
        handler = OpenLabelHandler(schema_path)
        handler.add_metadata("test_video.mp4")
        handler.add_frame_objects(0, [sample_detection], sample_class_map)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            handler.save_to_file(temp_path)

            # Verify file exists and is valid JSON
            assert os.path.exists(temp_path)

            with open(temp_path, "r") as f:
                loaded_data = json.load(f)

            assert "openlabel" in loaded_data
            assert loaded_data["openlabel"]["frames"]["0"] is not None

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @pytest.mark.skipif(not JSONSCHEMA_AVAILABLE, reason="jsonschema not installed")
    def test_schema_validation(self, schema_path, sample_detection, sample_class_map):
        """Test that generated OpenLabel JSON validates against schema."""
        handler = OpenLabelHandler(schema_path)
        handler.add_metadata("test_video.mp4")
        handler.add_frame_objects(0, [sample_detection], sample_class_map)

        # Load schema
        with open(schema_path, "r") as f:
            schema = json.load(f)

        # Validate generated data against schema
        try:
            jsonschema.validate(handler.openlabel_data, schema)
        except jsonschema.ValidationError as e:
            pytest.fail(
                f"Generated OpenLabel JSON does not validate against schema: {e}"
            )

    def test_multiple_objects_in_frame(self, schema_path, sample_class_map):
        """Test adding multiple objects to a single frame."""
        handler = OpenLabelHandler(schema_path)
        handler.add_metadata("test_video.mp4")

        # Create two detections
        detection1 = DetectionResult(
            object_id=1,
            class_id=1,
            confidence=0.95,
            oriented_bbox=np.array(
                [[100.0, 100.0], [200.0, 100.0], [200.0, 150.0], [100.0, 150.0]]
            ),
            center=(150.0, 125.0),
            angle=0.0,
            source_engine="yolo",
            width=100.0,
            height=50.0,
        )

        detection2 = DetectionResult(
            object_id=2,
            class_id=2,
            confidence=0.88,
            oriented_bbox=np.array(
                [[300.0, 200.0], [400.0, 200.0], [400.0, 250.0], [300.0, 250.0]]
            ),
            center=(350.0, 225.0),
            angle=0.0,
            source_engine="yolo",
            width=100.0,
            height=50.0,
        )

        # Add both detections to frame 0
        handler.add_frame_objects(0, [detection1, detection2], sample_class_map)

        # Check both objects exist in frame 0
        frames = handler.openlabel_data.get("openlabel", {}).get("frames", {})
        frame_0_objects = frames.get("0", {}).get("objects", {})
        assert len(frame_0_objects) == 2

    def test_empty_frame(self, schema_path, sample_class_map):
        """Test adding a frame with no detections."""
        handler = OpenLabelHandler(schema_path)
        handler.add_metadata("test_video.mp4")

        # Add frame with empty detections list
        handler.add_frame_objects(0, [], sample_class_map)

        # Frame should exist but have no objects
        frames = handler.openlabel_data.get("openlabel", {}).get("frames", {})
        if "0" in frames:
            frame_0_objects = frames.get("0", {}).get("objects", {})
            assert len(frame_0_objects) == 0

    def test_numpy_encoder_handles_numpy_types(self, schema_path, sample_class_map):
        """Test that NumpyEncoder properly handles NumPy data types."""
        handler = OpenLabelHandler(schema_path)
        handler.add_metadata("test_video.mp4")

        # Create detection with explicit NumPy types
        detection = DetectionResult(
            object_id=np.int64(1),
            class_id=np.int32(1),
            confidence=np.float32(0.95),
            oriented_bbox=np.array(
                [[100.0, 100.0], [200.0, 100.0], [200.0, 150.0], [100.0, 150.0]],
                dtype=np.float32,
            ),
            center=(np.float64(150.0), np.float64(125.0)),
            angle=np.float64(0.0),
            source_engine="yolo",
            width=np.float32(100.0),
            height=np.float32(50.0),
        )

        handler.add_frame_objects(0, [detection], sample_class_map)

        # Save to temporary file to ensure encoding works
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            handler.save_to_file(temp_path)

            # Should successfully save and load
            with open(temp_path, "r") as f:
                loaded_data = json.load(f)

            assert "openlabel" in loaded_data

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestOpenLabelStructure:
    """Tests for OpenLabel data structure format."""

    def test_openlabel_root_structure(self, schema_path):
        """Test that root structure has required fields."""
        handler = OpenLabelHandler(schema_path)
        handler.add_metadata("test_video.mp4")

        assert "openlabel" in handler.openlabel_data
        openlabel = handler.openlabel_data["openlabel"]
        assert "metadata" in openlabel
        assert "frames" in openlabel or "streams" in openlabel

    def test_metadata_structure(self, schema_path):
        """Test metadata structure."""
        handler = OpenLabelHandler(schema_path)
        handler.add_metadata("test_video.mp4")

        metadata = handler.openlabel_data.get("openlabel", {}).get("metadata", {})
        assert metadata is not None
        # Check for expected metadata fields (schema-dependent)
        assert "schema_version" in metadata or "annotator" in metadata
