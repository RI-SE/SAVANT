"""
Integration tests for markit - end-to-end video processing pipeline.
"""

import argparse
import json
import os
import tempfile
import pytest

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False

from markit.markitlib import MarkitConfig
from markit.markitlib.processing import VideoProcessor
from markit.markitlib.openlabel import OpenLabelHandler
from markit.markitlib.postprocessing import (
    PostprocessingPipeline,
    GapDetectionPass,
    AngleNormalizationPass,
)


class TestEndToEndPipeline:
    """End-to-end integration tests for video processing pipeline."""

    @pytest.fixture
    def temp_output_json(self):
        """Create temporary output JSON file path."""
        fd, path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    @pytest.fixture
    def basic_args(self, test_video_path, test_model_path, schema_path, ontology_path, temp_output_json):
        """Create basic arguments for testing."""
        return argparse.Namespace(
            weights=test_model_path,
            input=test_video_path,
            output_json=temp_output_json,
            schema=schema_path,
            output_video=None,
            ontology=ontology_path,
            aruco_csv=None,
            detection_method="yolo",
            motion_threshold=0.5,
            min_object_area=200,
            iou_threshold=0.3,
            verbose_conflicts=False,
            disable_conflict_resolution=False,
            housekeeping=False,
            duplicate_avg_iou=0.7,
            duplicate_min_iou=0.3,
            rotation_threshold=0.1,
            min_movement_pixels=5.0,
            temporal_smoothing=0.3,
            edge_distance=200,
            static_threshold=5,
            static_mark=False,
            verbose=False
        )

    def test_config_initialization(self, basic_args):
        """Test that configuration initializes correctly."""
        config = MarkitConfig(basic_args)

        assert config.weights_path == basic_args.weights
        assert config.video_path == basic_args.input
        assert config.output_json_path == basic_args.output_json
        assert config.use_yolo is True
        assert isinstance(config.class_map, dict)

    def test_video_processor_initialization(self, basic_args):
        """Test video processor initialization."""
        config = MarkitConfig(basic_args)
        processor = VideoProcessor(config)

        assert processor.config == config
        assert len(processor.engines) > 0  # Should have YOLO engine

    def test_video_processor_initialize(self, basic_args):
        """Test video processor can open video file."""
        config = MarkitConfig(basic_args)
        processor = VideoProcessor(config)
        processor.initialize()

        assert processor.cap is not None
        assert processor.frame_width > 0
        assert processor.frame_height > 0
        assert processor.fps > 0

        processor.cleanup()

    def test_video_processor_read_frame(self, basic_args):
        """Test video processor can read frames."""
        config = MarkitConfig(basic_args)
        processor = VideoProcessor(config)
        processor.initialize()

        success, frame = processor.read_frame()

        assert success is True
        assert frame is not None
        assert frame.shape[0] > 0  # Height
        assert frame.shape[1] > 0  # Width
        assert frame.shape[2] == 3  # RGB channels

        processor.cleanup()

    def test_video_processor_process_frame(self, basic_args):
        """Test video processor can process a frame."""
        config = MarkitConfig(basic_args)
        processor = VideoProcessor(config)
        processor.initialize()

        success, frame = processor.read_frame()
        assert success

        # Process frame with detection
        detections = processor.process_frame(frame)

        assert isinstance(detections, list)
        # May or may not have detections depending on frame content

        processor.cleanup()

    def test_openlabel_handler_initialization(self, schema_path):
        """Test OpenLabel handler initialization."""
        handler = OpenLabelHandler(schema_path)

        assert handler.openlabel_data is not None
        assert "openlabel" in handler.openlabel_data

    def test_basic_pipeline_execution(self, basic_args):
        """Test basic pipeline: config -> video processor -> openlabel handler."""
        config = MarkitConfig(basic_args)
        processor = VideoProcessor(config)
        handler = OpenLabelHandler(config.schema_path)

        # Initialize
        processor.initialize()
        handler.add_metadata(config.video_path)

        # Process a few frames
        max_frames = 5
        frame_count = 0

        for _ in range(max_frames):
            success, frame = processor.read_frame()
            if not success:
                break

            detections = processor.process_frame(frame)
            handler.add_frame_objects(frame_count, detections, config.class_map)
            frame_count += 1

        # Cleanup
        processor.cleanup()

        # Save output
        handler.save_to_file(config.output_json_path)

        # Verify output file exists
        assert os.path.exists(config.output_json_path)

        # Verify JSON is valid
        with open(config.output_json_path, 'r') as f:
            output_data = json.load(f)

        assert "openlabel" in output_data
        assert "frames" in output_data["openlabel"]

    def test_pipeline_with_postprocessing(self, basic_args):
        """Test pipeline with postprocessing passes."""
        config = MarkitConfig(basic_args)
        processor = VideoProcessor(config)
        handler = OpenLabelHandler(config.schema_path)

        # Initialize
        processor.initialize()
        handler.add_metadata(config.video_path)

        # Process a few frames
        max_frames = 5
        frame_count = 0

        for _ in range(max_frames):
            success, frame = processor.read_frame()
            if not success:
                break

            detections = processor.process_frame(frame)
            handler.add_frame_objects(frame_count, detections, config.class_map)
            frame_count += 1

        # Create postprocessing pipeline
        postprocessing = PostprocessingPipeline()
        postprocessing.set_video_properties(
            processor.frame_width,
            processor.frame_height,
            processor.fps
        )
        postprocessing.add_pass(GapDetectionPass())
        postprocessing.add_pass(AngleNormalizationPass())

        # Execute postprocessing
        handler.openlabel_data = postprocessing.execute(handler.openlabel_data)

        # Cleanup
        processor.cleanup()

        # Save output
        handler.save_to_file(config.output_json_path)

        # Verify output
        assert os.path.exists(config.output_json_path)

        with open(config.output_json_path, 'r') as f:
            output_data = json.load(f)

        assert "openlabel" in output_data

    @pytest.mark.skipif(not JSONSCHEMA_AVAILABLE, reason="jsonschema not installed")
    def test_pipeline_output_validates_against_schema(self, basic_args, schema_path):
        """Test that pipeline output validates against OpenLabel schema."""
        config = MarkitConfig(basic_args)
        processor = VideoProcessor(config)
        handler = OpenLabelHandler(config.schema_path)

        # Initialize
        processor.initialize()
        handler.add_metadata(config.video_path)

        # Process a few frames
        max_frames = 3
        frame_count = 0

        for _ in range(max_frames):
            success, frame = processor.read_frame()
            if not success:
                break

            detections = processor.process_frame(frame)
            handler.add_frame_objects(frame_count, detections, config.class_map)
            frame_count += 1

        # Cleanup
        processor.cleanup()

        # Load schema
        with open(schema_path, 'r') as f:
            schema = json.load(f)

        # Validate output against schema
        try:
            jsonschema.validate(handler.openlabel_data, schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"Pipeline output does not validate against schema: {e}")

    def test_detection_statistics(self, basic_args):
        """Test that video processor provides detection statistics."""
        config = MarkitConfig(basic_args)
        processor = VideoProcessor(config)

        processor.initialize()

        # Process a few frames
        max_frames = 5
        for _ in range(max_frames):
            success, frame = processor.read_frame()
            if not success:
                break

            processor.process_frame(frame)

        # Get statistics
        stats = processor.get_detection_statistics()

        assert isinstance(stats, dict)
        # Statistics should contain information about detections

        processor.cleanup()

    def test_empty_video_handling(self, basic_args):
        """Test pipeline handles video with no detections gracefully."""
        config = MarkitConfig(basic_args)
        processor = VideoProcessor(config)
        handler = OpenLabelHandler(config.schema_path)

        processor.initialize()
        handler.add_metadata(config.video_path)

        # Process just one frame
        success, frame = processor.read_frame()
        if success:
            detections = processor.process_frame(frame)
            handler.add_frame_objects(0, detections, config.class_map)

        processor.cleanup()
        handler.save_to_file(config.output_json_path)

        # Should still produce valid output
        assert os.path.exists(config.output_json_path)

        with open(config.output_json_path, 'r') as f:
            output_data = json.load(f)

        assert "openlabel" in output_data
