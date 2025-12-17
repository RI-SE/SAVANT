"""
Unit tests for postprocessing module - pipeline and basic passes.
"""

import pytest

from markit.markitlib.postprocessing import (
    PostprocessingPipeline,
    GapDetectionPass,
    AngleNormalizationPass,
    FrameIntervalPass,
)


@pytest.fixture
def sample_openlabel_data():
    """Create sample OpenLabel data structure for testing."""
    return {
        "openlabel": {
            "metadata": {"schema_version": "0.1"},
            "streams": {
                "camera1": {
                    "type": "camera",
                    "stream_properties": {
                        "sync": {"frame_shift": 0, "frame_vf": [0, 100]}
                    },
                }
            },
            "frames": {
                "0": {
                    "objects": {
                        "obj_1": {
                            "object_data": {
                                "rbbox": [
                                    {"name": "shape", "val": [150, 125, 100, 50, 0.0]}
                                ]
                            }
                        }
                    }
                },
                "1": {
                    "objects": {
                        "obj_1": {
                            "object_data": {
                                "rbbox": [
                                    {"name": "shape", "val": [155, 130, 100, 50, 0.0]}
                                ]
                            }
                        }
                    }
                },
            },
            "objects": {"obj_1": {"name": "obj_1", "type": "car"}},
        }
    }


class TestPostprocessingPipeline:
    """Tests for PostprocessingPipeline class."""

    def test_pipeline_initialization(self):
        """Test pipeline can be initialized."""
        pipeline = PostprocessingPipeline()
        assert pipeline.passes == []
        assert pipeline.frame_width is None
        assert pipeline.frame_height is None
        assert pipeline.fps is None

    def test_set_video_properties(self):
        """Test setting video properties."""
        pipeline = PostprocessingPipeline()
        pipeline.set_video_properties(1920, 1080, 30.0)

        assert pipeline.frame_width == 1920
        assert pipeline.frame_height == 1080
        assert pipeline.fps == 30.0

    def test_set_ontology_path(self, ontology_path):
        """Test setting ontology path."""
        pipeline = PostprocessingPipeline()
        pipeline.set_ontology_path(ontology_path)

        assert pipeline.ontology_path == ontology_path

    def test_add_pass(self):
        """Test adding passes to pipeline."""
        pipeline = PostprocessingPipeline()
        gap_pass = GapDetectionPass()

        pipeline.add_pass(gap_pass)

        assert len(pipeline.passes) == 1
        assert pipeline.passes[0] == gap_pass

    def test_add_multiple_passes(self):
        """Test adding multiple passes to pipeline."""
        pipeline = PostprocessingPipeline()
        gap_pass = GapDetectionPass()
        angle_pass = AngleNormalizationPass()

        pipeline.add_pass(gap_pass)
        pipeline.add_pass(angle_pass)

        assert len(pipeline.passes) == 2

    def test_execute_empty_pipeline(self, sample_openlabel_data):
        """Test executing pipeline with no passes."""
        pipeline = PostprocessingPipeline()
        result = pipeline.execute(sample_openlabel_data)

        # Should return unmodified data
        assert result == sample_openlabel_data

    def test_execute_single_pass(self, sample_openlabel_data):
        """Test executing pipeline with single pass."""
        pipeline = PostprocessingPipeline()
        pipeline.set_video_properties(1920, 1080, 30.0)
        pipeline.add_pass(GapDetectionPass())

        result = pipeline.execute(sample_openlabel_data)

        # Should return data (possibly modified)
        assert "openlabel" in result
        assert "frames" in result["openlabel"]

    def test_execute_multiple_passes(self, sample_openlabel_data):
        """Test executing pipeline with multiple passes."""
        pipeline = PostprocessingPipeline()
        pipeline.set_video_properties(1920, 1080, 30.0)
        pipeline.add_pass(GapDetectionPass())
        pipeline.add_pass(FrameIntervalPass())

        result = pipeline.execute(sample_openlabel_data)

        # Should execute all passes and return result
        assert "openlabel" in result


class TestGapDetectionPass:
    """Tests for GapDetectionPass."""

    def test_gap_detection_initialization(self):
        """Test gap detection pass initialization."""
        gap_pass = GapDetectionPass()
        assert gap_pass.gaps_detected == {}
        assert len(gap_pass.objects_with_gaps) == 0

    def test_gap_detection_process(self, sample_openlabel_data):
        """Test gap detection processing."""
        gap_pass = GapDetectionPass()
        result = gap_pass.process(sample_openlabel_data)

        # Should return data structure
        assert "openlabel" in result

    def test_gap_detection_statistics(self, sample_openlabel_data):
        """Test gap detection statistics."""
        gap_pass = GapDetectionPass()
        gap_pass.process(sample_openlabel_data)
        stats = gap_pass.get_statistics()

        assert isinstance(stats, dict)
        assert "total_gaps_detected" in stats
        assert "objects_with_gaps" in stats
        assert "gap_details" in stats


class TestAngleNormalizationPass:
    """Tests for AngleNormalizationPass."""

    def test_angle_normalization_initialization(self):
        """Test angle normalization pass initialization."""
        angle_pass = AngleNormalizationPass()
        assert angle_pass.angles_normalized == 0

    def test_angle_normalization_process(self, sample_openlabel_data):
        """Test angle normalization processing."""
        angle_pass = AngleNormalizationPass()
        result = angle_pass.process(sample_openlabel_data)

        # Should return data structure
        assert "openlabel" in result

    def test_angle_normalization_statistics(self, sample_openlabel_data):
        """Test angle normalization statistics."""
        angle_pass = AngleNormalizationPass()
        angle_pass.process(sample_openlabel_data)
        stats = angle_pass.get_statistics()

        assert isinstance(stats, dict)
        assert "angles_normalized" in stats


class TestFrameIntervalPass:
    """Tests for FrameIntervalPass."""

    def test_frame_interval_initialization(self):
        """Test frame interval pass initialization."""
        interval_pass = FrameIntervalPass()
        assert interval_pass.intervals_added == 0

    def test_frame_interval_process(self, sample_openlabel_data):
        """Test frame interval processing."""
        interval_pass = FrameIntervalPass()
        result = interval_pass.process(sample_openlabel_data)

        # Should return data structure
        assert "openlabel" in result

    def test_frame_interval_statistics(self, sample_openlabel_data):
        """Test frame interval statistics."""
        interval_pass = FrameIntervalPass()
        interval_pass.process(sample_openlabel_data)
        stats = interval_pass.get_statistics()

        assert isinstance(stats, dict)
        assert "intervals_added" in stats
