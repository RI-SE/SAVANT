"""
Unit tests for postprocessing module - pipeline and basic passes.
"""

import pytest

from markit.markitlib.postprocessing import (
    PostprocessingPipeline,
    GapDetectionPass,
    AngleNormalizationPass,
    DuplicateRemovalPass,
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


def _make_frame_object(annotator, x=100, y=100, w=50, h=30, r=0.0, conf=0.9):
    """Helper to create a frame object entry with annotator and bbox."""
    return {
        "object_data": {
            "rbbox": [{"name": "shape", "val": [x, y, w, h, r]}],
            "vec": [
                {"name": "annotator", "val": [annotator]},
                {"name": "confidence", "val": [conf]},
            ],
        }
    }


class TestDuplicateRemovalPass:
    """Tests for DuplicateRemovalPass, including frame transfer on merge."""

    def test_duplicate_removal_initialization(self):
        """Test duplicate removal pass initialization."""
        dup_pass = DuplicateRemovalPass()
        assert dup_pass.objects_deleted == 0
        assert dup_pass.frames_merged == 0

    def test_duplicate_blanket_delete_shared_frames(self):
        """Test that shared frames are deleted from the duplicate, not transferred."""
        # obj_a (yolo): frames 0-4
        # obj_b (oflow): frames 0-4  (complete overlap)
        # Result: obj_b deleted, no frames transferred (all shared)
        frames = {}
        for i in range(5):
            frames[str(i)] = {
                "objects": {
                    "obj_a": _make_frame_object("yolo", x=100 + i),
                    "obj_b": _make_frame_object("oflow", x=101 + i),
                }
            }

        data = {
            "openlabel": {
                "frames": frames,
                "objects": {
                    "obj_a": {"name": "obj_a", "type": "car"},
                    "obj_b": {"name": "obj_b", "type": "car"},
                },
            }
        }

        dup_pass = DuplicateRemovalPass()
        result = dup_pass.process(data)

        result_objects = result["openlabel"]["objects"]
        assert "obj_a" in result_objects
        assert "obj_b" not in result_objects
        assert dup_pass.frames_merged == 0
        assert dup_pass.frames_modified == 5

    def test_exclusive_frames_transferred_to_kept_object(self):
        """Test that frames exclusive to the deleted object are merged into the kept object.

        Simulates the Ekas_both_hk scenario:
        - obj_yolo (yolo): frames 0-5
        - obj_oflow (oflow): frames 3-9  (overlaps 3-5, exclusive 6-9)

        After duplicate removal, obj_yolo should own frames 0-9,
        with frames 6-9 transferred from obj_oflow.
        """
        frames = {}
        # Frames 0-2: only yolo
        for i in range(3):
            frames[str(i)] = {
                "objects": {
                    "obj_yolo": _make_frame_object("yolo", x=100 + i * 5),
                }
            }
        # Frames 3-5: both yolo and oflow (shared, high IoU)
        for i in range(3, 6):
            frames[str(i)] = {
                "objects": {
                    "obj_yolo": _make_frame_object("yolo", x=100 + i * 5),
                    "obj_oflow": _make_frame_object("oflow", x=101 + i * 5),
                }
            }
        # Frames 6-9: only oflow (exclusive to deleted object)
        for i in range(6, 10):
            frames[str(i)] = {
                "objects": {
                    "obj_oflow": _make_frame_object("oflow", x=101 + i * 5),
                }
            }

        data = {
            "openlabel": {
                "frames": frames,
                "objects": {
                    "obj_yolo": {"name": "obj_yolo", "type": "car"},
                    "obj_oflow": {"name": "obj_oflow", "type": "car"},
                },
            }
        }

        dup_pass = DuplicateRemovalPass()
        result = dup_pass.process(data)

        result_objects = result["openlabel"]["objects"]
        result_frames = result["openlabel"]["frames"]

        # oflow object entry should be deleted
        assert "obj_oflow" not in result_objects
        assert "obj_yolo" in result_objects

        # Exclusive frames 6-9 should now belong to obj_yolo
        for i in range(6, 10):
            frame_objs = result_frames[str(i)]["objects"]
            assert "obj_yolo" in frame_objs, f"Frame {i}: obj_yolo missing after merge"
            assert "obj_oflow" not in frame_objs, f"Frame {i}: obj_oflow not removed"

        # Original yolo frames 0-5 should still be there
        for i in range(6):
            assert "obj_yolo" in result_frames[str(i)]["objects"]

        # Verify statistics
        assert dup_pass.frames_merged == 4  # frames 6,7,8,9
        assert dup_pass.frames_modified == 3  # shared frames 3,4,5
        assert dup_pass.objects_deleted == 1

    def test_merge_statistics_in_get_statistics(self):
        """Test that frames_merged appears in statistics output."""
        dup_pass = DuplicateRemovalPass()
        stats = dup_pass.get_statistics()
        assert "frames_merged" in stats
        assert stats["frames_merged"] == 0
