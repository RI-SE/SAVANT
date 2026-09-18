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
    PositionalJitterPass,
    ShortDurationPass,
    StaticObjectRemovalPass,
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

    def test_get_decision_log_aggregates_passes(self):
        """Decision records from every pass are collected with correct source tagging."""
        frames = {
            "0": {"objects": {"obj_short": _make_frame_object("oflow")}},
            "1": {"objects": {"obj_short": _make_frame_object("oflow")}},
            "5": {"objects": {"obj_gap": _make_frame_object("yolo")}},
            "10": {"objects": {"obj_gap": _make_frame_object("yolo")}},
        }
        data = {
            "openlabel": {
                "frames": frames,
                "objects": {
                    "obj_short": {"name": "obj_short", "type": "car"},
                    "obj_gap": {"name": "obj_gap", "type": "car"},
                },
            }
        }

        pipeline = PostprocessingPipeline()
        pipeline.add_pass(GapDetectionPass())
        pipeline.add_pass(ShortDurationPass(min_frames=3, oflow_only=False))
        pipeline.execute(data)

        log = pipeline.get_decision_log()
        sources = {record["source"] for record in log}
        assert "GapDetectionPass" in sources
        assert "ShortDurationPass" in sources
        short_records = [r for r in log if r["source"] == "ShortDurationPass"]
        assert short_records[0]["object_id"] == "obj_short"
        assert short_records[0]["action"] == "remove_object"


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

    def test_gap_detection_decision_log(self, sample_openlabel_data):
        """get_decision_log returns one record per detected gap."""
        frames = {
            "0": {"objects": {"obj_1": _make_frame_object("yolo")}},
            "5": {"objects": {"obj_1": _make_frame_object("yolo")}},
        }
        data = {
            "openlabel": {
                "frames": frames,
                "objects": {"obj_1": {"name": "obj_1", "type": "car"}},
            }
        }
        gap_pass = GapDetectionPass()
        gap_pass.process(data)

        log = gap_pass.get_decision_log()
        assert len(log) == 1
        assert log[0]["action"] == "gap_detected"
        assert log[0]["object_id"] == "obj_1"
        assert log[0]["frame_range"] == {"start": 0, "end": 5}
        assert log[0]["details"]["gap_size"] == 4


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

    def test_iomin_detects_containment_duplicate(self):
        """Test that IoMin catches a large bbox enveloping a smaller one.

        When a large oflow bbox fully contains a smaller yolo bbox, IoU is low
        (suppressed by the large union) but IoMin is high. The IoMin criterion
        should flag them as duplicates.
        """
        # obj_yolo: small bbox (50x30)
        # obj_oflow: large bbox (200x150) at same center — contains yolo entirely
        # IoU ≈ (50*30) / (200*150) ≈ 0.05, but IoMin ≈ 1.0
        frames = {}
        for i in range(5):
            frames[str(i)] = {
                "objects": {
                    "obj_yolo": _make_frame_object("yolo", x=200, y=200, w=50, h=30),
                    "obj_oflow": _make_frame_object("oflow", x=200, y=200, w=200, h=150),
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

        dup_pass = DuplicateRemovalPass(iomin_threshold=0.7)
        result = dup_pass.process(data)

        # oflow should be removed as duplicate (lower priority engine)
        assert "obj_yolo" in result["openlabel"]["objects"]
        assert "obj_oflow" not in result["openlabel"]["objects"]
        assert dup_pass.duplicate_pairs_found == 1

    def test_iomin_no_false_positive_on_partial_overlap(self):
        """Test that IoMin doesn't flag objects with only partial overlap."""
        # Two bboxes side by side with small overlap — IoMin should be below threshold
        frames = {}
        for i in range(5):
            frames[str(i)] = {
                "objects": {
                    "obj_a": _make_frame_object("yolo", x=100, y=100, w=50, h=30),
                    "obj_b": _make_frame_object("oflow", x=140, y=100, w=50, h=30),
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

        dup_pass = DuplicateRemovalPass(iomin_threshold=0.7)
        result = dup_pass.process(data)

        # Both should survive — partial overlap is not containment
        assert "obj_a" in result["openlabel"]["objects"]
        assert "obj_b" in result["openlabel"]["objects"]
        assert dup_pass.duplicate_pairs_found == 0

    def test_merge_statistics_in_get_statistics(self):
        """Test that frames_merged appears in statistics output."""
        dup_pass = DuplicateRemovalPass()
        stats = dup_pass.get_statistics()
        assert "frames_merged" in stats
        assert stats["frames_merged"] == 0

    def test_decision_log_includes_metrics(self):
        """get_decision_log returns a merge_objects record with IoU metrics."""
        frames = {}
        for i in range(5):
            frames[str(i)] = {
                "objects": {
                    "obj_yolo": _make_frame_object("yolo", x=100 + i * 5),
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
        dup_pass.process(data)

        log = dup_pass.get_decision_log()
        assert len(log) == 1
        record = log[0]
        assert record["action"] == "merge_objects"
        assert set(record["object_ids"]) == {"obj_yolo", "obj_oflow"}
        assert record["details"]["deleted_object"] == "obj_oflow"
        assert record["details"]["kept_object"] == "obj_yolo"
        assert "avg_iou" in record["details"]
        assert record["reason"]


def _make_track(xs, y=100, w=50, h=30, annotator="yolo"):
    """Build an OpenLabel data structure with a single object 'obj_1' whose
    bbox center x-coordinate follows the given sequence, one frame apart."""
    frames = {
        str(i): {"objects": {"obj_1": _make_frame_object(annotator, x=x, y=y, w=w, h=h)}}
        for i, x in enumerate(xs)
    }
    return {
        "openlabel": {
            "frames": frames,
            "objects": {"obj_1": {"name": "obj_1", "type": "car"}},
        }
    }


class TestPositionalJitterPass:
    """Tests for PositionalJitterPass."""

    def test_no_jitter_on_straight_track(self):
        """A steadily-moving track should not have any frames removed."""
        data = _make_track([100 + 10 * i for i in range(10)])
        jitter_pass = PositionalJitterPass()
        result = jitter_pass.process(data)

        assert len(result["openlabel"]["frames"]) == 10
        for frame_data in result["openlabel"]["frames"].values():
            assert "obj_1" in frame_data["objects"]
        assert jitter_pass.frames_removed == 0

    def test_brief_single_reversal_not_flagged(self):
        """A single bounce (one direction reversal) is not sustained enough
        to be jitter — only a repeated zig-zag run should be flagged."""
        data = _make_track([100, 110, 120, 110, 120, 130, 140])
        jitter_pass = PositionalJitterPass()
        result = jitter_pass.process(data)

        assert jitter_pass.frames_removed == 0
        assert jitter_pass.objects_with_jitter == 0
        for frame_data in result["openlabel"]["frames"].values():
            assert "obj_1" in frame_data["objects"]

    def test_zigzag_run_removed(self):
        """A sustained zig-zag run is removed, preserving the track's first
        and last frames as anchors."""
        data = _make_track([100, 110, 100, 110, 100, 110])
        jitter_pass = PositionalJitterPass()
        result = jitter_pass.process(data)

        frames = result["openlabel"]["frames"]
        assert jitter_pass.objects_with_jitter == 1
        assert jitter_pass.frames_removed == 4
        assert "obj_1" in frames["0"]["objects"]
        assert "obj_1" in frames["5"]["objects"]
        for i in range(1, 5):
            assert "obj_1" not in frames[str(i)]["objects"]

    def test_mark_only_tags_instead_of_removing(self):
        """With mark_only=True, jitter frames are tagged, not deleted."""
        data = _make_track([100, 110, 100, 110, 100, 110])
        jitter_pass = PositionalJitterPass(mark_only=True)
        result = jitter_pass.process(data)

        frames = result["openlabel"]["frames"]
        assert jitter_pass.frames_removed == 0
        assert jitter_pass.frames_marked == 4
        for i in range(6):
            assert "obj_1" in frames[str(i)]["objects"]
        for i in range(1, 5):
            annotator_vals = frames[str(i)]["objects"]["obj_1"]["object_data"]["vec"][0]["val"]
            assert any("jitter" in val for val in annotator_vals)

    def test_minimal_length_track_boundary(self):
        """A track of exactly min_run_length+2 frames with jitter across all
        interior corners is handled without error, endpoints preserved."""
        data = _make_track([100, 110, 100, 110, 100])
        jitter_pass = PositionalJitterPass(min_run_length=3)
        result = jitter_pass.process(data)

        frames = result["openlabel"]["frames"]
        assert jitter_pass.frames_removed == 3
        assert "obj_1" in frames["0"]["objects"]
        assert "obj_1" in frames["4"]["objects"]

    def test_statistics_keys(self):
        """get_statistics returns the expected keys."""
        jitter_pass = PositionalJitterPass()
        stats = jitter_pass.get_statistics()
        for key in (
            "objects_checked",
            "objects_with_jitter",
            "jitter_runs_found",
            "frames_removed",
            "frames_marked",
        ):
            assert key in stats

    def test_decision_log_contains_run_details(self):
        """get_decision_log returns one record per removed jitter run."""
        data = _make_track([100, 110, 100, 110, 100, 110])
        jitter_pass = PositionalJitterPass()
        jitter_pass.process(data)

        log = jitter_pass.get_decision_log()
        assert len(log) == 1
        record = log[0]
        assert record["action"] == "remove_frames"
        assert record["object_id"] == "obj_1"
        assert record["frame_indices"] == [1, 2, 3, 4]
        assert record["details"]["run_length"] == 4
        assert record["details"]["max_turn_angle_deg"] > record["details"]["angle_threshold_deg"]

    def test_decision_log_mark_action(self):
        """get_decision_log reports action='mark_frames' when mark_only=True."""
        data = _make_track([100, 110, 100, 110, 100, 110])
        jitter_pass = PositionalJitterPass(mark_only=True)
        jitter_pass.process(data)

        log = jitter_pass.get_decision_log()
        assert len(log) == 1
        assert log[0]["action"] == "mark_frames"


class TestShortDurationPass:
    """Tests for ShortDurationPass.get_decision_log()."""

    def test_decision_log_for_removed_object(self):
        """get_decision_log returns one record per removed short-duration object."""
        data = _make_track([100, 110], w=50, h=30, annotator="oflow")
        short_pass = ShortDurationPass(min_frames=5, oflow_only=False)
        short_pass.process(data)

        log = short_pass.get_decision_log()
        assert len(log) == 1
        record = log[0]
        assert record["action"] == "remove_object"
        assert record["object_id"] == "obj_1"
        assert record["details"]["frame_count"] == 2
        assert record["details"]["min_frames"] == 5
        assert "2" in record["reason"] and "5" in record["reason"]

    def test_decision_log_empty_when_nothing_removed(self):
        """get_decision_log is empty when no object is short enough to remove."""
        data = _make_track([100 + 10 * i for i in range(10)])
        short_pass = ShortDurationPass(min_frames=5, oflow_only=False)
        short_pass.process(data)

        assert short_pass.get_decision_log() == []


class TestStaticObjectRemovalPass:
    """Tests for StaticObjectRemovalPass.get_decision_log()."""

    def test_decision_log_for_removed_static_object(self, ontology_path):
        """get_decision_log returns a remove_object record for a static object."""
        data = _make_track([100, 101, 100, 101, 100], annotator="yolo")
        static_pass = StaticObjectRemovalPass(static_threshold=20)
        static_pass.set_ontology_path(ontology_path)
        static_pass.process(data)

        log = static_pass.get_decision_log()
        assert len(log) == 1
        record = log[0]
        assert record["action"] == "remove_object"
        assert record["object_id"] == "obj_1"
        assert record["details"]["static_threshold"] == 20

    def test_decision_log_for_marked_static_object(self, ontology_path):
        """With mark_only=True, get_decision_log returns a mark_object record."""
        data = _make_track([100, 101, 100, 101, 100], annotator="yolo")
        static_pass = StaticObjectRemovalPass(static_threshold=20, mark_only=True)
        static_pass.set_ontology_path(ontology_path)
        static_pass.process(data)

        log = static_pass.get_decision_log()
        assert len(log) == 1
        assert log[0]["action"] == "mark_object"
        assert "frame" in log[0]
