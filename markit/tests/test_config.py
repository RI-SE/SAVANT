"""
Unit tests for configuration module - config parsing and ontology loading.
"""

import argparse
import pytest

from markit.markitlib.config import (
    Constants,
    DetectionResult,
    OpticalFlowParams,
    ConflictResolutionConfig,
    MarkitConfig,
)


class TestConstants:
    """Tests for Constants class."""

    def test_constants_have_expected_values(self):
        """Verify constants have expected values."""
        assert Constants.MP4V_FOURCC == "mp4v"
        assert Constants.SCHEMA_VERSION == "1.1"
        assert "SAVANT markit" in Constants.ANNOTATOR_NAME

    def test_default_class_map_exists(self):
        """Default class map should be defined."""
        assert isinstance(Constants.DEFAULT_CLASS_MAP, dict)
        assert len(Constants.DEFAULT_CLASS_MAP) > 0
        # Check expected vehicle classes
        assert 0 in Constants.DEFAULT_CLASS_MAP
        assert "vehicle" in Constants.DEFAULT_CLASS_MAP.values()


class TestDetectionResult:
    """Tests for DetectionResult dataclass."""

    def test_detection_result_creation(self, sample_detection):
        """Test creating a DetectionResult instance."""
        assert sample_detection.object_id == 1
        assert sample_detection.class_id == 1
        assert sample_detection.confidence == 0.95
        assert sample_detection.center == (150.0, 125.0)
        assert sample_detection.angle == 0.0
        assert sample_detection.source_engine == "yolo"
        assert sample_detection.width == 100.0
        assert sample_detection.height == 50.0

    def test_detection_result_optional_fields(self, sample_obb_bbox):
        """Test DetectionResult with optional fields as None."""
        detection = DetectionResult(
            object_id=None,
            class_id=1,
            confidence=0.8,
            oriented_bbox=sample_obb_bbox,
            center=(150.0, 125.0),
            angle=0.5,
            source_engine="optical_flow",
        )
        assert detection.object_id is None
        assert detection.width is None
        assert detection.height is None


class TestOpticalFlowParams:
    """Tests for OpticalFlowParams dataclass."""

    def test_default_values(self):
        """Test default optical flow parameters."""
        params = OpticalFlowParams()
        assert params.motion_threshold == 0.5
        assert params.min_area == 200
        assert params.morph_kernel_size == 9

    def test_custom_values(self):
        """Test custom optical flow parameters."""
        params = OpticalFlowParams(
            motion_threshold=0.7, min_area=300, morph_kernel_size=11
        )
        assert params.motion_threshold == 0.7
        assert params.min_area == 300
        assert params.morph_kernel_size == 11


class TestConflictResolutionConfig:
    """Tests for ConflictResolutionConfig dataclass."""

    def test_default_values(self):
        """Test default conflict resolution config."""
        config = ConflictResolutionConfig()
        assert config.iou_threshold == 0.3
        assert config.yolo_precedence is True
        assert config.enable_logging is False

    def test_custom_values(self):
        """Test custom conflict resolution config."""
        config = ConflictResolutionConfig(
            iou_threshold=0.5, yolo_precedence=False, enable_logging=True
        )
        assert config.iou_threshold == 0.5
        assert config.yolo_precedence is False
        assert config.enable_logging is True


class TestMarkitConfig:
    """Tests for MarkitConfig class."""

    def test_basic_config_creation(
        self, test_video_path, test_model_path, schema_path, ontology_path
    ):
        """Test creating a basic configuration."""
        args = argparse.Namespace(
            weights=test_model_path,
            input=test_video_path,
            output_json="/tmp/test_output.json",
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
            verbose=False,
        )
        config = MarkitConfig(args)

        assert config.weights_path == test_model_path
        assert config.video_path == test_video_path
        assert config.use_yolo is True
        assert config.use_optical_flow is False
        assert config.use_aruco is False

    def test_optical_flow_detection_method(
        self, test_video_path, schema_path, ontology_path
    ):
        """Test optical flow detection method configuration."""
        args = argparse.Namespace(
            weights="markit_yolo.pt",
            input=test_video_path,
            output_json="/tmp/test_output.json",
            schema=schema_path,
            output_video=None,
            ontology=ontology_path,
            aruco_csv=None,
            detection_method="optical_flow",
            motion_threshold=0.7,
            min_object_area=300,
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
            verbose=False,
        )
        config = MarkitConfig(args)

        assert config.use_yolo is False
        assert config.use_optical_flow is True
        assert config.optical_flow_params.motion_threshold == 0.7
        assert config.optical_flow_params.min_area == 300

    def test_both_detection_methods(
        self, test_video_path, test_model_path, schema_path, ontology_path
    ):
        """Test using both YOLO and optical flow."""
        args = argparse.Namespace(
            weights=test_model_path,
            input=test_video_path,
            output_json="/tmp/test_output.json",
            schema=schema_path,
            output_video=None,
            ontology=ontology_path,
            aruco_csv=None,
            detection_method="both",
            motion_threshold=0.5,
            min_object_area=200,
            iou_threshold=0.5,
            verbose_conflicts=True,
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
            verbose=False,
        )
        config = MarkitConfig(args)

        assert config.use_yolo is True
        assert config.use_optical_flow is True
        assert config.enable_conflict_resolution is True
        assert config.iou_threshold == 0.5
        assert config.verbose_conflicts is True

    def test_housekeeping_enabled(
        self, test_video_path, test_model_path, schema_path, ontology_path
    ):
        """Test housekeeping (postprocessing) configuration."""
        args = argparse.Namespace(
            weights=test_model_path,
            input=test_video_path,
            output_json="/tmp/test_output.json",
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
            housekeeping=True,
            duplicate_avg_iou=0.8,
            duplicate_min_iou=0.4,
            rotation_threshold=0.2,
            min_movement_pixels=10.0,
            temporal_smoothing=0.5,
            edge_distance=150,
            static_threshold=3,
            static_mark=True,
            verbose=False,
        )
        config = MarkitConfig(args)

        assert config.enable_housekeeping is True
        assert config.duplicate_avg_iou == 0.8
        assert config.duplicate_min_iou == 0.4
        assert config.rotation_threshold == 0.2
        assert config.min_movement_pixels == 10.0
        assert config.temporal_smoothing == 0.5
        assert config.edge_distance == 150
        assert config.static_threshold == 3
        assert config.static_mark is True

    def test_validation_missing_video_file(
        self, test_model_path, schema_path, ontology_path
    ):
        """Test validation fails when video file is missing."""
        args = argparse.Namespace(
            weights=test_model_path,
            input="/nonexistent/video.mp4",
            output_json="/tmp/test_output.json",
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
            verbose=False,
        )
        with pytest.raises(FileNotFoundError, match="video.mp4"):
            MarkitConfig(args)

    def test_validation_invalid_iou_threshold(
        self, test_video_path, test_model_path, schema_path, ontology_path
    ):
        """Test validation fails with invalid IoU threshold."""
        args = argparse.Namespace(
            weights=test_model_path,
            input=test_video_path,
            output_json="/tmp/test_output.json",
            schema=schema_path,
            output_video=None,
            ontology=ontology_path,
            aruco_csv=None,
            detection_method="yolo",
            motion_threshold=0.5,
            min_object_area=200,
            iou_threshold=1.5,  # Invalid: > 1.0
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
            verbose=False,
        )
        with pytest.raises(ValueError, match="IoU threshold"):
            MarkitConfig(args)

    def test_ontology_loading(
        self, test_video_path, test_model_path, schema_path, ontology_path
    ):
        """Test that ontology is loaded and class map is created."""
        args = argparse.Namespace(
            weights=test_model_path,
            input=test_video_path,
            output_json="/tmp/test_output.json",
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
            verbose=False,
        )
        config = MarkitConfig(args)

        # Should have loaded class map from ontology
        assert isinstance(config.class_map, dict)
        assert len(config.class_map) > 0

    def test_ontology_fallback_to_default(
        self, test_video_path, test_model_path, schema_path
    ):
        """Test fallback to default class map when ontology is missing."""
        args = argparse.Namespace(
            weights=test_model_path,
            input=test_video_path,
            output_json="/tmp/test_output.json",
            schema=schema_path,
            output_video=None,
            ontology="/nonexistent/ontology.ttl",
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
            verbose=False,
        )
        config = MarkitConfig(args)

        # Should fall back to default class map
        assert config.class_map == Constants.DEFAULT_CLASS_MAP
