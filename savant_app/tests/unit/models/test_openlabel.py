import pytest
from savant_app.models.OpenLabel import OpenLabel, ObjectMetadata, FrameObjects
from pydantic import ValidationError
from tests.unit.test_utils import read_json
from pathlib import Path
from savant_app.models.OpenLabel import GeometryData, RotatedBBox as RB


class TestOpenLabel:
    """Unit tests for OpenLabel model validation and serialization"""

    def setup_method(self):
        self.expected_output = {
            "openlabel": {
                "metadata": {
                    "schema_version": "0.1",
                    "tagged_file": "datasets/videos/Kraklanda_short.mp4",
                    "annotator": "SAVANT Markit v0.1",
                },
                "ontologies": {"0": "https://savant.ri.se/savant_ontology_1.0.0.ttl"},
                "objects": {
                    "1": {"name": "Object-1", "type": "car", "ontology_uid": "0"},
                    "2": {"name": "Object-2", "type": "car", "ontology_uid": "0"},
                    "3": {"name": "Object-3", "type": "car", "ontology_uid": "0"},
                },
                "actions": {},
                "frames": {
                    "0": {
                        "objects": {
                            "1": {
                                "object_data": {
                                    "rbbox": [
                                        {
                                            "name": "shape",
                                            "val": [
                                                3594.0,
                                                1159.0,
                                                125.0,
                                                67.0,
                                                0.0019,
                                            ],
                                        }
                                    ],
                                    "vec": [
                                        {
                                            "name": "confidence",
                                            "val": [0.9142142534255981],
                                        }
                                    ],
                                }
                            },
                            "2": {
                                "object_data": {
                                    "rbbox": [
                                        {
                                            "name": "shape",
                                            "val": [
                                                3678.0,
                                                1031.0,
                                                134.0,
                                                70.0,
                                                0.0016,
                                            ],
                                        }
                                    ],
                                    "vec": [
                                        {
                                            "name": "confidence",
                                            "val": [0.9118316769599915],
                                        }
                                    ],
                                }
                            },
                        }
                    },
                    "1": {
                        "objects": {
                            "1": {
                                "object_data": {
                                    "rbbox": [
                                        {
                                            "name": "shape",
                                            "val": [3603.0, 1158.0, 75.0, 118.0, 1.57],
                                        }
                                    ],
                                    "vec": [
                                        {
                                            "name": "confidence",
                                            "val": [0.9127626419067383],
                                        }
                                    ],
                                }
                            },
                            "2": {
                                "object_data": {
                                    "rbbox": [
                                        {
                                            "name": "shape",
                                            "val": [
                                                3664.0,
                                                1031.0,
                                                135.0,
                                                69.0,
                                                0.0012,
                                            ],
                                        }
                                    ],
                                    "vec": [
                                        {
                                            "name": "confidence",
                                            "val": [0.905584454536438],
                                        }
                                    ],
                                }
                            },
                            "3": {
                                "object_data": {
                                    "rbbox": [
                                        {
                                            "name": "shape",
                                            "val": [473.0, 1129.0, 114.0, 74.0, 0.0003],
                                        }
                                    ],
                                    "vec": [
                                        {
                                            "name": "confidence",
                                            "val": [0.8935688138008118],
                                        }
                                    ],
                                }
                            },
                        }
                    },
                    "2": {
                        "objects": {
                            "1": {
                                "object_data": {
                                    "rbbox": [
                                        {
                                            "name": "shape",
                                            "val": [3618.0, 1158.0, 65.0, 125.0, 1.57],
                                        }
                                    ],
                                    "vec": [
                                        {
                                            "name": "confidence",
                                            "val": [0.9137364029884338],
                                        }
                                    ],
                                }
                            },
                            "3": {
                                "object_data": {
                                    "rbbox": [
                                        {
                                            "name": "shape",
                                            "val": [488.0, 1132.0, 122.0, 66.0, 0.0037],
                                        }
                                    ],
                                    "vec": [
                                        {
                                            "name": "confidence",
                                            "val": [0.9032099843025208],
                                        }
                                    ],
                                }
                            },
                        }
                    },
                },
            }
        }

    def test_add_new_object(self):
        """Test adding a new object to the OpenLabel model"""
        ol = OpenLabel(**self.expected_output["openlabel"])
        new_object_id = "4"
        obj_type = "person"

        ol.add_new_object(obj_type, new_object_id)

        assert new_object_id in ol.objects
        obj = ol.objects[new_object_id]
        assert isinstance(obj, ObjectMetadata)
        assert obj.type == obj_type
        assert obj.name == f"Object-{new_object_id}"

    def test_full_serialization(self):
        """Test complete structure serialization matches expected format"""
        ol = OpenLabel(**self.expected_output["openlabel"])
        assert ol.model_dump() == self.expected_output["openlabel"]

    def test_append_new_object_bbox(self):
        """Test appending a new bounding box to a frame"""
        ol = OpenLabel(**self.expected_output["openlabel"])
        frame_id = 0
        bbox_coordinates = [100, 200, 50, 60]  # Pass as list instead of dictionary
        confidence_data = {"val": [0.95]}
        annotater_data = {"val": ["test_annotator"]}
        new_bbox_key = "10"

        # Create frame if it doesn't exist
        if str(frame_id) not in ol.frames.keys():
            ol.frames[str(frame_id)] = FrameObjects(objects={})

        ol.append_object_bbox(
            frame_id, bbox_coordinates, confidence_data, annotater_data, new_bbox_key
        )

        frame = ol.frames[str(frame_id)]
        assert new_bbox_key in frame.objects
        bbox = frame.objects[new_bbox_key].object_data.rbbox[0].val
        assert bbox.x_center == 100
        assert bbox.y_center == 200
        assert bbox.width == 50
        assert bbox.height == 60
        assert bbox.rotation == 0  # Method currently sets rotation to 0

    def test_append_existing_object_bbox(self):
        """Test appending a bbox to an existing object"""
        ol = OpenLabel(**self.expected_output["openlabel"])
        frame_id = 1  # Use a frame that already exists
        obj_id = "1"  # Existing object from setup
        bbox_coordinates = [300, 400, 70, 80]
        confidence_data = {"val": [0.85]}
        annotater_data = {"val": ["another_annotator"]}

        ol.append_object_bbox(
            frame_id=frame_id,
            bbox_coordinates=bbox_coordinates,
            confidence_data=confidence_data,
            annotater_data=annotater_data,
            obj_id=obj_id,
        )

        # Verify the object is still in the frame
        frame = ol.frames[str(frame_id)]
        assert obj_id in frame.objects

        # Verify bbox data
        bbox = frame.objects[obj_id].object_data.rbbox[0].val
        assert bbox.x_center == 300
        assert bbox.y_center == 400
        assert bbox.width == 70
        assert bbox.height == 80
        assert bbox.rotation == 0

    def test_append_existing_object_bbox_new_object(self):
        """Test appending a bbox to a new object in a frame"""
        ol = OpenLabel(**self.expected_output["openlabel"])
        frame_id = 2  # Use a frame that already exists
        obj_id = "new_obj"  # New object for this frame
        bbox_coordinates = [500, 600, 90, 100]
        confidence_data = {"val": [0.75]}
        annotater_data = {"val": ["new_annotator"]}

        # Ensure object doesn't exist in this frame initially
        if obj_id in ol.frames[str(frame_id)].objects:
            del ol.frames[str(frame_id)].objects[obj_id]

        ol.append_object_bbox(
            frame_id=frame_id,
            bbox_coordinates=bbox_coordinates,
            confidence_data=confidence_data,
            annotater_data=annotater_data,
            obj_id=obj_id,
        )

        # Verify the object is now in the frame
        frame = ol.frames[str(frame_id)]
        assert obj_id in frame.objects

        # Verify bbox data
        bbox = frame.objects[obj_id].object_data.rbbox[0].val
        assert bbox.x_center == 500
        assert bbox.y_center == 600
        assert bbox.width == 90
        assert bbox.height == 100
        assert bbox.rotation == 0

    def test_model_dump_excludes_none(self):
        """Test model_dump excludes None values"""
        # Create a minimal OpenLabel instance with None values
        data = {
            "metadata": {
                "schema_version": "0.1",
                "tagged_file": None,  # Should be excluded
                "annotator": None,  # Should be excluded
            },
            "ontologies": {"0": "https://example.com/ontology.ttl"},
            "objects": {},
            "frames": {},
        }
        ol = OpenLabel(**data)
        dumped = ol.model_dump()

        # Verify None values are excluded
        assert "tagged_file" not in dumped["metadata"]
        assert "annotator" not in dumped["metadata"]
        assert "actions" not in dumped  # Not present and should be excluded
        assert "frame_intervals" not in dumped["objects"]  # Not present in objects

    def test_missing_metadata(self):
        """Test validation fails when required metadata is missing"""
        invalid_data = self.expected_output["openlabel"].copy()
        del invalid_data["metadata"]
        with pytest.raises(ValidationError):
            OpenLabel(**invalid_data)

    def test_invalid_confidence_value(self):
        """Test confidence values outside 0-1 range are rejected"""
        invalid_data = self.expected_output["openlabel"].copy()
        invalid_data["frames"]["0"]["objects"]["1"]["object_data"]["vec"][0]["val"] = [
            1.1
        ]
        with pytest.raises(ValidationError):
            OpenLabel(**invalid_data)

    def test_real_world_json_validation(self):
        """Test validation of actual JSON file from TestVids directory"""
        json_path = str(
            Path(__file__).parent.parent.parent / "assets" / "Kraklanda_short.json"
        )
        test_data = read_json(json_path)
        # Validate against Pydantic model
        ol = OpenLabel(**test_data["openlabel"])
        assert ol.model_dump() == test_data["openlabel"]

    def teardown_method(self):
        del self.expected_output
    
    def test_get_bbox_happy_path(self):
        """get_bbox returns the expected RotatedBBox for an existing frame/object."""
        ol = OpenLabel(**self.expected_output["openlabel"])
        bbox = ol.get_bbox(frame_key=1, object_key=1)
        assert bbox.x_center == 3603.0
        assert bbox.y_center == 1158.0
        assert bbox.width == 75.0
        assert bbox.height == 118.0
        assert bbox.rotation == 1.57

    def test_get_bbox_missing_frame_raises(self):
        """get_bbox should raise a clear KeyError for a missing frame."""
        ol = OpenLabel(**self.expected_output["openlabel"])
        with pytest.raises(KeyError, match="Frame '99' not found"):
            ol.get_bbox(frame_key=99, object_key=1)

    def test_get_bbox_missing_object_raises(self):
        """get_bbox should raise a clear KeyError for a missing object in a frame."""
        ol = OpenLabel(**self.expected_output["openlabel"])
        with pytest.raises(KeyError, match="Object '999' not found"):
            ol.get_bbox(frame_key=1, object_key=999)

    def test_get_bbox_index_out_of_range_raises(self):
        """get_bbox should raise IndexError if bbox_index is out of range."""
        ol = OpenLabel(**self.expected_output["openlabel"])
        with pytest.raises(IndexError, match="rbbox index 1 out of range"):
            ol.get_bbox(frame_key=1, object_key=1, bbox_index=1)

    def test_update_bbox_absolute_replacement(self):
        """update_bbox replaces values when absolute args are provided."""

        ol = OpenLabel(**self.expected_output["openlabel"])
        updated = ol.update_bbox(
            frame_key=1,
            object_key=1,
            x_center=200.0,
            y_center=80.0,
            width=50.0,
            height=30.0,
            rotation=0.1,
        )
        assert (updated.x_center, updated.y_center, updated.width,
                updated.height, updated.rotation) == (
            200.0, 80.0, 50.0, 30.0, 0.1
        )
        again = ol.get_bbox(frame_key=1, object_key=1)
        assert again == updated

    def test_update_bbox_deltas_add_to_current(self):
        """update_bbox applies deltas relative to current values."""

        ol = OpenLabel(**self.expected_output["openlabel"])
        updated = ol.update_bbox(
            frame_key=1,
            object_key=1,
            delta_x=5.0,
            delta_y=-2.0,
            delta_w=10.0,
            delta_h=4.0,
            delta_theta=0.25,
        )
        assert updated.x_center == 3608.0
        assert updated.y_center == 1156.0
        assert updated.width == 85.0
        assert updated.height == 122.0
        assert updated.rotation == pytest.approx(1.57 + 0.25)

    def test_update_bbox_mix_absolute_and_delta(self):
        """Absolute values override current; deltas then apply on top of that."""

        ol = OpenLabel(**self.expected_output["openlabel"])
        updated = ol.update_bbox(frame_key=1, object_key=1, width=50.0, delta_y=10.0)
        assert updated.width == 50.0
        assert updated.y_center == 1158.0 + 10.0

    def test_update_bbox_clamps_min_size(self):
        """Clamps prevent non-positive sizes."""

        ol = OpenLabel(**self.expected_output["openlabel"])
        updated = ol.update_bbox(
            frame_key=1,
            object_key=1,
            width=1e-9,
            height=1e-9,
            min_width=1e-6,
            min_height=1e-6,
        )
        assert updated.width == pytest.approx(1e-6)
        assert updated.height == pytest.approx(1e-6)

    def test_update_bbox_validation_error_if_forced_invalid(self):
        """If clamping is disabled and width/height are forced invalid, Pydantic should raise."""

        ol = OpenLabel(**self.expected_output["openlabel"])
        with pytest.raises(ValidationError):
            ol.update_bbox(
                frame_key=1,
                object_key=1,
                width=0.0,
                height=0.0,
                min_width=0.0,
                min_height=0.0,
            )

    def test_update_specific_bbox_index_multiple_entries(self):
        """When multiple rbboxes exist, bbox_index targets only the intended entry."""

        ol = OpenLabel(**self.expected_output["openlabel"])
        frame = ol.frames["1"]
        obj = frame.objects["1"]
        obj.object_data.rbbox.append(obj.object_data.rbbox[0])
        first_list = obj.object_data.rbbox[0].val.model_dump()
        obj.object_data.rbbox[-1] = GeometryData(val=RB.model_validate(first_list))

        updated = ol.update_bbox(frame_key=1, object_key=1, bbox_index=1, delta_x=3.0)
        assert updated.x_center == obj.object_data.rbbox[1].val.x_center == 3603.0 + 3.0

        first = ol.get_bbox(frame_key=1, object_key=1, bbox_index=0)
        assert first.x_center == 3603.0
