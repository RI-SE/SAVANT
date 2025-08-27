import pytest
from savant_app.models.OpenLabel import OpenLabel
from pydantic import ValidationError
from tests.unit.test_utils import read_json
from pathlib import Path


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

    def test_full_serialization(self):
        """Test complete structure serialization matches expected format"""
        ol = OpenLabel(**self.expected_output["openlabel"])
        assert ol.model_dump() == self.expected_output["openlabel"]

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
