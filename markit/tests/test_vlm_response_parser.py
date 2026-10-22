"""
Unit tests for VLM response parser - JSON parsing and OpenLABEL conversion.

Tests the response_parser module without requiring a VLM connection.
"""

import json
import pytest

from markit.markitlib.vlm.response_parser import VLMResponseParser
ONTOLOGY_UID = "1"


def _to_tags(results, model_name="test-model", frames_analyzed=1):
    return VLMResponseParser.to_openlabel_tags(
        results, model_name, frames_analyzed, ontology_uid=ONTOLOGY_UID
    )


def _to_contexts(results, frame_intervals):
    return VLMResponseParser.to_openlabel_contexts(results, frame_intervals, ONTOLOGY_UID)


def _tag_by_field(tags, field_name):
    """Return the tag whose tag_data carries field_name."""
    for tag in tags.values():
        for items in tag["tag_data"].values():
            if any(item["name"] == field_name for item in items):
                return tag
    raise AssertionError(f"No tag for field {field_name}")


# --- Fixtures ---

@pytest.fixture
def sample_comprehensive_response():
    """Sample VLM response matching the new schema."""
    return {
        "weather": {
            "precipitation": "none",
            "precipitation_intensity": "none",
            "particulates": "none",
            "visibility_km": 10,
            "time_of_day": "day",
            "sun_position": "behind",
            "cloud_cover": "clear"
        },
        "road": {
            "drivable_area_type": "distributor",
            "geometry_horizontal": "straight",
            "geometry_longitudinal": "level",
            "divided": True,
            "surface_type": "asphalt",
            "surface_condition": "dry",
            "surface_quality": "good",
            "lane_count": 4,
            "lane_markings_visible": True
        },
        "junction": {
            "present": True,
            "type": "crossroads",
            "roundabout_type": "not_applicable",
            "signalized": True,
            "pedestrian_crossing": True,
            "rail_crossing": False
        },
        "traffic": {
            "density": "moderate",
            "flow": "stable",
            "pedestrians_present": True,
            "cyclists_present": False,
            "special_vehicles_present": False,
            "temporary_structures": "none"
        },
        "structures": {
            "bridge": False,
            "tunnel": False,
            "toll_plaza": False,
            "barriers_present": False,
            "street_lighting": "present"
        },
        "notes": None,
        "confidence": 0.9
    }


@pytest.fixture
def sample_roundabout_response():
    """Sample VLM response for a roundabout scene."""
    return {
        "weather": {
            "precipitation": "rain",
            "precipitation_intensity": "light",
            "particulates": "none",
            "visibility_km": 5,
            "time_of_day": "day",
            "sun_position": "overcast",
            "cloud_cover": "overcast"
        },
        "road": {
            "drivable_area_type": "minor",
            "geometry_horizontal": "curved",
            "geometry_longitudinal": "level",
            "divided": False,
            "surface_type": "asphalt",
            "surface_condition": "wet",
            "surface_quality": "good",
            "lane_count": 2,
            "lane_markings_visible": True
        },
        "junction": {
            "present": True,
            "type": "roundabout",
            "roundabout_type": "normal",
            "signalized": False,
            "pedestrian_crossing": False,
            "rail_crossing": False
        },
        "traffic": {
            "density": "sparse",
            "flow": "free_flow",
            "pedestrians_present": False,
            "cyclists_present": False,
            "special_vehicles_present": False,
            "temporary_structures": "none"
        },
        "structures": {
            "bridge": False,
            "tunnel": False,
            "toll_plaza": False,
            "barriers_present": False,
            "street_lighting": "present"
        },
        "notes": "Traffic flowing smoothly through roundabout",
        "confidence": 0.85
    }


# --- JSON Extraction Tests ---

class TestExtractJson:
    """Tests for JSON extraction from VLM responses."""

    def test_extract_direct_json(self, sample_comprehensive_response):
        """Extract JSON when response is pure JSON."""
        json_str = json.dumps(sample_comprehensive_response)
        result = VLMResponseParser.extract_json(json_str)
        assert result == sample_comprehensive_response

    def test_extract_json_with_whitespace(self, sample_comprehensive_response):
        """Extract JSON with leading/trailing whitespace."""
        json_str = f"  \n{json.dumps(sample_comprehensive_response)}\n  "
        result = VLMResponseParser.extract_json(json_str)
        assert result == sample_comprehensive_response

    def test_extract_json_from_markdown_code_block(self, sample_comprehensive_response):
        """Extract JSON from markdown code block."""
        text = f"Here is the analysis:\n```json\n{json.dumps(sample_comprehensive_response)}\n```\nEnd of analysis."
        result = VLMResponseParser.extract_json(text)
        assert result == sample_comprehensive_response

    def test_extract_json_from_generic_code_block(self, sample_comprehensive_response):
        """Extract JSON from generic code block."""
        text = f"Analysis:\n```\n{json.dumps(sample_comprehensive_response)}\n```"
        result = VLMResponseParser.extract_json(text)
        assert result == sample_comprehensive_response

    def test_extract_json_embedded_in_text(self, sample_comprehensive_response):
        """Extract JSON embedded in surrounding text."""
        text = f"Based on my analysis, {json.dumps(sample_comprehensive_response)} is the result."
        result = VLMResponseParser.extract_json(text)
        assert result == sample_comprehensive_response

    def test_extract_json_returns_none_for_invalid(self):
        """Return None when no valid JSON found."""
        text = "This is not JSON at all"
        result = VLMResponseParser.extract_json(text)
        assert result is None

    def test_extract_json_returns_none_for_malformed(self):
        """Return None for malformed JSON."""
        text = '{"weather": {"precipitation": "none"'  # Missing closing braces
        result = VLMResponseParser.extract_json(text)
        assert result is None


# --- Comprehensive Response Parsing Tests ---

class TestParseComprehensiveResponse:
    """Tests for parsing comprehensive VLM responses."""

    def test_parse_valid_response(self, sample_comprehensive_response):
        """Parse a valid comprehensive response."""
        json_str = json.dumps(sample_comprehensive_response)
        result = VLMResponseParser.parse_comprehensive_response(json_str)
        assert result is not None
        assert "weather" in result
        assert "road" in result
        assert "junction" in result
        assert "traffic" in result
        assert "structures" in result

    def test_parse_partial_response(self):
        """Parse response with only some sections."""
        partial = {
            "weather": {"precipitation": "none", "time_of_day": "day"},
            "road": {"drivable_area_type": "motorway", "surface_condition": "dry"},
            "confidence": 0.7
        }
        result = VLMResponseParser.parse_comprehensive_response(json.dumps(partial))
        assert result is not None
        assert "weather" in result
        assert "road" in result

    def test_parse_returns_none_for_insufficient_keys(self):
        """Return None when no expected keys present."""
        # Parser returns partial data if at least 1 expected key is found
        # Only returns None when 0 expected keys are found
        insufficient = {"confidence": 0.5, "other": "data"}
        result = VLMResponseParser.parse_comprehensive_response(json.dumps(insufficient))
        assert result is None

    def test_parse_returns_none_for_invalid_json(self):
        """Return None for invalid JSON."""
        result = VLMResponseParser.parse_comprehensive_response("not json")
        assert result is None

    def test_parse_returns_none_for_non_object(self):
        assert VLMResponseParser.parse_comprehensive_response('["weather", "road"]') is None

    def test_malformed_values_are_coerced_or_dropped(self):
        malformed = {
            "weather": "sunny",
            "road": {
                "surface_condition": "Wet",
                "surface_condition_confidence": "0.8",
                "lane_count": "3",
                "surface_type": "gravel",
                "divided": "false",
                "divided_confidence": 7,
            },
            "traffic": {"special_vehicles_present": "false", "pedestrians_present": "maybe"},
            "junction": {"present": "true"},
            "confidence": "high",
        }
        result = VLMResponseParser.parse_comprehensive_response(json.dumps(malformed))
        assert "weather" not in result
        assert "confidence" not in result
        assert result["road"] == {
            "surface_condition": "wet",
            "surface_condition_confidence": 0.8,
            "lane_count": 3,
            "divided": False,
        }
        assert result["traffic"] == {"special_vehicles_present": False}
        assert result["junction"] == {"present": True}

    def test_malformed_response_produces_tags_without_crash(self):
        malformed = {
            "road": {"surface_condition": "wet", "surface_condition_confidence": "0.8"},
            "traffic": {"special_vehicles_present": "false"},
            "structures": 5,
        }
        parsed = VLMResponseParser.parse_comprehensive_response(json.dumps(malformed))
        results = [dict(parsed, _frame_idx=0), dict(parsed, _frame_idx=30)]
        types = {tag["type"] for tag in _to_tags(results, frames_analyzed=2).values()}
        assert "SurfaceConditionWet" in types
        assert "TrafficSpecialVehicle" not in types


# --- Context Creation Tests ---

class TestCreateFieldContext:
    """Tests for single-field context creation."""

    def test_context_typed_by_class(self):
        weather = {
            "precipitation": "rain",
            "precipitation_confidence": 0.8,
            "precipitation_rationale": "Wet streaks visible",
        }
        result = VLMResponseParser._create_field_context(
            "precipitation", weather, 0, 100, ONTOLOGY_UID
        )

        assert result["name"] == "precipitation"
        assert result["type"] == "WeatherRain"
        assert result["ontology_uid"] == ONTOLOGY_UID
        assert result["frame_intervals"] == [{"frame_start": 0, "frame_end": 100}]
        text_items = {i["name"]: i["val"] for i in result["context_data"]["text"]}
        assert text_items == {
            "precipitation": "rain",
            "precipitation_rationale": "Wet streaks visible",
        }
        vec_items = {i["name"]: i["val"] for i in result["context_data"]["vec"]}
        assert vec_items["precipitation_confidence"] == [0.8]

    def test_context_typed_by_extension_class(self):
        result = VLMResponseParser._create_field_context(
            "density", {"density": "dense"}, 0, 100, ONTOLOGY_UID
        )
        assert result["type"] == "TrafficDensityDense"

    def test_value_without_class_gives_no_context(self):
        result = VLMResponseParser._create_field_context(
            "precipitation", {"precipitation": "unknown"}, 0, 100, ONTOLOGY_UID
        )
        assert result is None


# --- OpenLABEL Contexts Tests ---

class TestToOpenlabelContexts:
    """Tests for converting analysis results to OpenLABEL contexts."""

    def test_single_frame_creates_contexts(self, sample_roundabout_response):
        """Only dynamic fields (precipitation, density) become contexts."""
        results = [{**sample_roundabout_response, "_frame_idx": 0}]
        frame_intervals = [{"frame_start": 0, "frame_end": 100}]

        contexts = _to_contexts(results, frame_intervals)

        assert {c["type"] for c in contexts.values()} == {
            "WeatherRain", "TrafficDensitySparse"
        }

    def test_condition_change_creates_new_context(self):
        """Changing conditions create new context segments; a value holds until the next sample.

        Note: Segments need >= MIN_CONTEXT_SAMPLES samples to avoid being merged as noise.
        """
        results = [
            {"_frame_idx": 0, "weather": {"precipitation": "snow"}, "traffic": {"density": "sparse"}},
            {"_frame_idx": 10, "weather": {"precipitation": "snow"}, "traffic": {"density": "sparse"}},
            {"_frame_idx": 50, "weather": {"precipitation": "rain"}, "traffic": {"density": "sparse"}},
            {"_frame_idx": 60, "weather": {"precipitation": "rain"}, "traffic": {"density": "sparse"}},
        ]
        frame_intervals = [{"frame_start": 0, "frame_end": 100}]

        contexts = _to_contexts(results, frame_intervals)

        by_type = {c["type"]: c["frame_intervals"][0] for c in contexts.values()}
        assert by_type["WeatherSnow"] == {"frame_start": 0, "frame_end": 49}
        assert by_type["WeatherRain"] == {"frame_start": 50, "frame_end": 100}
        assert by_type["TrafficDensitySparse"] == {"frame_start": 0, "frame_end": 100}

    def test_context_counts_samples_of_its_interval(self):
        """A one-sample outlier merged into a segment lowers that segment's agreement."""
        results = [
            {"_frame_idx": 0, "weather": {"precipitation": "snow"}},
            {"_frame_idx": 10, "weather": {"precipitation": "snow"}},
            {"_frame_idx": 20, "weather": {"precipitation": "rain"}},
            {"_frame_idx": 30, "weather": {"precipitation": "snow"}},
        ]
        contexts = _to_contexts(results, [{"frame_start": 0, "frame_end": 40}])

        (snow,) = [c for c in contexts.values() if c["type"] == "WeatherSnow"]
        num = {i["name"]: i["val"] for i in snow["context_data"]["num"]}
        assert num["precipitation_n_samples"] == 4 and num["precipitation_agreement"] == 0.75

    def test_no_precipitation_segment_is_explicit(self):
        """'none' gets its own context, so dry frames differ from undeterminable ones (no context)."""
        results = [
            {"_frame_idx": 0, "weather": {"precipitation": "none"}},
            {"_frame_idx": 10, "weather": {"precipitation": "none"}},
            {"_frame_idx": 50, "weather": {"precipitation": "rain"}},
            {"_frame_idx": 60, "weather": {"precipitation": "rain"}},
        ]
        contexts = _to_contexts(results, [{"frame_start": 0, "frame_end": 100}])

        by_type = {c["type"]: c["frame_intervals"][0] for c in contexts.values()}
        assert by_type == {"WeatherNoPrecipitation": {"frame_start": 0, "frame_end": 49},
                           "WeatherRain": {"frame_start": 50, "frame_end": 100}}

    def test_unknown_precipitation_segment_omitted(self):
        results = [
            {"_frame_idx": 0, "weather": {"precipitation": "unknown"}},
            {"_frame_idx": 10, "weather": {"precipitation": "unknown"}},
            {"_frame_idx": 50, "weather": {"precipitation": "rain"}},
            {"_frame_idx": 60, "weather": {"precipitation": "rain"}},
        ]
        contexts = _to_contexts(results, [{"frame_start": 0, "frame_end": 100}])
        assert [c["type"] for c in contexts.values()] == ["WeatherRain"]

    def test_empty_results_returns_empty(self):
        """Empty results return empty contexts dict."""
        assert _to_contexts([], []) == {}

    def test_single_frame_intervals_merged(self):
        """Single-frame intervals are merged with adjacent segments.

        Single-frame intervals are likely VLM noise and should be merged
        with adjacent intervals using the most common value.
        """
        results = [
            {"_frame_idx": 0, "weather": {"precipitation": "snow"}},
            {"_frame_idx": 50, "weather": {"precipitation": "rain"}},  # Single frame - merged
            {"_frame_idx": 100, "weather": {"precipitation": "snow"}},
        ]
        contexts = _to_contexts(results, [{"frame_start": 0, "frame_end": 200}])

        intervals = [c["frame_intervals"] for c in contexts.values() if c["type"] == "WeatherSnow"]
        assert intervals == [[{"frame_start": 0, "frame_end": 200}]]

    def test_segments_cover_whole_video_without_gaps(self):
        results = [
            {"_frame_idx": 5, "weather": {"precipitation": "rain"}},
            {"_frame_idx": 30, "weather": {"precipitation": "rain"}},
            {"_frame_idx": 60, "weather": {"precipitation": "snow"}},
            {"_frame_idx": 90, "weather": {"precipitation": "snow"}},
        ]
        contexts = _to_contexts(results, [{"frame_start": 0, "frame_end": 120}])

        by_type = {c["type"]: c["frame_intervals"][0] for c in contexts.values()}
        assert by_type["WeatherRain"] == {"frame_start": 0, "frame_end": 59}
        assert by_type["WeatherSnow"] == {"frame_start": 60, "frame_end": 120}

    def test_majority_vote_tie_goes_to_first_seen_value(self):
        results = [
            {"_frame_idx": 0, "road": {"surface_condition": "wet"}},
            {"_frame_idx": 30, "road": {"surface_condition": "dry"}},
        ]
        types = {t["type"] for t in _to_tags(results, frames_analyzed=2).values()}
        assert "SurfaceConditionWet" in types and "SurfaceConditionDry" not in types


# --- OpenLABEL Tags Tests ---

class TestToOpenlabelTags:
    """Tests for converting analysis results to OpenLABEL tags."""

    def test_tags_typed_by_scenario_classes(self, sample_comprehensive_response):
        """Each field value with a class becomes one tag of that class."""
        tags = _to_tags([sample_comprehensive_response])

        assert {t["ontology_uid"] for t in tags.values()} == {ONTOLOGY_UID}
        assert {t["type"] for t in tags.values()} == {
            "WeatherNoPrecipitation",
            "EnvironmentVisibility",
            "IlluminationDay",
            "SunPositionBehind",
            "CloudinessClear",
            "RoadTypeDistributor",
            "HorizontalStraights",
            "LongitudinalLevelPlane",
            "TransverseDivided",
            "SurfaceTypeAsphalt",
            "SurfaceConditionDry",
            "LaneSpecificationLaneCount",
            "LaneSpecificationMarking",
            "IntersectionCrossroad",
            "JunctionSignalized",
            "SpecialStructurePedestrianCrossing",
            "FixedStructureStreetlight",
            "TrafficDensityModerate",
            "TrafficFlowStable",
            "TrafficAgentType",
            "VLMAnalysisTag",
        }

    def test_tag_keys_are_sequential(self, sample_comprehensive_response):
        tags = _to_tags([sample_comprehensive_response])
        assert list(tags) == [str(i) for i in range(len(tags))]

    def test_classless_values_produce_no_tag(self, sample_comprehensive_response):
        """'none' (except precipitation), 'good', 'not_applicable' and False presence values yield no tags."""
        tags = _to_tags([sample_comprehensive_response])
        fields = {
            item["name"]
            for t in tags.values()
            for items in t["tag_data"].values()
            for item in items
        }
        for absent in ("precipitation_intensity", "particulates", "surface_quality",
                       "roundabout_type", "bridge", "cyclists_present"):
            assert absent not in fields

    def test_numeric_value_uses_property_name(self, sample_comprehensive_response):
        tags = _to_tags([sample_comprehensive_response])
        lane_tag = _tag_by_field(tags, "laneSpecificationLaneCountValue")
        assert lane_tag["tag_data"]["num"] == [
            {"name": "laneSpecificationLaneCountValue", "val": 4},
            {"name": "laneSpecificationLaneCountValue_n_samples", "val": 1},
            {"name": "laneSpecificationLaneCountValue_agreement", "val": 1.0},
        ]

    def test_agent_type_tag_has_openlabel_value(self, sample_comprehensive_response):
        tags = _to_tags([sample_comprehensive_response])
        tag = _tag_by_field(tags, "pedestrians_present")
        text_items = {i["name"]: i["val"] for i in tag["tag_data"]["text"]}
        assert text_items["trafficAgentTypeValue"] == "HumanPedestrian"

    def test_metadata_tag_content(self, sample_comprehensive_response):
        """Metadata tag contains model info and statistics."""
        tags = _to_tags([sample_comprehensive_response], "Qwen2.5-VL-32B", 5)

        metadata_tag = next(t for t in tags.values() if t["type"] == "VLMAnalysisTag")
        text_items = {item["name"]: item["val"] for item in metadata_tag["tag_data"]["text"]}
        num_items = {item["name"]: item["val"] for item in metadata_tag["tag_data"]["num"]}

        assert text_items["analyzer"] == "markit_vlm"
        assert text_items["model"] == "Qwen2.5-VL-32B"
        assert num_items["frames_analyzed"] == 5
        assert num_items["average_confidence"] == 0.9

    def test_notes_tag_created_when_notes_present(self, sample_roundabout_response):
        """Notes tag created when frames have notes."""
        tags = _to_tags([sample_roundabout_response])

        notes_tags = [t for t in tags.values() if t["type"] == "NotesTag"]
        assert len(notes_tags) == 1
        note_value = notes_tags[0]["tag_data"]["text"][0]["val"]
        assert "Traffic flowing smoothly" in note_value

    def test_roundabout_tags(self, sample_roundabout_response):
        """Roundabout produces junction and roundabout-size classes."""
        tags = _to_tags([sample_roundabout_response])

        types = {t["type"] for t in tags.values()}
        assert {"JunctionRoundabout", "RoundaboutNormal", "WeatherRain",
                "PrecipitationLight", "CloudinessOvercast",
                "TransverseUndivided"} <= types

    def test_only_first_frame_notes_used(self):
        """Only the first frame's notes are used for static scenes."""
        results = [
            {"weather": {"precipitation": "none"}, "notes": "Clear day with good visibility"},
            {"weather": {"precipitation": "none"}, "notes": "Sunny conditions observed"},
            {"weather": {"precipitation": "none"}, "notes": "Traffic flowing smoothly"},
        ]
        tags = _to_tags(results, frames_analyzed=3)

        notes_tags = [t for t in tags.values() if t["type"] == "NotesTag"]
        assert len(notes_tags) == 1
        note_value = notes_tags[0]["tag_data"]["text"][0]["val"]
        assert note_value == "Clear day with good visibility"


# --- Aggregation Tests ---

class TestAggregateResults:
    """Tests for aggregating multiple frame results."""

    def test_single_result_returned_unchanged(self, sample_comprehensive_response):
        """Single result returned as-is, plus one agreeing sample per field."""
        results = [sample_comprehensive_response]
        aggregated = VLMResponseParser._aggregate_results(results)
        for category, data in sample_comprehensive_response.items():
            if not isinstance(data, dict):
                assert aggregated[category] == data
                continue
            assert {k: v for k, v in aggregated[category].items() if k in data} == data
            added = {k: v for k, v in aggregated[category].items() if k not in data}
            assert set(added.values()) <= {1, 1.0} and all(k.endswith(("_n_samples", "_agreement")) for k in added)
        assert aggregated["weather"]["precipitation_n_samples"] == 1

    def test_sample_counts_and_agreement(self):
        results = [
            {"weather": {"precipitation": "none"}, "junction": {"pedestrian_crossing": False}},
            {"weather": {"precipitation": "none"}, "junction": {"pedestrian_crossing": True}},
            {"weather": {"precipitation": "rain"}, "junction": {"pedestrian_crossing": False}},
        ]
        aggregated = VLMResponseParser._aggregate_results(results)
        assert aggregated["weather"]["precipitation_n_samples"] == 3
        assert aggregated["weather"]["precipitation_agreement"] == pytest.approx(0.6667)
        # Presence: seen once is enough; agreement says how often it was seen
        assert aggregated["junction"]["pedestrian_crossing"] is True
        assert aggregated["junction"]["pedestrian_crossing_agreement"] == pytest.approx(0.3333)

    def test_majority_voting_for_categorical(self):
        """Categorical fields use majority voting."""
        results = [
            {"weather": {"precipitation": "none"}},
            {"weather": {"precipitation": "none"}},
            {"weather": {"precipitation": "rain"}},
        ]
        aggregated = VLMResponseParser._aggregate_results(results)
        assert aggregated["weather"]["precipitation"] == "none"

    def test_averaging_for_numeric(self):
        """Numeric fields use averaging."""
        results = [
            {"weather": {"visibility_km": 10}},
            {"weather": {"visibility_km": 5}},
            {"weather": {"visibility_km": 6}},
        ]
        aggregated = VLMResponseParser._aggregate_results(results)
        assert aggregated["weather"]["visibility_km"] == 7.0  # (10+5+6)/3

    def test_lane_count_rounds_to_int(self):
        """Lane count averaging rounds to integer."""
        results = [
            {"road": {"lane_count": 2}},
            {"road": {"lane_count": 3}},
            {"road": {"lane_count": 3}},
        ]
        aggregated = VLMResponseParser._aggregate_results(results)
        # (2+3+3)/3 = 2.67 -> rounds to 3
        assert aggregated["road"]["lane_count"] == 3

    def test_any_true_wins_for_presence(self):
        """Presence fields (pedestrians, cyclists) use any-True logic."""
        results = [
            {"traffic": {"pedestrians_present": False, "cyclists_present": False}},
            {"traffic": {"pedestrians_present": True, "cyclists_present": False}},
            {"traffic": {"pedestrians_present": False, "cyclists_present": False}},
        ]
        aggregated = VLMResponseParser._aggregate_results(results)
        assert aggregated["traffic"]["pedestrians_present"] is True
        assert aggregated["traffic"]["cyclists_present"] is False

    def test_junction_presence_any_true(self):
        """Junction presence uses any-True logic."""
        results = [
            {"junction": {"present": False, "type": "none"}},
            {"junction": {"present": True, "type": "roundabout"}},
        ]
        aggregated = VLMResponseParser._aggregate_results(results)
        assert aggregated["junction"]["present"] is True
        # Type should be majority voted - tie goes to first in set order
        assert aggregated["junction"]["type"] in ["none", "roundabout"]

    def test_structures_any_true(self):
        """Structure booleans use any-True logic."""
        results = [
            {"structures": {"bridge": False, "tunnel": True}},
            {"structures": {"bridge": False, "tunnel": False}},
        ]
        aggregated = VLMResponseParser._aggregate_results(results)
        assert aggregated["structures"]["bridge"] is False
        assert aggregated["structures"]["tunnel"] is True

    def test_confidence_averaging(self):
        """Confidence values are averaged."""
        results = [
            {"confidence": 0.9},
            {"confidence": 0.8},
            {"confidence": 0.7},
        ]
        aggregated = VLMResponseParser._aggregate_results(results)
        assert aggregated["confidence"] == pytest.approx(0.8, rel=1e-6)  # (0.9+0.8+0.7)/3

    def test_empty_results_returns_empty(self):
        """Empty results return empty dict."""
        assert VLMResponseParser._aggregate_results([]) == {}


# --- Confidence Calculation Tests ---

class TestAverageConfidence:
    """Tests for confidence averaging."""

    def test_average_multiple_confidences(self):
        """Average confidence from multiple results."""
        results = [
            {"confidence": 0.9},
            {"confidence": 0.8},
            {"confidence": 0.7},
        ]
        avg = VLMResponseParser._average_confidence(results)
        assert avg == pytest.approx(0.8, rel=1e-6)

    def test_single_confidence(self):
        """Single confidence returned as-is."""
        results = [{"confidence": 0.85}]
        avg = VLMResponseParser._average_confidence(results)
        assert avg == 0.85

    def test_missing_confidence_ignored(self):
        """Results without confidence are ignored."""
        results = [
            {"confidence": 0.9},
            {"weather": {"precipitation": "none"}},  # No confidence
            {"confidence": 0.7},
        ]
        avg = VLMResponseParser._average_confidence(results)
        assert avg == 0.8  # (0.9+0.7)/2

    def test_no_confidence_returns_zero(self):
        """No confidence values returns 0.0."""
        results = [{"weather": {"precipitation": "none"}}]
        avg = VLMResponseParser._average_confidence(results)
        assert avg == 0.0

    def test_empty_results_returns_zero(self):
        """Empty results returns 0.0."""
        avg = VLMResponseParser._average_confidence([])
        assert avg == 0.0


# --- Tag Provenance Fields Tests ---

class TestTagProvenanceFields:
    """Tests for per-field annotator and confidence provenance fields on VLM tags.

    Provenance fields use vec format (list-based) to support multi-annotator tracking.
    Each field has its own {field}_annotator and {field}_confidence entries.
    When a human edits a field, their annotator ID is prepended to that field's list.
    """

    def _get_vec_items(self, tag_data: dict) -> dict:
        """Helper to extract vec items as a dict of name -> val."""
        return {item["name"]: item["val"] for item in tag_data.get("vec", [])}

    @pytest.mark.parametrize("field_name", [
        "time_of_day", "drivable_area_type", "density", "junction_type",
        "street_lighting", "divided", "laneSpecificationLaneCountValue",
    ])
    def test_field_tag_has_provenance(self, sample_comprehensive_response, field_name):
        tags = _to_tags([sample_comprehensive_response])
        vec_items = self._get_vec_items(_tag_by_field(tags, field_name)["tag_data"])

        assert vec_items[f"{field_name}_annotator"] == ["markit_vlm"]
        assert f"{field_name}_confidence" in vec_items

    def test_notes_tag_has_per_field_provenance(self, sample_roundabout_response):
        """NotesTag has per-field annotator and confidence in vec format."""
        tags = _to_tags([sample_roundabout_response])

        notes_tag = next(t for t in tags.values() if t["type"] == "NotesTag")
        vec_items = self._get_vec_items(notes_tag["tag_data"])

        assert vec_items["notes_annotator"] == ["markit_vlm"]
        assert "notes_confidence" in vec_items

    def test_vlm_analysis_tag_unchanged(self, sample_comprehensive_response):
        """VLMAnalysisTag should NOT have vec provenance fields."""
        tags = _to_tags([sample_comprehensive_response])

        metadata_tag = next(t for t in tags.values() if t["type"] == "VLMAnalysisTag")
        text_items = {item["name"]: item["val"] for item in metadata_tag["tag_data"]["text"]}
        num_items = {item["name"]: item["val"] for item in metadata_tag["tag_data"]["num"]}

        assert "analyzer" in text_items
        assert "annotator" not in text_items
        assert "average_confidence" in num_items
        assert "vec" not in metadata_tag["tag_data"]

    def test_per_field_confidence_uses_averaged_value(self):
        """Per-field confidence uses averaged value when field has confidence from VLM."""
        results = [
            {"weather": {"precipitation": "rain", "precipitation_confidence": 0.9}, "confidence": 0.9},
            {"weather": {"precipitation": "rain", "precipitation_confidence": 0.7}, "confidence": 0.7},
        ]
        tags = _to_tags(results, frames_analyzed=2)

        rain_tag = next(t for t in tags.values() if t["type"] == "WeatherRain")
        vec_items = self._get_vec_items(rain_tag["tag_data"])

        # Average of 0.9 and 0.7 is 0.8
        assert vec_items["precipitation_confidence"] == [0.8]


def test_every_field_spec_survives_multi_frame_aggregation():
    """Aggregation is driven by FIELD_SPECS, so a new field can't be silently dropped."""
    from savant_common.scenario_tags import FIELD_SPECS

    frame = {}
    for spec in FIELD_SPECS.values():
        value = 3 if spec.is_numeric else spec.choices[0]
        frame.setdefault(spec.category, {})[spec.vlm_key] = value
    results = [dict(frame, _frame_idx=0), dict(frame, _frame_idx=30)]

    aggregated = VLMResponseParser._aggregate_results(results)

    for spec in FIELD_SPECS.values():
        assert spec.vlm_key in aggregated[spec.category], (spec.category, spec.vlm_key)
