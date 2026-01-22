"""
Unit tests for VLM response parser - JSON parsing and OpenLABEL conversion.

Tests the response_parser module without requiring a VLM connection.
"""

import json
import pytest

from markit.markitlib.vlm.response_parser import VLMResponseParser, SCENARIO_ONTOLOGY_UID


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


# --- Context Creation Tests ---

class TestCreateWeatherContext:
    """Tests for weather context creation."""

    def test_create_weather_context_full(self):
        """Create weather context with all fields."""
        weather = {
            "precipitation": "rain",
            "precipitation_intensity": "moderate",
            "particulates": "fog",
            "visibility_km": 2,
            "time_of_day": "day",
            "sun_position": "overcast",
            "cloud_cover": "overcast"
        }
        result = VLMResponseParser._create_weather_context(weather, 0, 100)

        assert result["name"] == "weather_conditions"
        assert result["type"] == "WeatherContext"
        assert result["ontology_uid"] == SCENARIO_ONTOLOGY_UID
        assert result["frame_intervals"] == [{"frame_start": 0, "frame_end": 100}]

        text_items = {item["name"]: item["val"] for item in result["context_data"]["text"]}
        assert text_items["precipitation"] == "rain"
        assert text_items["precipitation_intensity"] == "moderate"
        assert text_items["particulates"] == "fog"
        assert text_items["time_of_day"] == "day"
        assert text_items["sun_position"] == "overcast"
        assert text_items["cloud_cover"] == "overcast"

        num_items = {item["name"]: item["val"] for item in result["context_data"]["num"]}
        assert num_items["visibility_km"] == 2.0

    def test_create_weather_context_minimal(self):
        """Create weather context with minimal fields."""
        weather = {"precipitation": "none", "time_of_day": "night"}
        result = VLMResponseParser._create_weather_context(weather, 50, 150)

        assert result["frame_intervals"] == [{"frame_start": 50, "frame_end": 150}]
        text_items = {item["name"]: item["val"] for item in result["context_data"]["text"]}
        assert text_items["precipitation"] == "none"
        assert text_items["time_of_day"] == "night"


class TestCreateTrafficContext:
    """Tests for traffic context creation."""

    def test_create_traffic_context_full(self):
        """Create traffic context with all fields."""
        traffic = {
            "density": "dense",
            "flow": "unstable",
            "pedestrians_present": True,
            "cyclists_present": True,
            "special_vehicles_present": True,
            "temporary_structures": "construction"
        }
        result = VLMResponseParser._create_traffic_context(traffic, 0, 100)

        assert result["name"] == "traffic_conditions"
        assert result["type"] == "TrafficContext"

        text_items = {item["name"]: item["val"] for item in result["context_data"]["text"]}
        assert text_items["density"] == "dense"
        assert text_items["flow"] == "unstable"
        assert text_items["temporary_structures"] == "construction"

        bool_items = {item["name"]: item["val"] for item in result["context_data"]["boolean"]}
        assert bool_items["pedestrians_present"] is True
        assert bool_items["cyclists_present"] is True
        assert bool_items["special_vehicles_present"] is True


# --- OpenLABEL Contexts Tests ---

class TestToOpenlabelContexts:
    """Tests for converting analysis results to OpenLABEL contexts."""

    def test_single_frame_creates_contexts(self, sample_comprehensive_response):
        """Single frame analysis creates dynamic context types only.

        Note: Road, junction, and structures are static in aerial views and only
        appear in tags, not in frame-bound contexts.
        """
        results = [{**sample_comprehensive_response, "_frame_idx": 0}]
        frame_intervals = [{"frame_start": 0, "frame_end": 100}]

        contexts = VLMResponseParser.to_openlabel_contexts(results, frame_intervals)

        # Should have only weather and traffic contexts (dynamic types)
        # Road, junction, structures are static and only appear in tags
        assert len(contexts) == 2
        context_types = {c["type"] for c in contexts.values()}
        assert "WeatherContext" in context_types
        assert "TrafficContext" in context_types
        # Static types should NOT be in contexts
        assert "RoadContext" not in context_types
        assert "JunctionContext" not in context_types
        assert "StructuresContext" not in context_types

    def test_condition_change_creates_new_context(self):
        """Changing conditions create new context segments.

        Note: Intervals must be long enough (>= MIN_CONTEXT_INTERVAL_FRAMES)
        to avoid being merged as noise.
        """
        # Use multiple samples per condition to create intervals longer than 1 frame
        results = [
            {
                "_frame_idx": 0,
                "weather": {"precipitation": "none"},
                "traffic": {"density": "sparse"},
            },
            {
                "_frame_idx": 10,
                "weather": {"precipitation": "none"},  # Same as frame 0
                "traffic": {"density": "sparse"},
            },
            {
                "_frame_idx": 50,
                "weather": {"precipitation": "rain"},  # Changed!
                "traffic": {"density": "sparse"},
            },
            {
                "_frame_idx": 60,
                "weather": {"precipitation": "rain"},  # Same as frame 50
                "traffic": {"density": "sparse"},
            },
        ]
        frame_intervals = [{"frame_start": 0, "frame_end": 100}]

        contexts = VLMResponseParser.to_openlabel_contexts(results, frame_intervals)

        # Should have 2 weather contexts (one for each precipitation state)
        weather_contexts = [c for c in contexts.values() if c["type"] == "WeatherContext"]
        assert len(weather_contexts) == 2

        # First should end at frame 10, second should extend to video end
        weather_by_start = sorted(weather_contexts, key=lambda c: c["frame_intervals"][0]["frame_start"])
        assert weather_by_start[0]["frame_intervals"][0]["frame_end"] == 10
        assert weather_by_start[1]["frame_intervals"][0]["frame_end"] == 100

    def test_empty_results_returns_empty(self):
        """Empty results return empty contexts dict."""
        contexts = VLMResponseParser.to_openlabel_contexts([], [])
        assert contexts == {}

    def test_single_frame_intervals_merged(self):
        """Single-frame intervals are merged with adjacent segments.

        Single-frame intervals are likely VLM noise and should be merged
        with adjacent intervals using the most common value.
        """
        # Most common precipitation is "none" (frames 0 and 100)
        # Frame 50 has "rain" which is a single-frame interval
        results = [
            {
                "_frame_idx": 0,
                "weather": {"precipitation": "none"},
                "traffic": {"density": "sparse"},
            },
            {
                "_frame_idx": 50,
                "weather": {"precipitation": "rain"},  # Single frame - should be merged
                "traffic": {"density": "sparse"},
            },
            {
                "_frame_idx": 100,
                "weather": {"precipitation": "none"},
                "traffic": {"density": "sparse"},
            },
        ]
        frame_intervals = [{"frame_start": 0, "frame_end": 200}]

        contexts = VLMResponseParser.to_openlabel_contexts(results, frame_intervals)

        # The single-frame "rain" interval should be merged, resulting in
        # fewer weather contexts than without merging
        weather_contexts = [c for c in contexts.values() if c["type"] == "WeatherContext"]

        # With merging, we should have fewer contexts
        # The exact number depends on merging behavior, but it should be less than 3
        assert len(weather_contexts) <= 2


# --- OpenLABEL Tags Tests ---

class TestToOpenlabelTags:
    """Tests for converting analysis results to OpenLABEL tags."""

    def test_creates_all_tag_types(self, sample_comprehensive_response):
        """Creates tags for all categories plus metadata."""
        results = [sample_comprehensive_response]
        tags = VLMResponseParser.to_openlabel_tags(results, "test-model", 1)

        tag_types = {t["type"] for t in tags.values()}
        assert "WeatherTag" in tag_types
        assert "RoadTag" in tag_types
        assert "TrafficTag" in tag_types
        assert "JunctionTag" in tag_types
        assert "StructuresTag" in tag_types
        assert "VLMAnalysisTag" in tag_types

    def test_metadata_tag_content(self, sample_comprehensive_response):
        """Metadata tag contains model info and statistics."""
        results = [sample_comprehensive_response]
        tags = VLMResponseParser.to_openlabel_tags(results, "Qwen2.5-VL-32B", 5)

        metadata_tag = next(t for t in tags.values() if t["type"] == "VLMAnalysisTag")
        text_items = {item["name"]: item["val"] for item in metadata_tag["tag_data"]["text"]}
        num_items = {item["name"]: item["val"] for item in metadata_tag["tag_data"]["num"]}

        assert text_items["analyzer"] == "markit_vlm"
        assert text_items["model"] == "Qwen2.5-VL-32B"
        assert num_items["frames_analyzed"] == 5
        assert num_items["average_confidence"] == 0.9

    def test_notes_tag_created_when_notes_present(self, sample_roundabout_response):
        """Notes tag created when frames have notes."""
        results = [sample_roundabout_response]
        tags = VLMResponseParser.to_openlabel_tags(results, "test-model", 1)

        notes_tags = [t for t in tags.values() if t["type"] == "NotesTag"]
        assert len(notes_tags) == 1
        note_value = notes_tags[0]["tag_data"]["text"][0]["val"]
        assert "Traffic flowing smoothly" in note_value

    def test_junction_tag_captures_roundabout(self, sample_roundabout_response):
        """Junction tag correctly captures roundabout type."""
        results = [sample_roundabout_response]
        tags = VLMResponseParser.to_openlabel_tags(results, "test-model", 1)

        junction_tag = next(t for t in tags.values() if t["type"] == "JunctionTag")
        text_items = {item["name"]: item["val"] for item in junction_tag["tag_data"]["text"]}
        assert text_items["junction_type"] == "roundabout"
        assert text_items["roundabout_type"] == "normal"

    def test_only_first_frame_notes_used(self):
        """Only the first frame's notes are used for static scenes."""
        results = [
            {"weather": {"precipitation": "none"}, "notes": "Clear day with good visibility"},
            {"weather": {"precipitation": "none"}, "notes": "Sunny conditions observed"},
            {"weather": {"precipitation": "none"}, "notes": "Traffic flowing smoothly"},
        ]
        tags = VLMResponseParser.to_openlabel_tags(results, "test-model", 3)

        notes_tags = [t for t in tags.values() if t["type"] == "NotesTag"]
        assert len(notes_tags) == 1

        # Should only have the first frame's note
        note_value = notes_tags[0]["tag_data"]["text"][0]["val"]
        assert note_value == "Clear day with good visibility"


# --- Aggregation Tests ---

class TestAggregateResults:
    """Tests for aggregating multiple frame results."""

    def test_single_result_returned_unchanged(self, sample_comprehensive_response):
        """Single result returned as-is."""
        results = [sample_comprehensive_response]
        aggregated = VLMResponseParser._aggregate_results(results)
        assert aggregated == sample_comprehensive_response

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

    def test_weather_tag_has_per_field_provenance(self, sample_comprehensive_response):
        """WeatherTag has per-field annotator and confidence in vec format."""
        results = [sample_comprehensive_response]
        tags = VLMResponseParser.to_openlabel_tags(results, "test-model", 1)

        weather_tag = next(t for t in tags.values() if t["type"] == "WeatherTag")
        vec_items = self._get_vec_items(weather_tag["tag_data"])

        # Check per-field provenance for precipitation
        assert "precipitation_annotator" in vec_items
        assert "precipitation_confidence" in vec_items
        assert vec_items["precipitation_annotator"] == ["markit_vlm"]

        # Check other weather fields have provenance
        assert "time_of_day_annotator" in vec_items
        assert "time_of_day_confidence" in vec_items

    def test_road_tag_has_per_field_provenance(self, sample_comprehensive_response):
        """RoadTag has per-field annotator and confidence in vec format."""
        results = [sample_comprehensive_response]
        tags = VLMResponseParser.to_openlabel_tags(results, "test-model", 1)

        road_tag = next(t for t in tags.values() if t["type"] == "RoadTag")
        vec_items = self._get_vec_items(road_tag["tag_data"])

        assert "drivable_area_type_annotator" in vec_items
        assert "drivable_area_type_confidence" in vec_items
        assert vec_items["drivable_area_type_annotator"] == ["markit_vlm"]

    def test_traffic_tag_has_per_field_provenance(self, sample_comprehensive_response):
        """TrafficTag has per-field annotator and confidence in vec format."""
        results = [sample_comprehensive_response]
        tags = VLMResponseParser.to_openlabel_tags(results, "test-model", 1)

        traffic_tag = next(t for t in tags.values() if t["type"] == "TrafficTag")
        vec_items = self._get_vec_items(traffic_tag["tag_data"])

        assert "density_annotator" in vec_items
        assert "density_confidence" in vec_items

    def test_junction_tag_has_per_field_provenance(self, sample_comprehensive_response):
        """JunctionTag has per-field annotator and confidence in vec format."""
        results = [sample_comprehensive_response]
        tags = VLMResponseParser.to_openlabel_tags(results, "test-model", 1)

        junction_tag = next(t for t in tags.values() if t["type"] == "JunctionTag")
        vec_items = self._get_vec_items(junction_tag["tag_data"])

        assert "junction_type_annotator" in vec_items
        assert "junction_type_confidence" in vec_items

    def test_structures_tag_has_per_field_provenance(self, sample_comprehensive_response):
        """StructuresTag has per-field annotator and confidence in vec format."""
        results = [sample_comprehensive_response]
        tags = VLMResponseParser.to_openlabel_tags(results, "test-model", 1)

        structures_tag = next(t for t in tags.values() if t["type"] == "StructuresTag")
        vec_items = self._get_vec_items(structures_tag["tag_data"])

        assert "street_lighting_annotator" in vec_items
        assert "street_lighting_confidence" in vec_items

    def test_notes_tag_has_per_field_provenance(self, sample_roundabout_response):
        """NotesTag has per-field annotator and confidence in vec format."""
        results = [sample_roundabout_response]
        tags = VLMResponseParser.to_openlabel_tags(results, "test-model", 1)

        notes_tag = next(t for t in tags.values() if t["type"] == "NotesTag")
        vec_items = self._get_vec_items(notes_tag["tag_data"])

        assert "notes_annotator" in vec_items
        assert "notes_confidence" in vec_items
        assert vec_items["notes_annotator"] == ["markit_vlm"]

    def test_vlm_analysis_tag_unchanged(self, sample_comprehensive_response):
        """VLMAnalysisTag should NOT have vec provenance fields."""
        results = [sample_comprehensive_response]
        tags = VLMResponseParser.to_openlabel_tags(results, "test-model", 1)

        metadata_tag = next(t for t in tags.values() if t["type"] == "VLMAnalysisTag")
        text_items = {item["name"]: item["val"] for item in metadata_tag["tag_data"]["text"]}
        num_items = {item["name"]: item["val"] for item in metadata_tag["tag_data"]["num"]}

        # Should have analyzer (not annotator) and average_confidence (not confidence)
        assert "analyzer" in text_items
        assert "annotator" not in text_items
        assert "average_confidence" in num_items
        # vec provenance is not added to VLMAnalysisTag
        assert "vec" not in metadata_tag["tag_data"]

    def test_per_field_confidence_uses_averaged_value(self):
        """Per-field confidence uses averaged value when field has confidence from VLM."""
        results = [
            {
                "weather": {
                    "precipitation": "none",
                    "precipitation_confidence": 0.9,
                },
                "confidence": 0.9,
            },
            {
                "weather": {
                    "precipitation": "none",
                    "precipitation_confidence": 0.7,
                },
                "confidence": 0.7,
            },
        ]
        tags = VLMResponseParser.to_openlabel_tags(results, "test-model", 2)

        weather_tag = next(t for t in tags.values() if t["type"] == "WeatherTag")
        vec_items = self._get_vec_items(weather_tag["tag_data"])

        # Average of 0.9 and 0.7 is 0.8
        assert vec_items["precipitation_confidence"] == [0.8]
