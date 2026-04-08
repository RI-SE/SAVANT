"""
response_parser - Parse VLM responses into structured data for OpenLABEL

Provides parsing of VLM text responses and conversion to OpenLABEL tags.
Schema aligned with BSI PAS-1883 ODD taxonomy.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# SAVANT scenario ontology namespace and URI
SCENARIO_ONTOLOGY_UID = "1"
SCENARIO_ONTOLOGY_URI = "https://github.com/RI-SE/SAVANT/tree/main/ontology/savant-scenario#"

# Default annotator name for VLM-generated data
VLM_ANNOTATOR = "markit_vlm"


class VLMResponseParser:
    """Parse VLM text responses into structured data."""

    @staticmethod
    def extract_json(text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON object from VLM response text.

        Args:
            text: Raw VLM response text

        Returns:
            Parsed JSON dict or None if extraction fails
        """
        # Try direct JSON parse first
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in text (VLM may include extra text)
        patterns = [
            r"```json\s*([\s\S]*?)\s*```",  # Markdown code block
            r"```\s*([\s\S]*?)\s*```",  # Generic code block
            r"(\{[\s\S]*\})",  # Raw JSON object
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue

        logger.warning(f"Failed to extract JSON from VLM response: {text[:200]}...")
        return None

    @staticmethod
    def parse_comprehensive_response(response: str) -> Optional[Dict[str, Any]]:
        """Parse comprehensive analysis response.

        Args:
            response: VLM response text

        Returns:
            Parsed analysis dict or None if parsing fails
        """
        data = VLMResponseParser.extract_json(response)
        if not data:
            return None

        # Validate expected structure (at least some keys should be present)
        expected_keys = ["weather", "road", "traffic", "junction", "structures"]
        found_keys = [key for key in expected_keys if key in data]

        if len(found_keys) < 2:
            logger.warning(
                f"VLM response missing expected keys. Found: {list(data.keys())}"
            )
            # Still return partial data if some analysis is present
            if not found_keys:
                return None

        return data

    # Minimum interval length in frames. Intervals shorter than this are considered
    # potential VLM noise and will be merged with adjacent intervals.
    MIN_CONTEXT_INTERVAL_FRAMES = 2

    @staticmethod
    def to_openlabel_contexts(
        analysis_results: List[Dict[str, Any]],
        frame_intervals: List[Dict[str, int]],
    ) -> Dict[str, Dict[str, Any]]:
        """Convert VLM analysis results to OpenLABEL contexts with frame intervals.

        Creates contexts that track when conditions change over time. A new context
        is only created when a condition differs from the previous analyzed frame.

        Note: Only dynamic context types (weather, traffic) are generated as contexts.
        Static types (road, junction, structures) are only included in tags since
        they don't change in a static aerial view.

        Args:
            analysis_results: List of parsed VLM analysis dicts (must have _frame_idx)
            frame_intervals: Video frame range for extending final segment

        Returns:
            OpenLABEL contexts dict with frame_intervals for each condition segment
        """
        if not analysis_results:
            return {}

        # Sort results by frame index
        sorted_results = sorted(
            analysis_results, key=lambda x: x.get("_frame_idx", 0)
        )

        # Get video frame range for final segment extension
        video_start = frame_intervals[0]["frame_start"] if frame_intervals else 0
        video_end = frame_intervals[0]["frame_end"] if frame_intervals else None

        # Build raw segments for dynamic context types only
        # Static types (road, junction, structures) are only in tags
        weather_segments = VLMResponseParser._build_segments(
            sorted_results, "weather", "precipitation", video_start, video_end
        )
        traffic_segments = VLMResponseParser._build_segments(
            sorted_results, "traffic", "density", video_start, video_end
        )

        # Merge single-frame intervals with adjacent segments
        weather_segments = VLMResponseParser._merge_short_intervals(
            weather_segments, sorted_results, "weather", "precipitation"
        )
        traffic_segments = VLMResponseParser._merge_short_intervals(
            traffic_segments, sorted_results, "traffic", "density"
        )

        # Build contexts dict
        contexts = {}
        context_id = 0

        for segment in weather_segments:
            contexts[str(context_id)] = VLMResponseParser._create_weather_context(
                segment["data"], segment["start"], segment["end"]
            )
            context_id += 1

        for segment in traffic_segments:
            contexts[str(context_id)] = VLMResponseParser._create_traffic_context(
                segment["data"], segment["start"], segment["end"]
            )
            context_id += 1

        return contexts

    @staticmethod
    def _build_segments(
        sorted_results: List[Dict[str, Any]],
        category: str,
        key_field: str,
        video_start: int,
        video_end: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Build raw segments for a context category by tracking key field changes.

        Args:
            sorted_results: Frame analysis results sorted by frame index
            category: Category name (e.g., "weather", "traffic")
            key_field: Field to track for changes (e.g., "precipitation", "density")
            video_start: First frame of video
            video_end: Last frame of video (for extending final segment)

        Returns:
            List of segment dicts with keys: start, end, key, data
        """
        segments = []
        current_segment = None
        prev_frame = video_start

        for result in sorted_results:
            frame_idx = result.get("_frame_idx", 0)
            data = result.get(category, {})
            key_value = data.get(key_field)

            if current_segment is None or key_value != current_segment["key"]:
                # Close previous segment
                if current_segment is not None:
                    current_segment["end"] = prev_frame
                    segments.append(current_segment)
                # Start new segment
                current_segment = {
                    "key": key_value,
                    "start": frame_idx,
                    "end": None,
                    "data": data,
                }

            prev_frame = frame_idx

        # Close final segment
        if current_segment is not None:
            current_segment["end"] = video_end if video_end is not None else prev_frame
            segments.append(current_segment)

        return segments

    @staticmethod
    def _merge_short_intervals(
        segments: List[Dict[str, Any]],
        sorted_results: List[Dict[str, Any]],
        category: str,
        key_field: str,
    ) -> List[Dict[str, Any]]:
        """Merge single-frame intervals with adjacent segments using most common value.

        Single-frame intervals are likely VLM noise. They should be merged into the
        adjacent interval that has the most common value for the key field.

        Args:
            segments: List of segments from _build_segments
            sorted_results: Original analysis results for computing most common value
            category: Category name for extracting values
            key_field: Field to use for most common value computation

        Returns:
            List of merged segments
        """
        if len(segments) <= 1:
            return segments

        # Compute most common value across all results
        values = [
            r.get(category, {}).get(key_field)
            for r in sorted_results
            if r.get(category, {}).get(key_field) is not None
        ]
        if not values:
            return segments

        most_common = max(set(values), key=values.count)

        # Find short intervals and merge them
        merged = []
        i = 0
        while i < len(segments):
            segment = segments[i]
            interval_length = segment["end"] - segment["start"]

            if interval_length < VLMResponseParser.MIN_CONTEXT_INTERVAL_FRAMES:
                # Short interval - merge with adjacent using most common value
                if i > 0 and merged[-1]["key"] == most_common:
                    # Extend previous segment to cover this one
                    merged[-1]["end"] = segment["end"]
                elif i < len(segments) - 1 and segments[i + 1]["key"] == most_common:
                    # Will be absorbed by next segment - skip this one
                    # Extend next segment's start backwards
                    segments[i + 1]["start"] = segment["start"]
                elif i > 0:
                    # No ideal match - just extend previous segment
                    merged[-1]["end"] = segment["end"]
                else:
                    # First segment and short - keep it but it will be merged next iteration
                    merged.append(segment)
            else:
                merged.append(segment)

            i += 1

        return merged

    @staticmethod
    def _create_weather_context(
        weather: Dict[str, Any], frame_start: int, frame_end: int
    ) -> Dict[str, Any]:
        """Create a weather context for a frame interval."""
        context_data = {"text": [], "num": [], "vec": []}

        # Weather fields and their confidence
        weather_fields = [
            "precipitation", "precipitation_intensity", "particulates",
            "time_of_day", "sun_position", "cloud_cover"
        ]

        for field in weather_fields:
            if weather.get(field):
                context_data["text"].append({"name": field, "val": weather[field]})
                # Add per-field annotator and confidence
                context_data["vec"].append(
                    {"name": f"{field}_annotator", "val": [VLM_ANNOTATOR]}
                )
                conf = weather.get(f"{field}_confidence", 0.5)
                context_data["vec"].append(
                    {"name": f"{field}_confidence", "val": [round(conf, 4)]}
                )
                # Add rationale if present
                rationale = weather.get(f"{field}_rationale")
                if rationale:
                    context_data["text"].append({"name": f"{field}_rationale", "val": rationale})

        # Visibility is numeric
        if weather.get("visibility_km") is not None:
            context_data["num"].append(
                {"name": "visibility_km", "val": float(weather["visibility_km"])}
            )
            context_data["vec"].append(
                {"name": "visibility_km_annotator", "val": [VLM_ANNOTATOR]}
            )
            conf = weather.get("visibility_km_confidence", 0.5)
            context_data["vec"].append(
                {"name": "visibility_km_confidence", "val": [round(conf, 4)]}
            )

        context_data = {k: v for k, v in context_data.items() if v}

        return {
            "name": "weather_conditions",
            "type": "WeatherContext",
            "ontology_uid": SCENARIO_ONTOLOGY_UID,
            "frame_intervals": [{"frame_start": frame_start, "frame_end": frame_end}],
            "context_data": context_data,
        }

    @staticmethod
    def _create_traffic_context(
        traffic: Dict[str, Any], frame_start: int, frame_end: int
    ) -> Dict[str, Any]:
        """Create a traffic context for a frame interval."""
        context_data = {"text": [], "boolean": [], "vec": []}

        # Text fields
        text_fields = ["density", "flow", "temporary_structures"]
        for field in text_fields:
            if traffic.get(field):
                context_data["text"].append({"name": field, "val": traffic[field]})
                context_data["vec"].append(
                    {"name": f"{field}_annotator", "val": [VLM_ANNOTATOR]}
                )
                conf = traffic.get(f"{field}_confidence", 0.5)
                context_data["vec"].append(
                    {"name": f"{field}_confidence", "val": [round(conf, 4)]}
                )
                # Add rationale if present
                rationale = traffic.get(f"{field}_rationale")
                if rationale:
                    context_data["text"].append({"name": f"{field}_rationale", "val": rationale})

        # Boolean fields
        bool_fields = ["pedestrians_present", "cyclists_present", "special_vehicles_present"]
        for field in bool_fields:
            if field in traffic:
                context_data["boolean"].append({"name": field, "val": traffic[field]})
                context_data["vec"].append(
                    {"name": f"{field}_annotator", "val": [VLM_ANNOTATOR]}
                )
                conf = traffic.get(f"{field}_confidence", 0.5)
                context_data["vec"].append(
                    {"name": f"{field}_confidence", "val": [round(conf, 4)]}
                )
                # Add rationale if present
                rationale = traffic.get(f"{field}_rationale")
                if rationale:
                    context_data["text"].append({"name": f"{field}_rationale", "val": rationale})

        context_data = {k: v for k, v in context_data.items() if v}

        return {
            "name": "traffic_conditions",
            "type": "TrafficContext",
            "ontology_uid": SCENARIO_ONTOLOGY_UID,
            "frame_intervals": [{"frame_start": frame_start, "frame_end": frame_end}],
            "context_data": context_data,
        }

    @staticmethod
    def _add_field_provenance(
        tag_data: Dict[str, Any],
        field_name: str,
        confidence: float,
    ) -> None:
        """Add per-field annotator and confidence to tag_data vec array.

        Args:
            tag_data: The tag_data dict to augment
            field_name: Name of the field
            confidence: The confidence value for this field
        """
        if "vec" not in tag_data:
            tag_data["vec"] = []
        tag_data["vec"].append({"name": f"{field_name}_annotator", "val": [VLM_ANNOTATOR]})
        tag_data["vec"].append({"name": f"{field_name}_confidence", "val": [round(confidence, 4)]})

    @staticmethod
    def to_openlabel_tags(
        analysis_results: List[Dict[str, Any]],
        model_name: str,
        frames_analyzed: int,
    ) -> Dict[str, Dict[str, Any]]:
        """Convert VLM analysis to OpenLABEL tags for scenario-level metadata.

        Per OpenLABEL spec, tags are for scenario categorization and organization.
        Each field has its own annotator and confidence in the vec array.

        Args:
            analysis_results: List of parsed VLM analysis dicts
            model_name: Name of the VLM model used
            frames_analyzed: Number of frames that were analyzed

        Returns:
            OpenLABEL tags dict ready for insertion
        """
        tags = {}
        tag_id = 0

        aggregated = VLMResponseParser._aggregate_results(analysis_results)
        avg_confidence = VLMResponseParser._average_confidence(analysis_results)

        # Weather tag
        if "weather" in aggregated:
            weather = aggregated["weather"]
            tag_data = {"text": [], "num": [], "vec": []}

            # Text fields with per-field provenance
            text_fields = [
                "precipitation", "precipitation_intensity", "particulates",
                "time_of_day", "sun_position", "cloud_cover"
            ]
            for field in text_fields:
                if field in weather:
                    tag_data["text"].append({"name": field, "val": weather[field]})
                    conf = weather.get(f"{field}_confidence", avg_confidence)
                    VLMResponseParser._add_field_provenance(tag_data, field, conf)
                    # Add rationale if present
                    rationale = weather.get(f"{field}_rationale")
                    if rationale:
                        tag_data["text"].append({"name": f"{field}_rationale", "val": rationale})

            # Numeric field
            if weather.get("visibility_km") is not None:
                tag_data["num"].append(
                    {"name": "visibility_km", "val": float(weather["visibility_km"])}
                )
                conf = weather.get("visibility_km_confidence", avg_confidence)
                VLMResponseParser._add_field_provenance(tag_data, "visibility_km", conf)

            tag_data = {k: v for k, v in tag_data.items() if v}

            if tag_data:
                tags[str(tag_id)] = {
                    "name": "weather_conditions",
                    "type": "WeatherTag",
                    "ontology_uid": SCENARIO_ONTOLOGY_UID,
                    "tag_data": tag_data,
                }
                tag_id += 1

        # Road tag
        if "road" in aggregated:
            road = aggregated["road"]
            tag_data = {"text": [], "num": [], "boolean": [], "vec": []}

            # Text fields
            text_fields = [
                "drivable_area_type", "geometry_horizontal", "geometry_longitudinal",
                "surface_type", "surface_condition", "surface_quality"
            ]
            for field in text_fields:
                if field in road:
                    tag_data["text"].append({"name": field, "val": road[field]})
                    conf = road.get(f"{field}_confidence", avg_confidence)
                    VLMResponseParser._add_field_provenance(tag_data, field, conf)
                    # Add rationale if present
                    rationale = road.get(f"{field}_rationale")
                    if rationale:
                        tag_data["text"].append({"name": f"{field}_rationale", "val": rationale})

            # Numeric field
            if road.get("lane_count") is not None:
                tag_data["num"].append(
                    {"name": "lane_count", "val": int(road["lane_count"])}
                )
                conf = road.get("lane_count_confidence", avg_confidence)
                VLMResponseParser._add_field_provenance(tag_data, "lane_count", conf)

            # Boolean fields
            bool_fields = ["divided", "lane_markings_visible"]
            for field in bool_fields:
                if field in road:
                    tag_data["boolean"].append({"name": field, "val": road[field]})
                    conf = road.get(f"{field}_confidence", avg_confidence)
                    VLMResponseParser._add_field_provenance(tag_data, field, conf)

            tag_data = {k: v for k, v in tag_data.items() if v}

            if tag_data:
                tags[str(tag_id)] = {
                    "name": "road_infrastructure",
                    "type": "RoadTag",
                    "ontology_uid": SCENARIO_ONTOLOGY_UID,
                    "tag_data": tag_data,
                }
                tag_id += 1

        # Traffic tag
        if "traffic" in aggregated:
            traffic = aggregated["traffic"]
            tag_data = {"text": [], "boolean": [], "vec": []}

            # Text fields
            text_fields = ["density", "flow", "temporary_structures"]
            for field in text_fields:
                if field in traffic:
                    tag_data["text"].append({"name": field, "val": traffic[field]})
                    conf = traffic.get(f"{field}_confidence", avg_confidence)
                    VLMResponseParser._add_field_provenance(tag_data, field, conf)
                    # Add rationale if present
                    rationale = traffic.get(f"{field}_rationale")
                    if rationale:
                        tag_data["text"].append({"name": f"{field}_rationale", "val": rationale})

            # Boolean fields
            bool_fields = ["pedestrians_present", "cyclists_present", "special_vehicles_present"]
            for field in bool_fields:
                if field in traffic:
                    tag_data["boolean"].append({"name": field, "val": traffic[field]})
                    conf = traffic.get(f"{field}_confidence", avg_confidence)
                    VLMResponseParser._add_field_provenance(tag_data, field, conf)
                    # Add rationale if present (includes frame info)
                    rationale = traffic.get(f"{field}_rationale")
                    if rationale:
                        tag_data["text"].append({"name": f"{field}_rationale", "val": rationale})

            tag_data = {k: v for k, v in tag_data.items() if v}

            if tag_data:
                tags[str(tag_id)] = {
                    "name": "traffic_conditions",
                    "type": "TrafficTag",
                    "ontology_uid": SCENARIO_ONTOLOGY_UID,
                    "tag_data": tag_data,
                }
                tag_id += 1

        # Junction tag
        if "junction" in aggregated:
            junction = aggregated["junction"]
            tag_data = {"text": [], "boolean": [], "vec": []}

            # Text fields
            text_fields = ["type", "roundabout_type"]
            field_names = {"type": "junction_type", "roundabout_type": "roundabout_type"}
            for field in text_fields:
                if field in junction:
                    tag_data["text"].append({"name": field_names[field], "val": junction[field]})
                    conf = junction.get(f"{field}_confidence", avg_confidence)
                    VLMResponseParser._add_field_provenance(tag_data, field_names[field], conf)

            # Boolean fields
            bool_fields = ["present", "signalized", "pedestrian_crossing", "rail_crossing"]
            bool_names = {
                "present": "junction_present",
                "signalized": "signalized",
                "pedestrian_crossing": "pedestrian_crossing",
                "rail_crossing": "rail_crossing"
            }
            for field in bool_fields:
                if field in junction:
                    tag_data["boolean"].append({"name": bool_names[field], "val": junction[field]})
                    conf = junction.get(f"{field}_confidence", avg_confidence)
                    VLMResponseParser._add_field_provenance(tag_data, bool_names[field], conf)

            tag_data = {k: v for k, v in tag_data.items() if v}

            if tag_data:
                tags[str(tag_id)] = {
                    "name": "junction_info",
                    "type": "JunctionTag",
                    "ontology_uid": SCENARIO_ONTOLOGY_UID,
                    "tag_data": tag_data,
                }
                tag_id += 1

        # Structures tag
        if "structures" in aggregated:
            structures = aggregated["structures"]
            tag_data = {"text": [], "boolean": [], "vec": []}

            # Text field
            if "street_lighting" in structures:
                tag_data["text"].append(
                    {"name": "street_lighting", "val": structures["street_lighting"]}
                )
                conf = structures.get("street_lighting_confidence", avg_confidence)
                VLMResponseParser._add_field_provenance(tag_data, "street_lighting", conf)

            # Boolean fields
            bool_fields = ["bridge", "tunnel", "toll_plaza", "barriers_present"]
            for field in bool_fields:
                if field in structures:
                    tag_data["boolean"].append({"name": field, "val": structures[field]})
                    conf = structures.get(f"{field}_confidence", avg_confidence)
                    VLMResponseParser._add_field_provenance(tag_data, field, conf)

            tag_data = {k: v for k, v in tag_data.items() if v}

            if tag_data:
                tags[str(tag_id)] = {
                    "name": "structures_info",
                    "type": "StructuresTag",
                    "ontology_uid": SCENARIO_ONTOLOGY_UID,
                    "tag_data": tag_data,
                }
                tag_id += 1

        # Notes tag - use only the first frame's notes since static scenes
        # produce similar (but differently phrased) observations
        first_note = next(
            (r.get("notes") for r in analysis_results if r.get("notes")), None
        )
        if first_note:
            tag_data = {
                "text": [{"name": "notes", "val": first_note}],
                "vec": [
                    {"name": "notes_annotator", "val": [VLM_ANNOTATOR]},
                    {"name": "notes_confidence", "val": [round(avg_confidence, 4)]},
                ],
            }
            tags[str(tag_id)] = {
                "name": "scene_notes",
                "type": "NotesTag",
                "ontology_uid": SCENARIO_ONTOLOGY_UID,
                "tag_data": tag_data,
            }
            tag_id += 1

        # VLM analysis metadata tag
        tags[str(tag_id)] = {
            "name": "vlm_analysis_metadata",
            "type": "VLMAnalysisTag",
            "ontology_uid": SCENARIO_ONTOLOGY_UID,
            "tag_data": {
                "text": [
                    {"name": "analyzer", "val": "markit_vlm"},
                    {"name": "model", "val": model_name},
                ],
                "num": [
                    {"name": "frames_analyzed", "val": frames_analyzed},
                    {"name": "average_confidence", "val": round(avg_confidence, 4)},
                ],
            },
        }

        return tags

    @staticmethod
    def _aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple frame analyses using voting/averaging.

        For per-field confidence values, averages are computed across frames.

        Args:
            results: List of analysis dicts from multiple frames

        Returns:
            Aggregated analysis dict with per-field confidence values
        """
        if not results:
            return {}

        if len(results) == 1:
            return results[0]

        aggregated = {}

        # Helper to aggregate confidence values
        def aggregate_confidence(field_name: str, category_data: List[Dict]) -> float:
            conf_key = f"{field_name}_confidence"
            confs = [d.get(conf_key) for d in category_data if d.get(conf_key) is not None]
            return sum(confs) / len(confs) if confs else 0.5

        # Aggregate weather using majority voting
        weather_fields = [
            "precipitation", "precipitation_intensity", "particulates",
            "time_of_day", "sun_position", "cloud_cover"
        ]
        weather_rationale_fields = [f"{f}_rationale" for f in weather_fields]
        weather_data = [r.get("weather", {}) for r in results if r.get("weather")]

        if weather_data:
            aggregated["weather"] = {}
            for field in weather_fields:
                values = [d.get(field) for d in weather_data if d.get(field) is not None]
                if values:
                    aggregated["weather"][field] = max(set(values), key=values.count)
                    # Average the confidence
                    aggregated["weather"][f"{field}_confidence"] = aggregate_confidence(field, weather_data)

            # Visibility is numeric - average
            vis_values = [d.get("visibility_km") for d in weather_data if d.get("visibility_km") is not None]
            if vis_values:
                aggregated["weather"]["visibility_km"] = sum(vis_values) / len(vis_values)
                aggregated["weather"]["visibility_km_confidence"] = aggregate_confidence("visibility_km", weather_data)

            # Rationale fields: use first non-empty value
            for rationale_field in weather_rationale_fields:
                for d in weather_data:
                    if d.get(rationale_field):
                        aggregated["weather"][rationale_field] = d[rationale_field]
                        break

        # Aggregate road
        road_text_fields = [
            "drivable_area_type", "geometry_horizontal", "geometry_longitudinal",
            "surface_type", "surface_condition", "surface_quality"
        ]
        road_bool_fields = ["divided", "lane_markings_visible"]
        road_rationale_fields = [
            "drivable_area_type_rationale", "surface_type_rationale",
            "surface_condition_rationale", "surface_quality_rationale"
        ]
        road_data = [r.get("road", {}) for r in results if r.get("road")]

        if road_data:
            aggregated["road"] = {}
            for field in road_text_fields:
                values = [d.get(field) for d in road_data if d.get(field) is not None]
                if values:
                    aggregated["road"][field] = max(set(values), key=values.count)
                    aggregated["road"][f"{field}_confidence"] = aggregate_confidence(field, road_data)

            # Boolean fields - majority voting
            for field in road_bool_fields:
                values = [d.get(field) for d in road_data if d.get(field) is not None]
                if values:
                    aggregated["road"][field] = sum(values) > len(values) / 2
                    aggregated["road"][f"{field}_confidence"] = aggregate_confidence(field, road_data)

            # Lane count - average and round
            lane_values = [d.get("lane_count") for d in road_data if d.get("lane_count") is not None]
            if lane_values:
                aggregated["road"]["lane_count"] = round(sum(lane_values) / len(lane_values))
                aggregated["road"]["lane_count_confidence"] = aggregate_confidence("lane_count", road_data)

            # Rationale fields
            for rationale_field in road_rationale_fields:
                for d in road_data:
                    if d.get(rationale_field):
                        aggregated["road"][rationale_field] = d[rationale_field]
                        break

        # Aggregate traffic
        traffic_text_fields = ["density", "flow", "temporary_structures"]
        traffic_bool_fields = ["pedestrians_present", "cyclists_present", "special_vehicles_present"]
        traffic_rationale_fields = ["density_rationale", "flow_rationale", "temporary_structures_rationale"]
        traffic_data = [r.get("traffic", {}) for r in results if r.get("traffic")]

        if traffic_data:
            aggregated["traffic"] = {}
            for field in traffic_text_fields:
                values = [d.get(field) for d in traffic_data if d.get(field) is not None]
                if values:
                    aggregated["traffic"][field] = max(set(values), key=values.count)
                    aggregated["traffic"][f"{field}_confidence"] = aggregate_confidence(field, traffic_data)

            # Boolean fields - any True wins, with frame tracking for rationales
            for field in traffic_bool_fields:
                values = [d.get(field) for d in traffic_data if d.get(field) is not None]
                if values:
                    aggregated["traffic"][field] = any(values)
                    aggregated["traffic"][f"{field}_confidence"] = aggregate_confidence(field, traffic_data)

                    # Aggregate rationales with frame info for presence fields
                    rationale_field = f"{field}_rationale"
                    rationales_with_frames = []
                    for r in results:
                        traffic = r.get("traffic", {})
                        if traffic.get(field) and traffic.get(rationale_field):
                            frame_idx = r.get("_frame_idx")
                            rationale = traffic[rationale_field]
                            if frame_idx is not None:
                                rationales_with_frames.append(f"[frame {frame_idx}] {rationale}")
                            else:
                                rationales_with_frames.append(rationale)
                    if rationales_with_frames:
                        aggregated["traffic"][rationale_field] = "; ".join(rationales_with_frames)

            # Rationale fields (text fields)
            for rationale_field in traffic_rationale_fields:
                for d in traffic_data:
                    if d.get(rationale_field):
                        aggregated["traffic"][rationale_field] = d[rationale_field]
                        break

        # Aggregate junction
        junction_text_fields = ["type", "roundabout_type"]
        junction_bool_fields = ["present", "signalized", "pedestrian_crossing", "rail_crossing"]
        junction_data = [r.get("junction", {}) for r in results if r.get("junction")]

        if junction_data:
            aggregated["junction"] = {}
            for field in junction_text_fields:
                values = [d.get(field) for d in junction_data if d.get(field) is not None]
                if values:
                    aggregated["junction"][field] = max(set(values), key=values.count)
                    aggregated["junction"][f"{field}_confidence"] = aggregate_confidence(field, junction_data)

            # Boolean fields - any True wins
            for field in junction_bool_fields:
                values = [d.get(field) for d in junction_data if d.get(field) is not None]
                if values:
                    aggregated["junction"][field] = any(values)
                    aggregated["junction"][f"{field}_confidence"] = aggregate_confidence(field, junction_data)

        # Aggregate structures
        structures_bool_fields = ["bridge", "tunnel", "toll_plaza", "barriers_present"]
        structures_data = [r.get("structures", {}) for r in results if r.get("structures")]

        if structures_data:
            aggregated["structures"] = {}
            # Street lighting is text
            sl_values = [d.get("street_lighting") for d in structures_data if d.get("street_lighting") is not None]
            if sl_values:
                aggregated["structures"]["street_lighting"] = max(set(sl_values), key=sl_values.count)
                aggregated["structures"]["street_lighting_confidence"] = aggregate_confidence("street_lighting", structures_data)

            # Boolean fields - any True wins
            for field in structures_bool_fields:
                values = [d.get(field) for d in structures_data if d.get(field) is not None]
                if values:
                    aggregated["structures"][field] = any(values)
                    aggregated["structures"][f"{field}_confidence"] = aggregate_confidence(field, structures_data)

        # Copy through overall confidence
        confidences = [r.get("confidence") for r in results if "confidence" in r]
        if confidences:
            aggregated["confidence"] = sum(confidences) / len(confidences)

        return aggregated

    @staticmethod
    def _average_confidence(results: List[Dict[str, Any]]) -> float:
        """Calculate average confidence across all results.

        Args:
            results: List of analysis dicts

        Returns:
            Average confidence value (0.0 if no confidence values found)
        """
        confidences = []
        for r in results:
            if "confidence" in r and r["confidence"] is not None:
                confidences.append(float(r["confidence"]))

        return sum(confidences) / len(confidences) if confidences else 0.0
