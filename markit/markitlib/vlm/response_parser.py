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
SCENARIO_ONTOLOGY_URI = "http://github.com/RI-SE/SAVANT/scenario-ontology#"


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
        context_data = {"text": [], "num": []}

        if weather.get("precipitation"):
            context_data["text"].append(
                {"name": "precipitation", "val": weather["precipitation"]}
            )
        if weather.get("precipitation_intensity"):
            context_data["text"].append(
                {"name": "precipitation_intensity", "val": weather["precipitation_intensity"]}
            )
        if weather.get("particulates"):
            context_data["text"].append(
                {"name": "particulates", "val": weather["particulates"]}
            )
        if weather.get("time_of_day"):
            context_data["text"].append(
                {"name": "time_of_day", "val": weather["time_of_day"]}
            )
        if weather.get("sun_position"):
            context_data["text"].append(
                {"name": "sun_position", "val": weather["sun_position"]}
            )
        if weather.get("cloud_cover"):
            context_data["text"].append(
                {"name": "cloud_cover", "val": weather["cloud_cover"]}
            )
        if weather.get("visibility_km") is not None:
            context_data["num"].append(
                {"name": "visibility_km", "val": float(weather["visibility_km"])}
            )

        # Rationale fields (if present)
        if weather.get("precipitation_rationale"):
            context_data["text"].append(
                {"name": "precipitation_rationale", "val": weather["precipitation_rationale"]}
            )
        if weather.get("precipitation_intensity_rationale"):
            context_data["text"].append(
                {"name": "precipitation_intensity_rationale", "val": weather["precipitation_intensity_rationale"]}
            )
        if weather.get("particulates_rationale"):
            context_data["text"].append(
                {"name": "particulates_rationale", "val": weather["particulates_rationale"]}
            )
        if weather.get("time_of_day_rationale"):
            context_data["text"].append(
                {"name": "time_of_day_rationale", "val": weather["time_of_day_rationale"]}
            )
        if weather.get("cloud_cover_rationale"):
            context_data["text"].append(
                {"name": "cloud_cover_rationale", "val": weather["cloud_cover_rationale"]}
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
        context_data = {"text": [], "boolean": []}

        if traffic.get("density"):
            context_data["text"].append({"name": "density", "val": traffic["density"]})
        if traffic.get("flow"):
            context_data["text"].append({"name": "flow", "val": traffic["flow"]})
        if traffic.get("temporary_structures"):
            context_data["text"].append(
                {"name": "temporary_structures", "val": traffic["temporary_structures"]}
            )
        if "pedestrians_present" in traffic:
            context_data["boolean"].append(
                {"name": "pedestrians_present", "val": traffic["pedestrians_present"]}
            )
        if "cyclists_present" in traffic:
            context_data["boolean"].append(
                {"name": "cyclists_present", "val": traffic["cyclists_present"]}
            )
        if "special_vehicles_present" in traffic:
            context_data["boolean"].append(
                {"name": "special_vehicles_present", "val": traffic["special_vehicles_present"]}
            )

        context_data = {k: v for k, v in context_data.items() if v}

        return {
            "name": "traffic_conditions",
            "type": "TrafficContext",
            "ontology_uid": SCENARIO_ONTOLOGY_UID,
            "frame_intervals": [{"frame_start": frame_start, "frame_end": frame_end}],
            "context_data": context_data,
        }

    @staticmethod
    def _add_provenance_fields(tag_data: Dict[str, Any], confidence: float) -> Dict[str, Any]:
        """Add annotator and confidence provenance fields to tag_data using vec format.

        Uses vec format (list-based) to support multi-annotator tracking. When a human
        edits a VLM tag, their annotator ID and confidence (1.0) are prepended to the
        lists, preserving the original VLM values.

        Args:
            tag_data: The tag_data dict to augment
            confidence: The aggregated confidence value from VLM analysis

        Returns:
            The tag_data dict with provenance fields added
        """
        if "vec" not in tag_data:
            tag_data["vec"] = []
        tag_data["vec"].append({"name": "annotator", "val": ["markit_vlm"]})
        tag_data["vec"].append({"name": "confidence", "val": [round(confidence, 4)]})

        return tag_data

    @staticmethod
    def to_openlabel_tags(
        analysis_results: List[Dict[str, Any]],
        model_name: str,
        frames_analyzed: int,
    ) -> Dict[str, Dict[str, Any]]:
        """Convert VLM analysis to OpenLABEL tags for scenario-level metadata.

        Per OpenLABEL spec, tags are for scenario categorization and organization.

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
            tag_data = {"text": [], "num": []}

            if "precipitation" in weather:
                tag_data["text"].append(
                    {"name": "precipitation", "val": weather["precipitation"]}
                )
            if "precipitation_intensity" in weather:
                tag_data["text"].append(
                    {"name": "precipitation_intensity", "val": weather["precipitation_intensity"]}
                )
            if "particulates" in weather:
                tag_data["text"].append(
                    {"name": "particulates", "val": weather["particulates"]}
                )
            if "time_of_day" in weather:
                tag_data["text"].append(
                    {"name": "time_of_day", "val": weather["time_of_day"]}
                )
            if "sun_position" in weather:
                tag_data["text"].append(
                    {"name": "sun_position", "val": weather["sun_position"]}
                )
            if "cloud_cover" in weather:
                tag_data["text"].append(
                    {"name": "cloud_cover", "val": weather["cloud_cover"]}
                )
            if weather.get("visibility_km") is not None:
                tag_data["num"].append(
                    {"name": "visibility_km", "val": float(weather["visibility_km"])}
                )

            # Rationale fields (if present)
            if weather.get("precipitation_rationale"):
                tag_data["text"].append(
                    {"name": "precipitation_rationale", "val": weather["precipitation_rationale"]}
                )
            if weather.get("precipitation_intensity_rationale"):
                tag_data["text"].append(
                    {"name": "precipitation_intensity_rationale", "val": weather["precipitation_intensity_rationale"]}
                )
            if weather.get("particulates_rationale"):
                tag_data["text"].append(
                    {"name": "particulates_rationale", "val": weather["particulates_rationale"]}
                )
            if weather.get("time_of_day_rationale"):
                tag_data["text"].append(
                    {"name": "time_of_day_rationale", "val": weather["time_of_day_rationale"]}
                )
            if weather.get("cloud_cover_rationale"):
                tag_data["text"].append(
                    {"name": "cloud_cover_rationale", "val": weather["cloud_cover_rationale"]}
                )

            tag_data = {k: v for k, v in tag_data.items() if v}

            if tag_data:
                VLMResponseParser._add_provenance_fields(tag_data, avg_confidence)
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
            tag_data = {"text": [], "num": [], "boolean": []}

            if "drivable_area_type" in road:
                tag_data["text"].append(
                    {"name": "drivable_area_type", "val": road["drivable_area_type"]}
                )
            if "geometry_horizontal" in road:
                tag_data["text"].append(
                    {"name": "geometry_horizontal", "val": road["geometry_horizontal"]}
                )
            if "geometry_longitudinal" in road:
                tag_data["text"].append(
                    {"name": "geometry_longitudinal", "val": road["geometry_longitudinal"]}
                )
            if "surface_type" in road:
                tag_data["text"].append(
                    {"name": "surface_type", "val": road["surface_type"]}
                )
            if "surface_condition" in road:
                tag_data["text"].append(
                    {"name": "surface_condition", "val": road["surface_condition"]}
                )
            if "surface_quality" in road:
                tag_data["text"].append(
                    {"name": "surface_quality", "val": road["surface_quality"]}
                )
            if road.get("lane_count") is not None:
                tag_data["num"].append(
                    {"name": "lane_count", "val": int(road["lane_count"])}
                )
            if "divided" in road:
                tag_data["boolean"].append(
                    {"name": "divided", "val": road["divided"]}
                )
            if "lane_markings_visible" in road:
                tag_data["boolean"].append(
                    {"name": "lane_markings_visible", "val": road["lane_markings_visible"]}
                )

            tag_data = {k: v for k, v in tag_data.items() if v}

            if tag_data:
                VLMResponseParser._add_provenance_fields(tag_data, avg_confidence)
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
            tag_data = {"text": [], "boolean": []}

            if "density" in traffic:
                tag_data["text"].append(
                    {"name": "density", "val": traffic["density"]}
                )
            if "flow" in traffic:
                tag_data["text"].append(
                    {"name": "flow", "val": traffic["flow"]}
                )
            if "temporary_structures" in traffic:
                tag_data["text"].append(
                    {"name": "temporary_structures", "val": traffic["temporary_structures"]}
                )
            if "pedestrians_present" in traffic:
                tag_data["boolean"].append(
                    {"name": "pedestrians_present", "val": traffic["pedestrians_present"]}
                )
            if "cyclists_present" in traffic:
                tag_data["boolean"].append(
                    {"name": "cyclists_present", "val": traffic["cyclists_present"]}
                )
            if "special_vehicles_present" in traffic:
                tag_data["boolean"].append(
                    {"name": "special_vehicles_present", "val": traffic["special_vehicles_present"]}
                )

            tag_data = {k: v for k, v in tag_data.items() if v}

            if tag_data:
                VLMResponseParser._add_provenance_fields(tag_data, avg_confidence)
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
            tag_data = {"text": [], "boolean": []}

            if "type" in junction:
                tag_data["text"].append(
                    {"name": "junction_type", "val": junction["type"]}
                )
            if "roundabout_type" in junction:
                tag_data["text"].append(
                    {"name": "roundabout_type", "val": junction["roundabout_type"]}
                )
            if "present" in junction:
                tag_data["boolean"].append(
                    {"name": "junction_present", "val": junction["present"]}
                )
            if "signalized" in junction:
                tag_data["boolean"].append(
                    {"name": "signalized", "val": junction["signalized"]}
                )
            if "pedestrian_crossing" in junction:
                tag_data["boolean"].append(
                    {"name": "pedestrian_crossing", "val": junction["pedestrian_crossing"]}
                )
            if "rail_crossing" in junction:
                tag_data["boolean"].append(
                    {"name": "rail_crossing", "val": junction["rail_crossing"]}
                )

            tag_data = {k: v for k, v in tag_data.items() if v}

            if tag_data:
                VLMResponseParser._add_provenance_fields(tag_data, avg_confidence)
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
            tag_data = {"text": [], "boolean": []}

            if "street_lighting" in structures:
                tag_data["text"].append(
                    {"name": "street_lighting", "val": structures["street_lighting"]}
                )
            if "bridge" in structures:
                tag_data["boolean"].append(
                    {"name": "bridge", "val": structures["bridge"]}
                )
            if "tunnel" in structures:
                tag_data["boolean"].append(
                    {"name": "tunnel", "val": structures["tunnel"]}
                )
            if "toll_plaza" in structures:
                tag_data["boolean"].append(
                    {"name": "toll_plaza", "val": structures["toll_plaza"]}
                )
            if "barriers_present" in structures:
                tag_data["boolean"].append(
                    {"name": "barriers_present", "val": structures["barriers_present"]}
                )

            tag_data = {k: v for k, v in tag_data.items() if v}

            if tag_data:
                VLMResponseParser._add_provenance_fields(tag_data, avg_confidence)
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
            tag_data = {"text": [{"name": "notes", "val": first_note}]}
            VLMResponseParser._add_provenance_fields(tag_data, avg_confidence)
            tags[str(tag_id)] = {
                "name": "scene_notes",
                "type": "NotesTag",
                "ontology_uid": SCENARIO_ONTOLOGY_UID,
                "tag_data": tag_data,
            }
            tag_id += 1

        # VLM analysis metadata tag (already has analyzer and average_confidence)
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

        Args:
            results: List of analysis dicts from multiple frames

        Returns:
            Aggregated analysis dict
        """
        if not results:
            return {}

        if len(results) == 1:
            return results[0]

        aggregated = {}

        # Aggregate weather using majority voting
        weather_values = {
            "precipitation": [],
            "precipitation_intensity": [],
            "particulates": [],
            "visibility_km": [],
            "time_of_day": [],
            "sun_position": [],
            "cloud_cover": [],
        }
        # Rationale fields - use first non-empty value (not majority voting)
        weather_rationale_fields = [
            "precipitation_rationale",
            "precipitation_intensity_rationale",
            "particulates_rationale",
            "time_of_day_rationale",
            "cloud_cover_rationale",
        ]
        for r in results:
            weather = r.get("weather", {})
            for key in weather_values:
                if key in weather and weather[key] is not None:
                    weather_values[key].append(weather[key])

        if any(weather_values.values()):
            aggregated["weather"] = {}
            for key, values in weather_values.items():
                if values:
                    if key == "visibility_km":
                        # Average for numeric
                        aggregated["weather"][key] = sum(values) / len(values)
                    else:
                        # Majority voting for categorical
                        aggregated["weather"][key] = max(set(values), key=values.count)

            # Rationale fields: use first non-empty value (rationales don't aggregate well)
            for rationale_field in weather_rationale_fields:
                for r in results:
                    rationale = r.get("weather", {}).get(rationale_field)
                    if rationale:
                        aggregated["weather"][rationale_field] = rationale
                        break

        # Aggregate road
        road_values = {
            "drivable_area_type": [],
            "geometry_horizontal": [],
            "geometry_longitudinal": [],
            "divided": [],
            "surface_type": [],
            "surface_condition": [],
            "surface_quality": [],
            "lane_count": [],
            "lane_markings_visible": [],
        }
        for r in results:
            road = r.get("road", {})
            for key in road_values:
                if key in road and road[key] is not None:
                    road_values[key].append(road[key])

        if any(road_values.values()):
            aggregated["road"] = {}
            for key, values in road_values.items():
                if values:
                    if key == "lane_count":
                        # Average for numeric, round to int
                        aggregated["road"][key] = round(sum(values) / len(values))
                    elif key in ("divided", "lane_markings_visible"):
                        # Majority for booleans
                        aggregated["road"][key] = sum(values) > len(values) / 2
                    else:
                        aggregated["road"][key] = max(set(values), key=values.count)

        # Aggregate traffic
        traffic_values = {
            "density": [],
            "flow": [],
            "pedestrians_present": [],
            "cyclists_present": [],
            "special_vehicles_present": [],
            "temporary_structures": [],
        }
        for r in results:
            traffic = r.get("traffic", {})
            for key in traffic_values:
                if key in traffic and traffic[key] is not None:
                    traffic_values[key].append(traffic[key])

        if any(traffic_values.values()):
            aggregated["traffic"] = {}
            for key, values in traffic_values.items():
                if values:
                    if key in ("pedestrians_present", "cyclists_present", "special_vehicles_present"):
                        # Any True wins for presence detection
                        aggregated["traffic"][key] = any(values)
                    else:
                        aggregated["traffic"][key] = max(set(values), key=values.count)

        # Aggregate junction
        junction_values = {
            "present": [],
            "type": [],
            "roundabout_type": [],
            "signalized": [],
            "pedestrian_crossing": [],
            "rail_crossing": [],
        }
        for r in results:
            junction = r.get("junction", {})
            for key in junction_values:
                if key in junction and junction[key] is not None:
                    junction_values[key].append(junction[key])

        if any(junction_values.values()):
            aggregated["junction"] = {}
            for key, values in junction_values.items():
                if values:
                    if key in ("present", "signalized", "pedestrian_crossing", "rail_crossing"):
                        # Any True wins for presence detection
                        aggregated["junction"][key] = any(values)
                    else:
                        aggregated["junction"][key] = max(set(values), key=values.count)

        # Aggregate structures
        structures_values = {
            "bridge": [],
            "tunnel": [],
            "toll_plaza": [],
            "barriers_present": [],
            "street_lighting": [],
        }
        for r in results:
            structures = r.get("structures", {})
            for key in structures_values:
                if key in structures and structures[key] is not None:
                    structures_values[key].append(structures[key])

        if any(structures_values.values()):
            aggregated["structures"] = {}
            for key, values in structures_values.items():
                if values:
                    if key in ("bridge", "tunnel", "toll_plaza", "barriers_present"):
                        # Any True wins for presence detection
                        aggregated["structures"][key] = any(values)
                    else:
                        aggregated["structures"][key] = max(set(values), key=values.count)

        # Copy through confidence
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
