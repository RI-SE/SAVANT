"""
response_parser - Parse VLM responses into structured data for OpenLABEL

Provides parsing of VLM text responses and conversion to OpenLABEL tags.
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
        expected_keys = ["weather", "road", "traffic", "scene"]
        found_keys = [key for key in expected_keys if key in data]

        if len(found_keys) < 2:
            logger.warning(
                f"VLM response missing expected keys. Found: {list(data.keys())}"
            )
            # Still return partial data if some analysis is present
            if not found_keys:
                return None

        return data

    @staticmethod
    def to_openlabel_contexts(
        analysis_results: List[Dict[str, Any]],
        frame_intervals: List[Dict[str, int]],
    ) -> Dict[str, Dict[str, Any]]:
        """Convert VLM analysis results to OpenLABEL contexts with frame intervals.

        Creates contexts that track when conditions change over time. A new context
        is only created when a condition differs from the previous analyzed frame.

        Args:
            analysis_results: List of parsed VLM analysis dicts (must have _frame_idx)
            frame_intervals: Video frame range for extending final segment

        Returns:
            OpenLABEL contexts dict with frame_intervals for each condition segment
        """
        if not analysis_results:
            return {}

        contexts = {}
        context_id = 0

        # Sort results by frame index
        sorted_results = sorted(
            analysis_results, key=lambda x: x.get("_frame_idx", 0)
        )

        # Get video frame range for final segment extension
        video_start = frame_intervals[0]["frame_start"] if frame_intervals else 0
        video_end = frame_intervals[0]["frame_end"] if frame_intervals else None

        # Track segments for each condition type
        # Each tracker: {"current": value, "start": frame, "data": full_data_dict}
        weather_tracker = {"current": None, "start": None, "data": None}
        road_tracker = {"current": None, "start": None, "data": None}
        traffic_tracker = {"current": None, "start": None, "data": None}
        risk_tracker = {"current": None, "start": None, "data": None}

        prev_frame = video_start

        for result in sorted_results:
            frame_idx = result.get("_frame_idx", 0)

            # Weather tracking - key on condition
            weather = result.get("weather", {})
            weather_key = weather.get("condition")
            if weather_key != weather_tracker["current"]:
                if weather_tracker["current"] is not None:
                    # Close previous weather segment
                    contexts[str(context_id)] = (
                        VLMResponseParser._create_weather_context(
                            weather_tracker["data"],
                            weather_tracker["start"],
                            prev_frame,
                        )
                    )
                    context_id += 1
                # Start new segment
                weather_tracker = {
                    "current": weather_key,
                    "start": frame_idx,
                    "data": weather,
                }

            # Road tracking - key on type + surface_condition
            road = result.get("road", {})
            road_key = (road.get("type"), road.get("surface_condition"))
            if road_key != road_tracker["current"]:
                if road_tracker["current"] is not None:
                    contexts[str(context_id)] = (
                        VLMResponseParser._create_road_context(
                            road_tracker["data"],
                            road_tracker["start"],
                            prev_frame,
                        )
                    )
                    context_id += 1
                road_tracker = {"current": road_key, "start": frame_idx, "data": road}

            # Traffic tracking - key on density
            traffic = result.get("traffic", {})
            traffic_key = traffic.get("density")
            if traffic_key != traffic_tracker["current"]:
                if traffic_tracker["current"] is not None:
                    contexts[str(context_id)] = (
                        VLMResponseParser._create_traffic_context(
                            traffic_tracker["data"],
                            traffic_tracker["start"],
                            prev_frame,
                        )
                    )
                    context_id += 1
                traffic_tracker = {
                    "current": traffic_key,
                    "start": frame_idx,
                    "data": traffic,
                }

            # Risk tracking - key on level
            risk = result.get("risk", {})
            risk_key = risk.get("level")
            if risk_key != risk_tracker["current"]:
                if risk_tracker["current"] is not None:
                    contexts[str(context_id)] = (
                        VLMResponseParser._create_risk_context(
                            risk_tracker["data"],
                            risk_tracker["start"],
                            prev_frame,
                        )
                    )
                    context_id += 1
                risk_tracker = {"current": risk_key, "start": frame_idx, "data": risk}

            prev_frame = frame_idx

        # Close final segments (extend to video end if available)
        final_frame = video_end if video_end is not None else prev_frame

        if weather_tracker["current"] is not None:
            contexts[str(context_id)] = VLMResponseParser._create_weather_context(
                weather_tracker["data"],
                weather_tracker["start"],
                final_frame,
            )
            context_id += 1

        if road_tracker["current"] is not None:
            contexts[str(context_id)] = VLMResponseParser._create_road_context(
                road_tracker["data"],
                road_tracker["start"],
                final_frame,
            )
            context_id += 1

        if traffic_tracker["current"] is not None:
            contexts[str(context_id)] = VLMResponseParser._create_traffic_context(
                traffic_tracker["data"],
                traffic_tracker["start"],
                final_frame,
            )
            context_id += 1

        if risk_tracker["current"] is not None:
            contexts[str(context_id)] = VLMResponseParser._create_risk_context(
                risk_tracker["data"],
                risk_tracker["start"],
                final_frame,
            )
            context_id += 1

        return contexts

    @staticmethod
    def _create_weather_context(
        weather: Dict[str, Any], frame_start: int, frame_end: int
    ) -> Dict[str, Any]:
        """Create a weather context for a frame interval."""
        context_data = {"text": []}

        if weather.get("condition"):
            context_data["text"].append(
                {"name": "condition", "val": weather["condition"]}
            )
        if weather.get("visibility"):
            context_data["text"].append(
                {"name": "visibility", "val": weather["visibility"]}
            )
        if weather.get("time_of_day"):
            context_data["text"].append(
                {"name": "time_of_day", "val": weather["time_of_day"]}
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
    def _create_road_context(
        road: Dict[str, Any], frame_start: int, frame_end: int
    ) -> Dict[str, Any]:
        """Create a road context for a frame interval."""
        context_data = {"text": [], "num": []}

        if road.get("type"):
            context_data["text"].append({"name": "road_type", "val": road["type"]})
        if road.get("surface_condition"):
            context_data["text"].append(
                {"name": "surface_condition", "val": road["surface_condition"]}
            )
        if road.get("lane_count") is not None:
            context_data["num"].append(
                {"name": "lane_count", "val": int(road["lane_count"])}
            )

        context_data = {k: v for k, v in context_data.items() if v}

        return {
            "name": "road_infrastructure",
            "type": "RoadContext",
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
        if "pedestrians_present" in traffic:
            context_data["boolean"].append(
                {"name": "pedestrians_present", "val": traffic["pedestrians_present"]}
            )
        if "cyclists_present" in traffic:
            context_data["boolean"].append(
                {"name": "cyclists_present", "val": traffic["cyclists_present"]}
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
    def _create_risk_context(
        risk: Dict[str, Any], frame_start: int, frame_end: int
    ) -> Dict[str, Any]:
        """Create a risk context for a frame interval."""
        context_data = {"text": [], "vec": []}

        if risk.get("level"):
            context_data["text"].append({"name": "risk_level", "val": risk["level"]})
        if risk.get("factors"):
            context_data["vec"].append({"name": "risk_factors", "val": risk["factors"]})

        context_data = {k: v for k, v in context_data.items() if v}

        return {
            "name": "risk_assessment",
            "type": "RiskContext",
            "ontology_uid": SCENARIO_ONTOLOGY_UID,
            "frame_intervals": [{"frame_start": frame_start, "frame_end": frame_end}],
            "context_data": context_data,
        }

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

        # Weather tag
        if "weather" in aggregated:
            weather = aggregated["weather"]
            tag_data = {"text": []}

            if "condition" in weather:
                tag_data["text"].append(
                    {"name": "condition", "val": weather["condition"]}
                )
            if "visibility" in weather:
                tag_data["text"].append(
                    {"name": "visibility", "val": weather["visibility"]}
                )
            if "time_of_day" in weather:
                tag_data["text"].append(
                    {"name": "time_of_day", "val": weather["time_of_day"]}
                )

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
            tag_data = {"text": [], "num": []}

            if "type" in road:
                tag_data["text"].append({"name": "road_type", "val": road["type"]})
            if "surface_condition" in road:
                tag_data["text"].append(
                    {"name": "surface_condition", "val": road["surface_condition"]}
                )
            if road.get("lane_count") is not None:
                tag_data["num"].append(
                    {"name": "lane_count", "val": int(road["lane_count"])}
                )

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
            tag_data = {"text": [], "boolean": []}

            if "density" in traffic:
                tag_data["text"].append(
                    {"name": "density", "val": traffic["density"]}
                )
            if "pedestrians_present" in traffic:
                tag_data["boolean"].append(
                    {"name": "pedestrians_present", "val": traffic["pedestrians_present"]}
                )
            if "cyclists_present" in traffic:
                tag_data["boolean"].append(
                    {"name": "cyclists_present", "val": traffic["cyclists_present"]}
                )

            tag_data = {k: v for k, v in tag_data.items() if v}

            if tag_data:
                tags[str(tag_id)] = {
                    "name": "traffic_conditions",
                    "type": "TrafficTag",
                    "ontology_uid": SCENARIO_ONTOLOGY_UID,
                    "tag_data": tag_data,
                }
                tag_id += 1

        # Risk tag
        if "risk" in aggregated:
            risk = aggregated["risk"]
            tag_data = {"text": [], "vec": []}

            if "level" in risk:
                tag_data["text"].append({"name": "risk_level", "val": risk["level"]})
            if "factors" in risk and risk["factors"]:
                tag_data["vec"].append(
                    {"name": "risk_factors", "val": risk["factors"]}
                )

            tag_data = {k: v for k, v in tag_data.items() if v}

            if tag_data:
                tags[str(tag_id)] = {
                    "name": "risk_assessment",
                    "type": "RiskTag",
                    "ontology_uid": SCENARIO_ONTOLOGY_UID,
                    "tag_data": tag_data,
                }
                tag_id += 1

        # Scene type tag
        if "scene" in aggregated:
            scene = aggregated["scene"]
            tag_data = {"text": []}

            if "type" in scene:
                tag_data["text"].append({"name": "scene_type", "val": scene["type"]})
            if "description" in scene:
                tag_data["text"].append(
                    {"name": "description", "val": scene["description"]}
                )

            tag_data = {k: v for k, v in tag_data.items() if v}

            if tag_data:
                tags[str(tag_id)] = {
                    "name": "scene_analysis",
                    "type": "SceneTypeTag",
                    "ontology_uid": SCENARIO_ONTOLOGY_UID,
                    "tag_data": tag_data,
                }
                tag_id += 1

        # VLM analysis metadata tag
        avg_confidence = VLMResponseParser._average_confidence(analysis_results)
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
            "condition": [],
            "visibility": [],
            "time_of_day": [],
            "confidence": [],
        }
        for r in results:
            weather = r.get("weather", {})
            for key in weather_values:
                if key in weather and weather[key] is not None:
                    weather_values[key].append(weather[key])

        if any(weather_values.values()):
            aggregated["weather"] = {}
            for key, values in weather_values.items():
                if values:
                    if key == "confidence":
                        aggregated["weather"][key] = sum(values) / len(values)
                    else:
                        # Majority voting for categorical
                        aggregated["weather"][key] = max(set(values), key=values.count)

        # Aggregate road
        road_values = {"type": [], "surface_condition": [], "lane_count": []}
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
                    else:
                        aggregated["road"][key] = max(set(values), key=values.count)

        # Aggregate traffic
        traffic_values = {
            "density": [],
            "pedestrians_present": [],
            "cyclists_present": [],
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
                    if key in ("pedestrians_present", "cyclists_present"):
                        # Any True wins for presence detection
                        aggregated["traffic"][key] = any(values)
                    else:
                        aggregated["traffic"][key] = max(set(values), key=values.count)

        # Aggregate risk
        risk_values = {"level": [], "factors": []}
        for r in results:
            risk = r.get("risk", {})
            if "level" in risk:
                risk_values["level"].append(risk["level"])
            if "factors" in risk:
                risk_values["factors"].extend(risk["factors"])

        if any(risk_values.values()):
            aggregated["risk"] = {}
            if risk_values["level"]:
                # Use highest risk level (most conservative)
                risk_order = ["low", "medium", "high", "critical"]
                aggregated["risk"]["level"] = max(
                    risk_values["level"],
                    key=lambda x: risk_order.index(x) if x in risk_order else -1,
                )
            if risk_values["factors"]:
                # Deduplicate and take most common factors
                factor_counts = {}
                for f in risk_values["factors"]:
                    factor_counts[f] = factor_counts.get(f, 0) + 1
                aggregated["risk"]["factors"] = sorted(
                    factor_counts.keys(), key=lambda x: -factor_counts[x]
                )[:5]  # Top 5 factors

        # Aggregate scene (use most common scene type, combine descriptions)
        scene_values = {"type": [], "description": []}
        for r in results:
            scene = r.get("scene", {})
            if "type" in scene:
                scene_values["type"].append(scene["type"])
            if "description" in scene:
                scene_values["description"].append(scene["description"])

        if any(scene_values.values()):
            aggregated["scene"] = {}
            if scene_values["type"]:
                aggregated["scene"]["type"] = max(
                    set(scene_values["type"]), key=scene_values["type"].count
                )
            if scene_values["description"]:
                # Use the most common description or first one
                aggregated["scene"]["description"] = scene_values["description"][0]

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
            # Also check nested confidence in weather
            if "weather" in r and "confidence" in r["weather"]:
                confidences.append(float(r["weather"]["confidence"]))

        return sum(confidences) / len(confidences) if confidences else 0.0
