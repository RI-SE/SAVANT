"""
response_parser - Parse VLM responses into structured data for OpenLABEL

Provides parsing of VLM text responses and conversion to OpenLABEL tags.
Schema aligned with BSI PAS-1883 ODD taxonomy.
"""

import json
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from savant_common.scenario_tags import (
    CONTEXT_FIELDS,
    FIELD_SPECS,
    NOTES_TAG,
    VLM_ANALYSIS_TAG,
    tag_class_for,
)

logger = logging.getLogger(__name__)

# Default annotator name for VLM-generated data
VLM_ANNOTATOR = "markit_vlm"

# Confidence used for context fields lacking a per-field confidence
DEFAULT_FIELD_CONFIDENCE = 0.5

RESPONSE_CATEGORIES = ("weather", "road", "traffic", "junction", "structures")
# Boolean schema fields without a FieldSpec (not mapped to tags)
_UNMAPPED_BOOL_FIELDS = (("junction", "present"),)
_INVALID = object()


def _most_common(values: List[Any]) -> Any:
    """Majority vote; ties go to the value seen first, so results are reproducible."""
    counts = Counter(values)
    return max(dict.fromkeys(values), key=counts.__getitem__)


def _coerce_bool(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return _INVALID


def _coerce_number(value: Any) -> Any:
    if isinstance(value, bool):
        return _INVALID
    if isinstance(value, (int, float)):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _INVALID
    return int(number) if number.is_integer() else number


def _coerce_choice(value: Any, choices: tuple) -> Any:
    if isinstance(value, str) and value.strip().lower() in choices:
        return value.strip().lower()
    return _INVALID


def _coerce_confidence(value: Any) -> Any:
    number = _coerce_number(value)
    return float(number) if number is not _INVALID and 0.0 <= number <= 1.0 else _INVALID


def _coerce_field(spec, value: Any) -> Any:
    """Return value coerced to the spec's type/vocabulary, or _INVALID."""
    if spec.is_boolean:
        return _coerce_bool(value)
    if spec.is_numeric:
        return _coerce_number(value)
    return _coerce_choice(value, spec.choices)


def _sanitize_category(category: str, data: Dict[str, Any]) -> None:
    """Coerce or drop (with a warning) values in one category dict, in place."""
    coercers = {
        spec.vlm_key: (lambda v, s=spec: _coerce_field(s, v))
        for spec in FIELD_SPECS.values() if spec.category == category
    }
    coercers.update({key: _coerce_bool for cat, key in _UNMAPPED_BOOL_FIELDS if cat == category})
    for key in list(data):
        value = data[key]
        if value is None:
            continue
        if key.endswith("_confidence"):
            coerced = _coerce_confidence(value)
        elif key.endswith("_rationale"):
            coerced = value if isinstance(value, str) else _INVALID
        elif key in coercers:
            coerced = coercers[key](value)
        else:
            continue
        if coerced is _INVALID:
            logger.warning(f"VLM response: dropping invalid {category}.{key}={value!r}")
            del data[key]
        else:
            data[key] = coerced


def sanitize_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce VLM output to the expected types in place; drop values that can't be."""
    for category in RESPONSE_CATEGORIES:
        if category not in data:
            continue
        if isinstance(data[category], dict):
            _sanitize_category(category, data[category])
        else:
            logger.warning(f"VLM response: dropping non-object {category}={data[category]!r}")
            del data[category]
    if "confidence" in data and _coerce_confidence(data["confidence"]) is _INVALID:
        del data["confidence"]
    elif "confidence" in data:
        data["confidence"] = _coerce_confidence(data["confidence"])
    if "notes" in data and not isinstance(data["notes"], str):
        del data["notes"]
    return data


# Numeric fields reported as whole numbers
_ROUNDED_NUMERIC_FIELDS = {"lane_count"}
# Booleans describing the road itself (vote); other booleans mean "seen at least once"
_MAJORITY_BOOLEAN_FIELDS = {"divided", "lane_markings_visible"}


def _aggregation_kind(key: str, spec) -> str:
    """How values of one field combine across frames: vote, mean, round, majority, any."""
    if spec is None:  # unmapped presence boolean such as junction.present
        return "any"
    if spec.is_numeric:
        return "round" if key in _ROUNDED_NUMERIC_FIELDS else "mean"
    if spec.is_boolean:
        return "majority" if key in _MAJORITY_BOOLEAN_FIELDS else "any"
    return "vote"


def _category_fields(category: str) -> List[tuple]:
    fields = [(spec.vlm_key, spec) for spec in FIELD_SPECS.values() if spec.category == category]
    fields += [(key, None) for cat, key in _UNMAPPED_BOOL_FIELDS if cat == category]
    return fields


def _combine(kind: str, values: List[Any]) -> Any:
    if kind == "vote":
        return _most_common(values)
    if kind == "mean":
        return sum(values) / len(values)
    if kind == "round":
        return round(sum(values) / len(values))
    if kind == "majority":
        return sum(values) > len(values) / 2
    return any(values)


def _combine_rationales(key: str, kind: str, entries: List[tuple]) -> Optional[str]:
    """Presence fields keep every frame's reason it was seen; others the first one given."""
    rationale_key = f"{key}_rationale"
    if kind == "any":
        parts = [
            f"[frame {frame}] {data[rationale_key]}" if frame is not None else data[rationale_key]
            for frame, data in entries
            if data.get(key) and data.get(rationale_key)
        ]
        return "; ".join(parts) or None
    return next((data[rationale_key] for _, data in entries if data.get(rationale_key)), None)


def _aggregate_category(category: str, entries: List[tuple]) -> Dict[str, Any]:
    """Aggregate one response category over (frame_idx, data) entries."""
    out: Dict[str, Any] = {}
    for key, spec in _category_fields(category):
        values = [data[key] for _, data in entries if data.get(key) is not None]
        if not values:
            continue
        kind = _aggregation_kind(key, spec)
        out[key] = _combine(kind, values)
        confidences = [
            data[f"{key}_confidence"] for _, data in entries
            if data.get(f"{key}_confidence") is not None
        ]
        out[f"{key}_confidence"] = (
            sum(confidences) / len(confidences) if confidences else DEFAULT_FIELD_CONFIDENCE
        )
        rationale = _combine_rationales(key, kind, entries)
        if rationale:
            out[f"{key}_rationale"] = rationale
    return out


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
        if not isinstance(data, dict) or not data:
            return None
        sanitize_response(data)

        # Validate expected structure (at least some keys should be present)
        found_keys = [key for key in RESPONSE_CATEGORIES if key in data]

        if len(found_keys) < 2:
            logger.warning(
                f"VLM response missing expected keys. Found: {list(data.keys())}"
            )
            # Still return partial data if some analysis is present
            if not found_keys:
                return None

        return data

    # Minimum number of agreeing samples in a segment. Segments with fewer are
    # considered potential VLM noise and will be merged with adjacent segments.
    MIN_CONTEXT_SAMPLES = 2

    @staticmethod
    def to_openlabel_contexts(
        analysis_results: List[Dict[str, Any]],
        frame_intervals: List[Dict[str, int]],
        ontology_uid: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Convert VLM analysis results to OpenLABEL contexts with frame intervals.

        One context per segment where a CONTEXT_FIELDS value stays constant, typed by the
        value's tag class. Segments whose value has no class (e.g. no precipitation) are
        omitted. Static fields (road, junction, structures) are only included in tags.

        Args:
            analysis_results: List of parsed VLM analysis dicts (must have _frame_idx)
            frame_intervals: Video frame range for extending final segment
            ontology_uid: Ontologies-section uid of the SAVANT scenario ontology

        Returns:
            OpenLABEL contexts dict with frame_intervals for each condition segment
        """
        if not analysis_results:
            return {}

        sorted_results = sorted(
            analysis_results, key=lambda x: x.get("_frame_idx", 0)
        )
        video_start = frame_intervals[0]["frame_start"] if frame_intervals else 0
        video_end = frame_intervals[0]["frame_end"] if frame_intervals else None

        contexts = []
        for field_name in CONTEXT_FIELDS:
            spec = FIELD_SPECS[field_name]
            segments = VLMResponseParser._build_segments(
                sorted_results, spec.category, spec.vlm_key, video_start, video_end
            )
            # Merge single-frame intervals with adjacent segments
            segments = VLMResponseParser._merge_short_intervals(
                segments, sorted_results, spec.category, spec.vlm_key
            )
            for segment in segments:
                context = VLMResponseParser._create_field_context(
                    field_name, segment["data"], segment["start"], segment["end"],
                    ontology_uid,
                )
                if context:
                    contexts.append(context)

        return {str(i): context for i, context in enumerate(contexts)}

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
            List of segment dicts with keys: start, end, key, data, samples

        Assumes a sampled value holds until the next sample, so segments tile the
        whole video without gaps (the first starts at video_start).
        """
        segments = []
        current_segment = None
        last_frame = video_start

        for result in sorted_results:
            frame_idx = result.get("_frame_idx", 0)
            data = result.get(category, {})
            key_value = data.get(key_field)

            if current_segment is None or key_value != current_segment["key"]:
                if current_segment is not None:
                    current_segment["end"] = frame_idx - 1
                    segments.append(current_segment)
                current_segment = {
                    "key": key_value,
                    "start": video_start if current_segment is None else frame_idx,
                    "end": None,
                    "data": data,
                    "samples": 0,
                }
            current_segment["samples"] += 1
            last_frame = frame_idx

        if current_segment is not None:
            current_segment["end"] = video_end if video_end is not None else last_frame
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

        Single-sample segments are likely VLM noise. They should be merged into the
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

        most_common = _most_common(values)

        # Find short intervals and merge them
        merged = []
        i = 0
        while i < len(segments):
            segment = segments[i]

            if segment["samples"] < VLMResponseParser.MIN_CONTEXT_SAMPLES:
                # Short interval - merge with adjacent using most common value
                if merged and merged[-1]["key"] == most_common:
                    # Extend previous segment to cover this one
                    merged[-1]["end"] = segment["end"]
                elif i < len(segments) - 1 and segments[i + 1]["key"] == most_common:
                    # Will be absorbed by next segment - skip this one
                    # Extend next segment's start backwards
                    segments[i + 1]["start"] = segment["start"]
                elif merged:
                    # No ideal match - just extend previous segment
                    merged[-1]["end"] = segment["end"]
                else:
                    # First segment and short - keep it but it will be merged next iteration
                    merged.append(segment)
            elif merged and merged[-1]["key"] == segment["key"]:
                # Same value on both sides of a merged-away segment: one context
                merged[-1]["end"] = segment["end"]
            else:
                merged.append(segment)

            i += 1

        return merged

    @staticmethod
    def _create_field_context(
        field_name: str,
        data: Dict[str, Any],
        frame_start: int,
        frame_end: int,
        ontology_uid: str,
    ) -> Optional[Dict[str, Any]]:
        """Create a context for one field over a frame interval, or None if its value has no class."""
        spec = FIELD_SPECS[field_name]
        type_name = tag_class_for(field_name, data.get(spec.vlm_key))
        if type_name is None:
            return None
        return {
            "name": field_name,
            "type": type_name,
            "ontology_uid": ontology_uid,
            "frame_intervals": [{"frame_start": frame_start, "frame_end": frame_end}],
            "context_data": VLMResponseParser._field_data(
                field_name, data, DEFAULT_FIELD_CONFIDENCE
            ),
        }

    @staticmethod
    def _field_data(
        field_name: str, data: Dict[str, Any], default_confidence: float
    ) -> Dict[str, Any]:
        """Build tag/context data for one field: value, rationale and per-field provenance."""
        spec = FIELD_SPECS[field_name]
        name = spec.data_name or field_name
        value_type = "num" if spec.is_numeric else "boolean" if spec.is_boolean else "text"
        result: Dict[str, List[Dict[str, Any]]] = {"text": [], "num": [], "boolean": []}
        result[value_type].append({"name": name, "val": data[spec.vlm_key]})
        result["text"].extend({"name": n, "val": v} for n, v in spec.extra_text)

        rationale = data.get(f"{spec.vlm_key}_rationale")
        if rationale:
            result["text"].append({"name": f"{name}_rationale", "val": rationale})

        confidence = data.get(f"{spec.vlm_key}_confidence", default_confidence)
        result["vec"] = [
            {"name": f"{name}_annotator", "val": [VLM_ANNOTATOR]},
            {"name": f"{name}_confidence", "val": [round(confidence, 4)]},
        ]
        return {k: v for k, v in result.items() if v}

    @staticmethod
    def _make_tag(
        type_name: str, tag_data: Dict[str, Any], ontology_uid: str
    ) -> Dict[str, Any]:
        """Create an OpenLABEL tag of the given class."""
        return {
            "type": type_name,
            "ontology_uid": ontology_uid,
            "tag_data": tag_data,
        }

    @staticmethod
    def to_openlabel_tags(
        analysis_results: List[Dict[str, Any]],
        model_name: str,
        frames_analyzed: int,
        ontology_uid: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Convert VLM analysis to OpenLABEL scenario tags.

        Each aggregated field value with a tag class becomes one tag typed by that class
        in the SAVANT scenario ontology (see savant_common.scenario_tags), carrying the
        source value and its provenance.

        Args:
            analysis_results: List of parsed VLM analysis dicts
            model_name: Name of the VLM model used
            frames_analyzed: Number of frames that were analyzed
            ontology_uid: Ontologies-section uid of the SAVANT scenario ontology

        Returns:
            OpenLABEL tags dict ready for insertion
        """
        aggregated = VLMResponseParser._aggregate_results(analysis_results)
        avg_confidence = VLMResponseParser._average_confidence(analysis_results)
        tags = []

        for field_name, spec in FIELD_SPECS.items():
            data = aggregated.get(spec.category) or {}
            if data.get(spec.vlm_key) is None:
                continue
            type_name = tag_class_for(field_name, data[spec.vlm_key])
            if type_name is None:
                continue
            tag_data = VLMResponseParser._field_data(field_name, data, avg_confidence)
            tags.append(VLMResponseParser._make_tag(type_name, tag_data, ontology_uid))

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
            tags.append(VLMResponseParser._make_tag(NOTES_TAG, tag_data, ontology_uid))

        tag_data = {
            "text": [
                {"name": "analyzer", "val": "markit_vlm"},
                {"name": "model", "val": model_name},
            ],
            "num": [
                {"name": "frames_analyzed", "val": frames_analyzed},
                {"name": "average_confidence", "val": round(avg_confidence, 4)},
            ],
        }
        tags.append(VLMResponseParser._make_tag(VLM_ANALYSIS_TAG, tag_data, ontology_uid))

        return {str(i): tag for i, tag in enumerate(tags)}

    @staticmethod
    def _aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine per-frame analyses into one, field by field (see _aggregation_kind)."""
        if not results:
            return {}
        if len(results) == 1:
            return results[0]

        aggregated: Dict[str, Any] = {}
        for category in RESPONSE_CATEGORIES:
            entries = [(r.get("_frame_idx"), r[category]) for r in results if r.get(category)]
            if entries:
                aggregated[category] = _aggregate_category(category, entries)

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
