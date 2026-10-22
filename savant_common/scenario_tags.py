"""
scenario_tags - Mapping of VLM scene fields to SAVANT scenario ontology tag classes

Each VLM field value is expressed as one tag whose type is a class in
ontology/savant_scenario.ttl. Classes mirroring ASAM OpenLABEL scenario tagging classes
use the same names and hierarchy, so the ontology can be swapped for OpenLABEL 1:1;
SAVANT extension classes subclass them where OpenLABEL lacks a value.
Values without any class (e.g. "none", "unknown", False for presence fields) produce no tag.

Used by markit (tag creation) and edit (re-typing tags when values are edited).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

SCENARIO_ONTOLOGY_URI = "https://ri-se.github.io/SAVANT/ontology/savant_scenario#"


@dataclass(frozen=True)
class FieldSpec:
    """How one VLM response field maps to tag classes.

    For categorical/boolean fields, `classes` maps values to classes and `choices` lists
    all values the VLM may return. Numeric fields have a fixed `numeric_class` and store
    the value under `data_name` (a property name in the scenario ontology).
    """

    category: str
    vlm_key: str
    classes: Mapping[Any, str] = field(default_factory=dict)
    choices: Tuple[Any, ...] = ()
    numeric_class: Optional[str] = None
    data_name: Optional[str] = None
    extra_text: Tuple[Tuple[str, str], ...] = ()

    @property
    def is_numeric(self) -> bool:
        return self.numeric_class is not None

    @property
    def is_boolean(self) -> bool:
        return self.choices == (True, False)


_BOOL = (True, False)

# Keyed by the field name stored in tag_data; names are unique across categories.
FIELD_SPECS: Dict[str, FieldSpec] = {
    # --- Environment: weather, particulates, illumination ---
    "precipitation": FieldSpec(
        "weather", "precipitation",
        {"rain": "WeatherRain", "snow": "WeatherSnow"},
        ("none", "rain", "snow", "unknown"),
    ),
    "precipitation_intensity": FieldSpec(
        "weather", "precipitation_intensity",
        {
            "light": "PrecipitationLight",
            "moderate": "PrecipitationModerate",
            "heavy": "PrecipitationHeavy",
        },
        ("none", "light", "moderate", "heavy", "unknown"),
    ),
    "particulates": FieldSpec(
        "weather", "particulates",
        {
            "fog": "ParticulatesFog",
            "mist": "ParticulatesMist",
            "haze": "ParticulatesHaze",
            "dust": "ParticulatesDust",
            "smoke": "ParticulatesSmoke",
        },
        ("none", "fog", "mist", "haze", "dust", "smoke", "unknown"),
    ),
    "visibility_km": FieldSpec(
        "weather", "visibility_km",
        numeric_class="EnvironmentVisibility",
        data_name="environmentVisibilityValue",
    ),
    "time_of_day": FieldSpec(
        "weather", "time_of_day",
        {
            "day": "IlluminationDay",
            "night": "LowLightNight",
            "dawn": "LowLightDawn",
            "dusk": "LowLightDusk",
        },
        ("day", "night", "dawn", "dusk"),
    ),
    "sun_position": FieldSpec(
        "weather", "sun_position",
        {
            "front": "SunPositionFront",
            "left": "SunPositionLeft",
            "right": "SunPositionRight",
            "behind": "SunPositionBehind",
        },
        ("front", "left", "right", "behind", "overcast", "not_applicable"),
    ),
    "cloud_cover": FieldSpec(
        "weather", "cloud_cover",
        {
            "clear": "CloudinessClear",
            "partly_cloudy": "CloudinessPartlyCloudy",
            "overcast": "CloudinessOvercast",
        },
        ("clear", "partly_cloudy", "overcast", "unknown"),
    ),
    # --- Scenery: drivable area ---
    "drivable_area_type": FieldSpec(
        "road", "drivable_area_type",
        {
            "motorway": "RoadTypeMotorway",
            "motorway_atm": "MotorwayManaged",
            "radial": "RoadTypeRadial",
            "distributor": "RoadTypeDistributor",
            "minor": "RoadTypeMinor",
            "slip_road": "RoadTypeSlip",
            "parking": "RoadTypeParking",
            "shared_space": "RoadTypeShared",
        },
        ("motorway", "motorway_atm", "radial", "distributor", "minor",
         "slip_road", "parking", "shared_space", "unknown"),
    ),
    "geometry_horizontal": FieldSpec(
        "road", "geometry_horizontal",
        {"straight": "HorizontalStraights", "curved": "HorizontalCurves"},
        ("straight", "curved", "unknown"),
    ),
    "geometry_longitudinal": FieldSpec(
        "road", "geometry_longitudinal",
        {
            "level": "LongitudinalLevelPlane",
            "upslope": "LongitudinalUpSlope",
            "downslope": "LongitudinalDownSlope",
        },
        ("level", "upslope", "downslope", "unknown"),
    ),
    "divided": FieldSpec(
        "road", "divided",
        {True: "TransverseDivided", False: "TransverseUndivided"},
        _BOOL,
    ),
    "surface_type": FieldSpec(
        "road", "surface_type",
        {
            "asphalt": "SurfaceTypeAsphalt",
            "concrete": "SurfaceTypeConcrete",
            "segmented": "SurfaceTypeSegmented",
            "loose": "SurfaceTypeLoose",
        },
        ("asphalt", "concrete", "segmented", "loose", "unknown"),
    ),
    "surface_condition": FieldSpec(
        "road", "surface_condition",
        {
            "dry": "SurfaceConditionDry",
            "wet": "SurfaceConditionWet",
            "icy": "SurfaceConditionIcy",
            "snow_covered": "SurfaceConditionSnow",
            "flooded": "SurfaceConditionFlooded",
        },
        ("dry", "wet", "icy", "snow_covered", "flooded", "unknown"),
    ),
    "surface_quality": FieldSpec(
        "road", "surface_quality",
        {
            "cracks": "SurfaceFeatureCrack",
            "potholes": "SurfaceFeaturePothole",
            "ruts": "SurfaceFeatureRut",
        },
        ("good", "cracks", "potholes", "ruts", "unknown"),
    ),
    "lane_count": FieldSpec(
        "road", "lane_count",
        numeric_class="LaneSpecificationLaneCount",
        data_name="laneSpecificationLaneCountValue",
    ),
    "lane_markings_visible": FieldSpec(
        "road", "lane_markings_visible", {True: "LaneSpecificationMarking"}, _BOOL
    ),
    "barriers_present": FieldSpec(
        "structures", "barriers_present", {True: "TransverseBarriers"}, _BOOL
    ),
    # --- Scenery: junctions ---
    "junction_type": FieldSpec(
        "junction", "type",
        {
            "t_junction": "IntersectionTJunction",
            "y_junction": "IntersectionYJunction",
            "crossroads": "IntersectionCrossroad",
            "roundabout": "JunctionRoundabout",
            "grade_separated": "IntersectionGradeSeperated",  # sic, OpenLABEL spelling
        },
        ("none", "t_junction", "y_junction", "crossroads", "roundabout",
         "grade_separated", "unknown"),
    ),
    "roundabout_type": FieldSpec(
        "junction", "roundabout_type",
        {
            "normal": "RoundaboutNormal",
            "compact": "RoundaboutCompact",
            "mini": "RoundaboutMini",
            "double": "RoundaboutDouble",
            "large": "RoundaboutLarge",
        },
        ("normal", "compact", "mini", "double", "large", "not_applicable"),
    ),
    "signalized": FieldSpec(
        "junction", "signalized", {True: "JunctionSignalized"}, _BOOL
    ),
    # --- Scenery: special, fixed and temporary structures ---
    "pedestrian_crossing": FieldSpec(
        "junction", "pedestrian_crossing",
        {True: "SpecialStructurePedestrianCrossing"}, _BOOL,
    ),
    "rail_crossing": FieldSpec(
        "junction", "rail_crossing", {True: "SpecialStructureRailCrossing"}, _BOOL
    ),
    "bridge": FieldSpec(
        "structures", "bridge", {True: "SpecialStructureBridge"}, _BOOL
    ),
    "tunnel": FieldSpec(
        "structures", "tunnel", {True: "SpecialStructureTunnel"}, _BOOL
    ),
    "toll_plaza": FieldSpec(
        "structures", "toll_plaza", {True: "SpecialStructureTollPlaza"}, _BOOL
    ),
    "street_lighting": FieldSpec(
        "structures", "street_lighting",
        {"present": "FixedStructureStreetlight"},
        ("none", "present", "unknown"),
    ),
    "temporary_structures": FieldSpec(
        "traffic", "temporary_structures",
        {
            "construction": "TemporaryStructureConstruction",
            "roadworks": "TemporaryStructureRoadWorks",
            "refuse_collection": "TemporaryStructureRefuseCollection",
        },
        ("none", "construction", "roadworks", "refuse_collection", "unknown"),
    ),
    # --- Dynamic elements: traffic ---
    "density": FieldSpec(
        "traffic", "density",
        {
            "empty": "TrafficDensityEmpty",
            "sparse": "TrafficDensitySparse",
            "moderate": "TrafficDensityModerate",
            "dense": "TrafficDensityDense",
            "congested": "TrafficDensityCongested",
        },
        ("empty", "sparse", "moderate", "dense", "congested"),
    ),
    "flow": FieldSpec(
        "traffic", "flow",
        {
            "free_flow": "TrafficFlowFree",
            "stable": "TrafficFlowStable",
            "unstable": "TrafficFlowUnstable",
            "forced": "TrafficFlowForced",
        },
        ("free_flow", "stable", "unstable", "forced", "unknown"),
    ),
    "pedestrians_present": FieldSpec(
        "traffic", "pedestrians_present", {True: "TrafficAgentType"}, _BOOL,
        extra_text=(("trafficAgentTypeValue", "HumanPedestrian"),),
    ),
    "cyclists_present": FieldSpec(
        "traffic", "cyclists_present", {True: "TrafficAgentType"}, _BOOL,
        extra_text=(("trafficAgentTypeValue", "HumanCyclist"),),
    ),
    "special_vehicles_present": FieldSpec(
        "traffic", "special_vehicles_present", {True: "TrafficSpecialVehicle"}, _BOOL
    ),
}

# Fields tracked over time as frame-interval contexts
CONTEXT_FIELDS = ("precipitation", "density")

# SAVANT-only tag types (not field-derived)
NOTES_TAG = "NotesTag"
VLM_ANALYSIS_TAG = "VLMAnalysisTag"
# Per-field VLM sample statistics (scenario-uncertainty SPEC P3.5), num entries next to <field>_confidence
N_SAMPLES_SUFFIX = "_n_samples"
AGREEMENT_SUFFIX = "_agreement"
SAMPLE_STAT_SUFFIXES = (N_SAMPLES_SUFFIX, AGREEMENT_SUFFIX)


def tag_class_for(field_name: str, value: Any) -> Optional[str]:
    """Return the tag class for a field value, or None if the value yields no tag."""
    spec = FIELD_SPECS.get(field_name)
    if spec is None:
        return None
    if spec.is_numeric:
        return spec.numeric_class if value is not None else None
    if not isinstance(value, (str, bool)):
        return None
    return spec.classes.get(value)


def find_ontology_uid(ontologies: Mapping[str, Any], uri: str) -> Optional[str]:
    """Return the ontologies-section key whose URI matches uri, ignoring a trailing '#' or '/'."""

    def _normalize(value: str) -> str:
        return value.strip().rstrip("#/")

    target = _normalize(uri)
    for key, entry in ontologies.items():
        if isinstance(entry, dict):
            entry = entry.get("uri", "")
        entry_uri = entry if isinstance(entry, str) else getattr(entry, "uri", "")
        if _normalize(entry_uri) == target:
            return key
    return None


def get_or_add_ontology_uid(ontologies: Dict[str, Any], uri: str) -> str:
    """Return the key for uri in an ontologies section, registering it under the next free integer key."""
    key = find_ontology_uid(ontologies, uri)
    if key is not None:
        return key
    numeric_keys = [int(k) for k in ontologies if str(k).isdigit()]
    key = str(max(numeric_keys) + 1) if numeric_keys else "0"
    ontologies[key] = uri
    return key
