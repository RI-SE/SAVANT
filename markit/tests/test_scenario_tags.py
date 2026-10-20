"""Tests for savant_common.scenario_tags and its consistency with the scenario ontology."""

import json
from pathlib import Path

import pytest
from rdflib import Graph, RDFS

from savant_common.scenario_tags import (
    FIELD_SPECS,
    NOTES_TAG,
    SCENARIO_ONTOLOGY_URI,
    VLM_ANALYSIS_TAG,
    get_or_add_ontology_uid,
    tag_class_for,
)

OPENLABEL_URI = "https://openlabel.asam.net/V1-0-0/ontologies/"
ONTOLOGY_DIR = Path(__file__).resolve().parents[2] / "ontology"
SCENARIO_TTL = ONTOLOGY_DIR / "savant_scenario.ttl"
SAVANT_TTL = ONTOLOGY_DIR / "savant.ttl"
PROMPTS_JSON = (
    Path(__file__).resolve().parents[1] / "markitlib" / "vlm" / "prompts" / "default_prompts.json"
)
# Schema fields deliberately not mapped to tags (implied by junction_type)
UNMAPPED_FIELDS = {("junction", "present")}
# Third-party file, gitignored (not redistributed); tests depending on it skip when absent
OPENLABEL_TTL = ONTOLOGY_DIR / "openlabel_ontology_scenario_tags.ttl"


def _local(uri, prefix):
    return str(uri)[len(prefix):] if str(uri).startswith(prefix) else None


def _used_classes():
    classes = {NOTES_TAG, VLM_ANALYSIS_TAG}
    for spec in FIELD_SPECS.values():
        classes.update(spec.classes.values())
        if spec.numeric_class:
            classes.add(spec.numeric_class)
    return classes


def _defined(graph, prefix):
    return {_local(s, prefix) for s in graph.subjects() if _local(s, prefix)}


def _parents(graph, prefix):
    """Map class name to parent class name within one namespace."""
    return {
        _local(s, prefix): _local(o, prefix)
        for s, o in graph.subject_objects(RDFS.subClassOf)
        if _local(s, prefix) and _local(o, prefix)
    }


def _mirrored(graph):
    """Map scenario resource name to the OpenLABEL name it declares via rdfs:seeAlso."""
    return {
        _local(s, SCENARIO_ONTOLOGY_URI): _local(o, OPENLABEL_URI)
        for s, o in graph.subject_objects(RDFS.seeAlso)
        if _local(o, OPENLABEL_URI)
    }


@pytest.fixture(scope="module")
def scenario_graph():
    return Graph().parse(SCENARIO_TTL, format="turtle")


@pytest.fixture(scope="module")
def openlabel_graph():
    if not OPENLABEL_TTL.exists():
        pytest.skip("OpenLABEL ontology file not available")
    return Graph().parse(OPENLABEL_TTL, format="turtle")


def _uids(graph):
    return [int(o) for p, o in graph.predicate_objects() if str(p).endswith("#uid")]


class TestScenarioOntology:
    def test_uids_unique_and_disjoint_from_savant(self, scenario_graph):
        """Scenario uids start at 1000; savant.ttl uids are used downstream and fixed."""
        scenario_uids = _uids(scenario_graph)
        savant_uids = set(_uids(Graph().parse(SAVANT_TTL, format="turtle")))
        assert len(scenario_uids) == len(set(scenario_uids))
        assert min(scenario_uids) >= 1000
        assert set(scenario_uids) & savant_uids == set()

    def test_used_classes_defined(self, scenario_graph):
        assert _used_classes() - _defined(scenario_graph, SCENARIO_ONTOLOGY_URI) == set()

    def test_only_used_classes_and_ancestors_defined(self, scenario_graph):
        parents = _parents(scenario_graph, SCENARIO_ONTOLOGY_URI)
        # Agent type values referenced by trafficAgentTypeValue
        needed = _used_classes() | {"HumanPedestrian", "HumanCyclist"}
        frontier = set(needed)
        while frontier:
            frontier = {parents[c] for c in frontier if c in parents} - needed
            needed |= frontier
        classes = {
            _local(s, SCENARIO_ONTOLOGY_URI)
            for s in scenario_graph.subjects(None, RDFS.Class)
        }
        assert classes - needed == set()

    def test_extension_classes_extend_mirrored_hierarchy(self, scenario_graph):
        """Every non-mirrored class subclasses a scenario class, so OpenLABEL can replace the mirror."""
        parents = _parents(scenario_graph, SCENARIO_ONTOLOGY_URI)
        mirrored = _mirrored(scenario_graph)
        for name in _used_classes() - set(mirrored):
            assert name in parents, name


class TestOpenLabelMirror:
    def test_mirrored_names_identical(self, scenario_graph):
        assert all(k == v for k, v in _mirrored(scenario_graph).items())

    def test_mirrored_names_exist_in_openlabel(self, scenario_graph, openlabel_graph):
        missing = set(_mirrored(scenario_graph).values()) - _defined(openlabel_graph, OPENLABEL_URI)
        assert missing == set()

    def test_mirrored_hierarchy_matches_openlabel(self, scenario_graph, openlabel_graph):
        ours = _parents(scenario_graph, SCENARIO_ONTOLOGY_URI)
        theirs = _parents(openlabel_graph, OPENLABEL_URI)
        for name in _mirrored(scenario_graph):
            assert ours.get(name) == theirs.get(name), name


class TestFieldSpecs:
    def test_mapped_values_are_valid_choices(self):
        for name, spec in FIELD_SPECS.items():
            assert set(spec.classes) <= set(spec.choices), name

    def test_tag_class_for(self):
        assert tag_class_for("precipitation", "rain") == "WeatherRain"
        assert tag_class_for("precipitation", "none") is None
        assert tag_class_for("bridge", False) is None
        assert tag_class_for("lane_count", 3) == "LaneSpecificationLaneCount"
        assert tag_class_for("unknown_field", "x") is None


def _schema_fields():
    """Yield (prompt, category, field, schema) for value fields in each prompt's response schema."""
    prompts = json.loads(PROMPTS_JSON.read_text())
    for prompt_name, prompt in prompts.items():
        if not isinstance(prompt, dict) or "response_schema" not in prompt:
            continue
        for category, cat_schema in prompt["response_schema"]["properties"].items():
            for field_name, schema in cat_schema.get("properties", {}).items():
                if not field_name.endswith(("_confidence", "_rationale")):
                    yield prompt_name, category, field_name, schema


SPECS_BY_VLM_KEY = {(spec.category, spec.vlm_key): spec for spec in FIELD_SPECS.values()}


class TestFieldSpecsMatchPrompts:
    """FIELD_SPECS must stay in step with the VLM prompt vocabulary."""

    @pytest.mark.parametrize(
        "prompt_name,category,field_name,schema",
        list(_schema_fields()),
        ids=lambda v: v if isinstance(v, str) else "",
    )
    def test_schema_field_matches_spec(self, prompt_name, category, field_name, schema):
        if (category, field_name) in UNMAPPED_FIELDS:
            return
        spec = SPECS_BY_VLM_KEY.get((category, field_name))
        assert spec is not None, f"{prompt_name}: {category}.{field_name} has no FieldSpec"
        if "enum" in schema:
            assert set(spec.choices) == set(schema["enum"])
        elif schema.get("type") == "boolean":
            assert spec.is_boolean
        else:
            assert spec.is_numeric

    def test_every_spec_in_comprehensive_prompt(self):
        schema_keys = {
            (category, field_name)
            for prompt_name, category, field_name, _ in _schema_fields()
            if prompt_name == "comprehensive"
        }
        assert set(SPECS_BY_VLM_KEY) - schema_keys == set()


class TestGetOrAddOntologyUid:
    def test_matches_ignoring_trailing_separator(self):
        ontologies = {"0": "https://ri-se.github.io/SAVANT/ontology/savant#",
                      "1": SCENARIO_ONTOLOGY_URI.rstrip("#")}
        assert get_or_add_ontology_uid(ontologies, SCENARIO_ONTOLOGY_URI) == "1"

    def test_matches_dict_entry_uri(self):
        ontologies = {"2": {"uri": SCENARIO_ONTOLOGY_URI}}
        assert get_or_add_ontology_uid(ontologies, SCENARIO_ONTOLOGY_URI) == "2"

    def test_adds_under_next_key(self):
        ontologies = {"0": "a", "3": "b"}
        assert get_or_add_ontology_uid(ontologies, SCENARIO_ONTOLOGY_URI) == "4"
        assert ontologies["4"] == SCENARIO_ONTOLOGY_URI
