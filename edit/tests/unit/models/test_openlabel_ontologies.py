import pytest

from edit.models.OpenLabel import OntologyDetails, OpenLabel, OpenLabelMetadata

SAVANT_NS = "https://ri-se.github.io/SAVANT/ontology/savant#"
SCENARIO_NS = "https://ri-se.github.io/SAVANT/ontology/savant_scenario#"


def _openlabel(ontologies: dict) -> OpenLabel:
    return OpenLabel(
        metadata=OpenLabelMetadata(schema_version="1.0.0"),
        ontologies=ontologies,
        objects={},
        frames={},
    )


@pytest.mark.parametrize("stored", [SAVANT_NS, SAVANT_NS.rstrip("#"), SAVANT_NS[:-1] + "/"])
def test_finds_existing_namespace_key(stored):
    ol = _openlabel({"0": SCENARIO_NS, "1": stored})
    assert ol.get_or_add_ontology_uid(SAVANT_NS) == "1"
    assert len(ol.ontologies) == 2


def test_matches_ontology_details_uri():
    ol = _openlabel({"3": OntologyDetails(uri=SAVANT_NS)})
    assert ol.get_or_add_ontology_uid(SAVANT_NS) == "3"


def test_registers_missing_namespace_under_next_key():
    ol = _openlabel({"0": SCENARIO_NS, "4": "local.ttl"})
    assert ol.get_or_add_ontology_uid(SAVANT_NS) == "5"
    assert ol.ontologies["5"] == SAVANT_NS


def test_registers_first_namespace_as_zero():
    ol = _openlabel({})
    assert ol.get_or_add_ontology_uid(SAVANT_NS) == "0"
