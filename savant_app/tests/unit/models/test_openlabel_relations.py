import pytest
from savant_app.models.OpenLabel import (
    OpenLabel,
    RelationMetadata,
    RDFItem,
    FrameInterval,
)
from savant_app.models.OpenLabel import OpenLabelMetadata, ObjectMetadata, FrameObjects


@pytest.fixture
def sample_openlabel_with_relations():
    """Return an OpenLabel object with some sample data and relations."""
    relations = {
        "0": RelationMetadata(
            name="touching",
            type="spatial",
            ontology_uid="123",
            rdf_subjects=[RDFItem(type="object", uid="1")],
            rdf_objects=[RDFItem(type="object", uid="2")],
            frame_intervals=[FrameInterval(frame_start=0, frame_end=10)],
        ),
        "1": RelationMetadata(
            name="following",
            type="temporal",
            ontology_uid="456",
            rdf_subjects=[RDFItem(type="object", uid="2")],
            rdf_objects=[RDFItem(type="object", uid="3")],
            frame_intervals=[FrameInterval(frame_start=11, frame_end=20)],
        ),
        "2": RelationMetadata(
            name="passing",
            type="spatio-temporal",
            ontology_uid="789",
            rdf_subjects=[RDFItem(type="object", uid="1")],
            rdf_objects=[RDFItem(type="object", uid="3")],
            frame_intervals=[FrameInterval(frame_start=21, frame_end=30)],
        ),
    }

    openlabel = OpenLabel(
        metadata=OpenLabelMetadata(schema_version="1.0.0"),
        ontologies={},
        objects={
            "1": ObjectMetadata(name="car", type="vehicle"),
            "2": ObjectMetadata(name="person", type="pedestrian"),
            "3": ObjectMetadata(name="bike", type="vehicle"),
        },
        frames={
            "0": FrameObjects(objects={}),
        },
        relations=relations,
    )
    return openlabel


def test_get_object_relationships(sample_openlabel_with_relations):
    """Test getting all relationships for a given object_id."""
    openlabel = sample_openlabel_with_relations

    # Test for object "1"
    relations_1 = openlabel.get_object_relationships("1")
    assert len(relations_1) == 2
    relation_names_1 = sorted(
        [rel["relation_metadata"].name for rel in relations_1]
    )
    assert relation_names_1 == ["passing", "touching"]

    # Test for object "2"
    relations_2 = openlabel.get_object_relationships("2")
    assert len(relations_2) == 2
    relation_names_2 = sorted(
        [rel["relation_metadata"].name for rel in relations_2]
    )
    assert relation_names_2 == ["following", "touching"]

    # Test for object "3"
    relations_3 = openlabel.get_object_relationships("3")
    assert len(relations_3) == 2
    relation_names_3 = sorted(
        [rel["relation_metadata"].name for rel in relations_3]
    )
    assert relation_names_3 == ["following", "passing"]

    # Test for an object with no relations
    relations_4 = openlabel.get_object_relationships("4")
    assert len(relations_4) == 0


def test_delete_relationship(sample_openlabel_with_relations):
    """Test deleting a relationship by its ID."""
    openlabel = sample_openlabel_with_relations

    # Delete relation "1"
    assert openlabel.delete_relationship("1")
    assert "1" not in openlabel.relations
    assert len(openlabel.relations) == 2

    # Try to delete a non-existent relation
    assert not openlabel.delete_relationship("100")
    assert len(openlabel.relations) == 2

    # Delete another relation
    assert openlabel.delete_relationship("0")
    assert "0" not in openlabel.relations
    assert len(openlabel.relations) == 1

    # Delete the last relation
    assert openlabel.delete_relationship("2")
    assert "2" not in openlabel.relations
    assert len(openlabel.relations) == 0

    # Try to delete from empty relations
    assert not openlabel.delete_relationship("0")
