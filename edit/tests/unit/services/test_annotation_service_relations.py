import pytest
from unittest.mock import MagicMock
from edit.services.annotation_service import AnnotationService
from edit.models.OpenLabel import (
    OpenLabel,
    RelationMetadata,
    RDFItem,
    FrameInterval,
)


@pytest.fixture
def mock_project_state():
    """Return a mock ProjectState object."""
    project_state = MagicMock()
    project_state.annotation_config = MagicMock(spec=OpenLabel)
    return project_state


def test_get_object_relationships(mock_project_state):
    """Test getting all relationships for a given object_id."""
    annotation_service = AnnotationService(mock_project_state)

    relation_metadata = RelationMetadata(
        name="touching",
        type="spatial",
        ontology_uid="123",
        rdf_subjects=[RDFItem(type="object", uid="1")],
        rdf_objects=[RDFItem(type="object", uid="2")],
        frame_intervals=[FrameInterval(frame_start=0, frame_end=10)],
    )

    relations = [{"relation_id": "0", "relation_metadata": relation_metadata}]

    mock_project_state.annotation_config.get_object_relationships.return_value = (
        relations
    )

    result = annotation_service.get_object_relationships("1")
    expected_result = [
        {
            "id": "0",
            "subject": "1",
            "type": "spatial",
            "object": "2",
        }
    ]
    assert result == expected_result
    mock_project_state.annotation_config.get_object_relationships.assert_called_once_with(
        "1"
    )


def test_delete_relationship(mock_project_state):
    """Test deleting a relationship by its ID."""
    annotation_service = AnnotationService(mock_project_state)

    relation_metadata = RelationMetadata(
        name="touching",
        type="spatial",
        ontology_uid="123",
        rdf_subjects=[RDFItem(type="object", uid="1")],
        rdf_objects=[RDFItem(type="object", uid="2")],
        frame_intervals=[FrameInterval(frame_start=0, frame_end=10)],
    )

    mock_project_state.annotation_config.delete_relationship.return_value = {
        "id": "1",
        "metadata": relation_metadata,
    }

    result = annotation_service.delete_relationship("1")
    expected_result = {
        "id": "1",
        "subject": "1",
        "type": "spatial",
        "object": "2",
    }
    assert result == expected_result
    mock_project_state.annotation_config.delete_relationship.assert_called_once_with(
        "1"
    )
