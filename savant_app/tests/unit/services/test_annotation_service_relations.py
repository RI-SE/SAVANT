import pytest
from unittest.mock import MagicMock
from savant_app.services.annotation_service import AnnotationService
from savant_app.models.OpenLabel import (
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

    relations = [
        RelationMetadata(
            name="touching",
            type="spatial",
            ontology_uid="123",
            rdf_subjects=[RDFItem(type="object", uid="1")],
            rdf_objects=[RDFItem(type="object", uid="2")],
            frame_intervals=[FrameInterval(frame_start=0, frame_end=10)],
        )
    ]

    mock_project_state.annotation_config.get_object_relationships.return_value = (
        relations
    )

    result = annotation_service.get_object_relationships("1")
    assert result == relations
    mock_project_state.annotation_config.get_object_relationships.assert_called_once_with(
        "1"
    )


def test_delete_relationship(mock_project_state):
    """Test deleting a relationship by its ID."""
    annotation_service = AnnotationService(mock_project_state)

    mock_project_state.annotation_config.delete_relationship.return_value = True

    result = annotation_service.delete_relationship("1")
    assert result is True
    mock_project_state.annotation_config.delete_relationship.assert_called_once_with(
        "1"
    )

    mock_project_state.annotation_config.delete_relationship.return_value = False
    result = annotation_service.delete_relationship("100")
    assert result is False
    mock_project_state.annotation_config.delete_relationship.assert_called_with("100")
