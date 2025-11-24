import pytest
from unittest.mock import MagicMock
from savant_app.controllers.annotation_controller import AnnotationController
from savant_app.models.OpenLabel import RelationMetadata, RDFItem, FrameInterval


@pytest.fixture
def mock_annotation_service():
    """Return a mock AnnotationService object."""
    return MagicMock()


def test_get_object_relationship(mock_annotation_service):
    """Test getting all relationships for a given object_id."""
    annotation_controller = AnnotationController(mock_annotation_service)

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

    mock_annotation_service.get_object_relationships.return_value = relations

    result = annotation_controller.get_object_relationship("1")
    assert result == relations
    mock_annotation_service.get_object_relationships.assert_called_once_with("1")


def test_delete_relationship(mock_annotation_service):
    """Test deleting a relationship by its ID."""
    annotation_controller = AnnotationController(mock_annotation_service)

    mock_annotation_service.delete_relationship.return_value = True

    result = annotation_controller.delete_relationship("1")
    assert result is True
    mock_annotation_service.delete_relationship.assert_called_once_with("1")

    mock_annotation_service.delete_relationship.return_value = False
    result = annotation_controller.delete_relationship("100")
    assert result is False
    mock_annotation_service.delete_relationship.assert_called_with("100")
