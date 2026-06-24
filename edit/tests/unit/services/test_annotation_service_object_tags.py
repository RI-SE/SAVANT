# edit/tests/unit/services/test_annotation_service_object_tags.py
from unittest.mock import MagicMock

import pytest

from edit.models.OpenLabel import ObjectMetadataData, ObjectMetadataVecEntry
from edit.services.annotation_service import AnnotationService
from edit.services.exceptions import ObjectNotFoundError


def _make_service_with_object(object_id="obj1", vec=None):
    """Build a minimal AnnotationService stub with one object."""
    from edit.models.OpenLabel import ObjectMetadata

    metadata = ObjectMetadata(name="TestObject", type="vehicle")
    if vec is not None:
        metadata.object_data = ObjectMetadataData(vec=vec)

    openlabel = MagicMock()
    openlabel.objects = {object_id: metadata}

    project_state = MagicMock()
    project_state.annotation_config = openlabel

    service = object.__new__(AnnotationService)
    service.project_state = project_state
    service._frame_tag_lookup_cache = None
    service._frame_tag_lookup_source = None
    service._tag_cache_key = None
    service._tag_cache_vals = None
    service._bbox_cache_key = None
    service._bbox_cache_vals = None
    return service, metadata


class TestAddObjectTag:
    def test_creates_new_entry(self):
        service, metadata = _make_service_with_object()
        service.add_object_tag("obj1", "parked", 5)
        assert metadata.object_data is not None
        assert metadata.object_data.vec is not None
        assert len(metadata.object_data.vec) == 1
        entry = metadata.object_data.vec[0]
        assert entry.name == "parked"
        assert entry.val == [5]

    def test_appends_to_existing_entry(self):
        vec = [ObjectMetadataVecEntry(name="parked", val=[3, 7])]
        service, metadata = _make_service_with_object(vec=vec)
        service.add_object_tag("obj1", "parked", 5)
        assert metadata.object_data.vec[0].val == [3, 5, 7]

    def test_idempotent_when_frame_already_present(self):
        vec = [ObjectMetadataVecEntry(name="parked", val=[5])]
        service, metadata = _make_service_with_object(vec=vec)
        service.add_object_tag("obj1", "parked", 5)
        assert metadata.object_data.vec[0].val == [5]

    def test_creates_second_entry_for_different_tag(self):
        vec = [ObjectMetadataVecEntry(name="parked", val=[3])]
        service, metadata = _make_service_with_object(vec=vec)
        service.add_object_tag("obj1", "moving", 3)
        assert len(metadata.object_data.vec) == 2
        names = {e.name for e in metadata.object_data.vec}
        assert names == {"parked", "moving"}

    def test_raises_for_unknown_object(self):
        service, _ = _make_service_with_object()
        with pytest.raises(ObjectNotFoundError):
            service.add_object_tag("nonexistent", "parked", 5)

    def test_result_is_sorted(self):
        service, metadata = _make_service_with_object()
        service.add_object_tag("obj1", "parked", 10)
        service.add_object_tag("obj1", "parked", 2)
        service.add_object_tag("obj1", "parked", 6)
        assert metadata.object_data.vec[0].val == [2, 6, 10]


class TestRemoveObjectTag:
    def test_removes_frame_from_entry(self):
        vec = [ObjectMetadataVecEntry(name="parked", val=[3, 5, 7])]
        service, metadata = _make_service_with_object(vec=vec)
        result = service.remove_object_tag("obj1", "parked", 5)
        assert result is True
        assert metadata.object_data.vec[0].val == [3, 7]

    def test_removes_entry_when_val_empty(self):
        vec = [ObjectMetadataVecEntry(name="parked", val=[5])]
        service, metadata = _make_service_with_object(vec=vec)
        result = service.remove_object_tag("obj1", "parked", 5)
        assert result is True
        assert metadata.object_data.vec is None

    def test_returns_false_when_object_not_found(self):
        service, _ = _make_service_with_object()
        result = service.remove_object_tag("nonexistent", "parked", 5)
        assert result is False

    def test_returns_false_when_tag_not_found(self):
        vec = [ObjectMetadataVecEntry(name="parked", val=[5])]
        service, _ = _make_service_with_object(vec=vec)
        result = service.remove_object_tag("obj1", "no_such_tag", 5)
        assert result is False

    def test_returns_false_when_frame_not_in_entry(self):
        vec = [ObjectMetadataVecEntry(name="parked", val=[3, 7])]
        service, _ = _make_service_with_object(vec=vec)
        result = service.remove_object_tag("obj1", "parked", 5)
        assert result is False

    def test_returns_false_when_no_annotation_config(self):
        service = object.__new__(AnnotationService)
        service.project_state = MagicMock()
        service.project_state.annotation_config = None
        result = service.remove_object_tag("obj1", "parked", 5)
        assert result is False
