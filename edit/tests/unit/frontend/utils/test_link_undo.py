"""Round-trip test of LinkObjectIdsCommand against the real model stack."""

from pathlib import Path
from types import SimpleNamespace

from edit.controllers.annotation_controller import AnnotationController
from edit.frontend.utils.undo import GatewayHolder, LinkObjectIdsCommand
from edit.frontend.utils.undo.gateways import ControllerAnnotationGateway
from edit.models.OpenLabel import (
    ObjectMetadataData,
    ObjectMetadataVecEntry,
    RDFItem,
    RelationMetadata,
)
from edit.services.annotation_service import AnnotationService
from edit.services.project_state import ProjectState

ASSET = Path(__file__).parents[3] / "assets" / "Kraklanda_short.json"


def _disjoint_pair(config):
    frames_by_obj = {}
    for key, frame in config.frames.items():
        for obj_id in frame.objects:
            frames_by_obj.setdefault(obj_id, set()).add(key)
    ids = sorted(frames_by_obj)
    return next(
        (a, b) for a in ids for b in ids
        if a != b and not frames_by_obj[a] & frames_by_obj[b]
    )


def test_link_moves_tags_and_relations_and_undo_restores_everything():
    state = ProjectState()
    state.load_openlabel_config(str(ASSET))
    config = state.annotation_config
    primary, secondary = _disjoint_pair(config)
    config.objects[secondary].object_data = ObjectMetadataData(
        vec=[ObjectMetadataVecEntry(name="occluded", val=[3])]
    )
    config.relations = {
        "0": RelationMetadata(
            name="Relation-0", type="follows", ontology_uid="0",
            rdf_subjects=[RDFItem(type="object", uid=secondary)],
            rdf_objects=[RDFItem(type="object", uid=primary)],
        )
    }
    before = config.model_dump()
    gateway = ControllerAnnotationGateway(
        annotation_controller=AnnotationController(AnnotationService(state)),
        project_state_controller=SimpleNamespace(project_state=state),
    )
    context = GatewayHolder(annotation_gateway=gateway)
    command = LinkObjectIdsCommand(primary, secondary)

    command.do(context)

    assert secondary not in config.objects
    assert config.relations["0"].rdf_subjects[0].uid == primary
    tags = {e.name: e.val for e in config.objects[primary].object_data.vec}
    assert tags["occluded"] == [3]

    command.undo(context)

    assert config.model_dump() == before
