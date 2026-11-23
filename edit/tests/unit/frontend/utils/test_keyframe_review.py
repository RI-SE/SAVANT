"""Key-frame review: navigation and undo of ConfirmFrameCommand against the real model stack."""

from pathlib import Path
from types import SimpleNamespace

from edit.controllers.annotation_controller import AnnotationController
from edit.frontend.utils.keyframe_review import next_keyframe
from edit.frontend.utils.undo import ConfirmFrameCommand, GatewayHolder
from edit.frontend.utils.undo.gateways import ControllerAnnotationGateway
from edit.services.annotation_service import AnnotationService
from edit.services.project_state import ProjectState

ASSET = Path(__file__).parents[3] / "assets" / "Kraklanda_short.json"


def test_next_keyframe():
    assert next_keyframe(0, 30, +1, 100) == 30
    assert next_keyframe(29, 30, +1, 100) == 30
    assert next_keyframe(30, 30, +1, 100) == 60
    assert next_keyframe(30, 30, -1, 100) == 0
    assert next_keyframe(31, 30, -1, 100) == 30
    assert next_keyframe(0, 30, -1, 100) is None
    assert next_keyframe(95, 30, +1, 100) is None


def test_confirm_frame_undo_redo():
    state = ProjectState()
    state.load_openlabel_config(str(ASSET))
    config = state.annotation_config
    frame = int(sorted(config.frames, key=int)[0])
    before = config.model_dump()
    gateway = ControllerAnnotationGateway(
        annotation_controller=AnnotationController(AnnotationService(state)),
        project_state_controller=SimpleNamespace(project_state=state),
    )
    context = GatewayHolder(annotation_gateway=gateway)
    command = ConfirmFrameCommand(frame_number=frame, annotator="alice")

    command.do(context)
    after = config.model_dump()
    assert after["frames"][str(frame)]["frame_properties"]["review"]["annotator"] == "alice"

    command.undo(context)
    assert config.model_dump() == before

    command.do(context)
    assert config.model_dump() == after
