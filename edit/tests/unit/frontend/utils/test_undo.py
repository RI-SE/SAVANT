from __future__ import annotations

from types import SimpleNamespace

import pytest

from edit.frontend.utils.undo import (
    BBoxGeometrySnapshot,
    CascadeBBoxCommand,
    CascadeDeltaBBoxCommand,
    DeleteBBoxCommand,
    FrameObjectSnapshot,
    GatewayHolder,
    UndoRedoManager,
)


class DummyCommand:
    description = "dummy"

    def __init__(self):
        self.do_calls = 0
        self.undo_calls = 0

    def do(self, context: GatewayHolder) -> None:  # pragma: no cover - context unused
        self.do_calls += 1

    def undo(self, context: GatewayHolder) -> None:  # pragma: no cover - context unused
        self.undo_calls += 1


class FakeDeleteGateway:
    def __init__(self):
        self.deleted = []
        self.restored = []
        self.snapshot = FrameObjectSnapshot(
            frame_number=3,
            object_id="obj-1",
            frame_object=object(),
        )

    def delete_bbox(self, frame_number: int, object_id: str):
        self.deleted.append((frame_number, object_id))
        return self.snapshot

    def restore_bbox(self, snapshot: FrameObjectSnapshot):
        self.restored.append(snapshot)


class FakeCascadeGateway:
    def __init__(self):
        self.geometries = {
            1: BBoxGeometrySnapshot(1.0, 2.0, 3.0, 4.0, 0.0),
            2: BBoxGeometrySnapshot(2.0, 3.0, 4.0, 5.0, 0.0),
            3: BBoxGeometrySnapshot(3.0, 4.0, 5.0, 6.0, 0.0),
        }
        self.cascade_calls = []
        self.applied = []

    def frames_for_object(self, object_id: str):  # pragma: no cover - trivial
        return sorted(self.geometries.keys())

    def capture_geometry(self, frame_number: int, object_id: str):
        geom = self.geometries[frame_number]
        return BBoxGeometrySnapshot(
            geom.center_x,
            geom.center_y,
            geom.width,
            geom.height,
            geom.rotation,
        )

    def cascade_bbox_edit(
        self,
        frame_start: int,
        frame_end: int | None,
        object_id: str,
        center_x: float | None,
        center_y: float | None,
        width: float | None,
        height: float | None,
        rotation: float | None,
        annotator: str,
    ):
        self.cascade_calls.append(
            (
                frame_start,
                frame_end,
                object_id,
                center_x,
                center_y,
                width,
                height,
                rotation,
                annotator,
            )
        )
        updated = []
        for frame in self.geometries:
            if frame < frame_start:
                continue
            if frame_end is not None and frame > frame_end:
                continue
            updated.append(frame)
            geom = self.geometries[frame]
            self.geometries[frame] = BBoxGeometrySnapshot(
                center_x if center_x is not None else geom.center_x + 0.5,
                center_y if center_y is not None else geom.center_y + 0.5,
                width or geom.width,
                height or geom.height,
                rotation if rotation is not None else geom.rotation,
            )
        return updated

    def apply_geometry(
        self,
        frame_number: int,
        object_id: str,
        geometry: BBoxGeometrySnapshot,
        annotator: str,
    ):
        self.applied.append((frame_number, geometry))
        self.geometries[frame_number] = geometry


@pytest.fixture()
def undo_context():
    # Commands under test only use annotation gateway
    return GatewayHolder(annotation_gateway=SimpleNamespace())


def test_undo_redo_manager_tracks_command_lifecycle(undo_context):
    manager = UndoRedoManager()
    command = DummyCommand()

    manager.execute(command, undo_context)

    assert manager.can_undo()
    assert not manager.can_redo()

    undone = manager.undo(undo_context)
    assert undone is command
    assert command.undo_calls == 1
    assert manager.can_redo()

    redone = manager.redo(undo_context)
    assert redone is command
    assert command.do_calls == 2  # initial execute + redo
    assert manager.can_undo()


def test_delete_bbox_command_restores_snapshot_on_undo():
    gateway = FakeDeleteGateway()
    context = GatewayHolder(annotation_gateway=gateway)
    command = DeleteBBoxCommand(frame_number=5, object_id="obj-9")

    command.do(context)

    assert gateway.deleted == [(5, "obj-9")]
    assert command._snapshot is gateway.snapshot  # internal bookkeeping

    command.undo(context)

    assert gateway.restored == [gateway.snapshot]

    command.undo(context)
    assert gateway.restored == [gateway.snapshot, gateway.snapshot]


def test_cascade_bbox_command_restores_original_geometry_after_undo():
    gateway = FakeCascadeGateway()
    context = GatewayHolder(annotation_gateway=gateway)
    command = CascadeBBoxCommand(
        object_id="obj-1",
        frame_start=1,
        frame_end=2,
        center_x=None,
        center_y=None,
        width=10.0,
        height=20.0,
        rotation=0.25,
        annotator="tester",
    )

    original = {frame: geom for frame, geom in gateway.geometries.items()}

    command.do(context)

    assert command.modified_frames == (1, 2)
    assert len(gateway.cascade_calls) == 1
    assert gateway.geometries[1].width == 10.0
    assert gateway.geometries[2].height == 20.0

    command.undo(context)

    assert gateway.geometries[1] == original[1]
    assert gateway.geometries[2] == original[2]

    command.do(context)  # redo without re-running cascade

    assert len(gateway.cascade_calls) == 1
    assert gateway.applied[-2][0] == 1
    assert gateway.applied[-1][0] == 2
    assert gateway.geometries[1].width == 10.0


def test_cascade_delta_bbox_command_applies_delta_on_do():
    gateway = FakeCascadeGateway()
    context = GatewayHolder(annotation_gateway=gateway)
    original = {f: BBoxGeometrySnapshot(g.center_x, g.center_y, g.width, g.height, g.rotation)
                for f, g in gateway.geometries.items()}

    command = CascadeDeltaBBoxCommand(
        object_id="obj-1",
        frame_start=1,
        frame_end=3,
        dcx=1.0,
        dcy=2.0,
        dw=0.0,
        dh=0.0,
        dtheta=0.0,
        annotator="tester",
    )

    command.do(context)

    # Each frame should have center shifted by (dcx, dcy)
    for f in (1, 2, 3):
        assert gateway.geometries[f].center_x == pytest.approx(original[f].center_x + 1.0)
        assert gateway.geometries[f].center_y == pytest.approx(original[f].center_y + 2.0)
        # Size unchanged
        assert gateway.geometries[f].width == pytest.approx(original[f].width)
        assert gateway.geometries[f].height == pytest.approx(original[f].height)


def test_cascade_delta_bbox_command_undo_restores_original_geometry():
    gateway = FakeCascadeGateway()
    context = GatewayHolder(annotation_gateway=gateway)
    original = {f: BBoxGeometrySnapshot(g.center_x, g.center_y, g.width, g.height, g.rotation)
                for f, g in gateway.geometries.items()}

    command = CascadeDeltaBBoxCommand(
        object_id="obj-1",
        frame_start=1,
        frame_end=3,
        dcx=5.0,
        dcy=5.0,
        dw=1.0,
        dh=1.0,
        dtheta=0.1,
        annotator="tester",
    )

    command.do(context)
    command.undo(context)

    for f in (1, 2, 3):
        assert gateway.geometries[f].center_x == pytest.approx(original[f].center_x)
        assert gateway.geometries[f].center_y == pytest.approx(original[f].center_y)
        assert gateway.geometries[f].width == pytest.approx(original[f].width)
        assert gateway.geometries[f].height == pytest.approx(original[f].height)
        assert gateway.geometries[f].rotation == pytest.approx(original[f].rotation)


def test_cascade_delta_bbox_command_redo_reapplies_after_geometry():
    gateway = FakeCascadeGateway()
    context = GatewayHolder(annotation_gateway=gateway)

    command = CascadeDeltaBBoxCommand(
        object_id="obj-1",
        frame_start=1,
        frame_end=None,
        dcx=2.0,
        dcy=0.0,
        dw=0.0,
        dh=0.0,
        dtheta=0.0,
        annotator="tester",
    )

    command.do(context)
    after_do = {f: BBoxGeometrySnapshot(g.center_x, g.center_y, g.width, g.height, g.rotation)
                for f, g in gateway.geometries.items()}

    command.undo(context)
    command.do(context)  # redo — should reapply _after without re-computing

    for f in gateway.geometries:
        assert gateway.geometries[f].center_x == pytest.approx(after_do[f].center_x)


def test_cascade_delta_bbox_command_respects_frame_range():
    gateway = FakeCascadeGateway()
    context = GatewayHolder(annotation_gateway=gateway)
    original_frame3 = BBoxGeometrySnapshot(
        gateway.geometries[3].center_x,
        gateway.geometries[3].center_y,
        gateway.geometries[3].width,
        gateway.geometries[3].height,
        gateway.geometries[3].rotation,
    )

    command = CascadeDeltaBBoxCommand(
        object_id="obj-1",
        frame_start=1,
        frame_end=2,  # frame 3 excluded
        dcx=99.0,
        dcy=0.0,
        dw=0.0,
        dh=0.0,
        dtheta=0.0,
        annotator="tester",
    )

    command.do(context)

    # Frame 3 must be untouched
    assert gateway.geometries[3].center_x == pytest.approx(original_frame3.center_x)


def test_cascade_delta_bbox_command_undo_before_do_is_noop():
    gateway = FakeCascadeGateway()
    context = GatewayHolder(annotation_gateway=gateway)
    original = {f: BBoxGeometrySnapshot(g.center_x, g.center_y, g.width, g.height, g.rotation)
                for f, g in gateway.geometries.items()}

    command = CascadeDeltaBBoxCommand(
        object_id="obj-1",
        frame_start=1,
        frame_end=None,
        dcx=10.0,
        dcy=10.0,
        dw=0.0,
        dh=0.0,
        dtheta=0.0,
        annotator="tester",
    )

    # Calling undo without a prior do should not raise and should not change anything
    command.undo(context)

    for f in gateway.geometries:
        assert gateway.geometries[f].center_x == pytest.approx(original[f].center_x)


class TestEditFrameTagCommand:
    """Tests for EditFrameTagCommand execute/undo round-trip."""

    def _make_context(self):
        from unittest.mock import MagicMock
        from edit.frontend.utils.undo.gateways import GatewayHolder, FrameTagGateway
        gateway = MagicMock(spec=FrameTagGateway)
        annotation_gateway = MagicMock()
        annotation_gateway.capture_geometry = MagicMock()
        holder = GatewayHolder(
            annotation_gateway=annotation_gateway,
            frame_tag_gateway=gateway,
        )
        return holder, gateway

    def test_do_calls_edit_frame_tag(self):
        from edit.frontend.utils.undo import (
            EditFrameTagCommand, EditFrameTagSnapshot, FrameTagSnapshot
        )
        old = FrameTagSnapshot("lanechange", 0, 10)
        new = FrameTagSnapshot("overtake", 0, 10)
        cmd = EditFrameTagCommand(EditFrameTagSnapshot(old=old, new=new))
        ctx, gateway = self._make_context()
        cmd.do(ctx)
        gateway.edit_frame_tag.assert_called_once_with(old, new)

    def test_undo_reverses_edit(self):
        from edit.frontend.utils.undo import (
            EditFrameTagCommand, EditFrameTagSnapshot, FrameTagSnapshot
        )
        old = FrameTagSnapshot("lanechange", 0, 10)
        new = FrameTagSnapshot("overtake", 0, 10)
        cmd = EditFrameTagCommand(EditFrameTagSnapshot(old=old, new=new))
        ctx, gateway = self._make_context()
        cmd.do(ctx)
        cmd.undo(ctx)
        assert gateway.edit_frame_tag.call_count == 2
        calls = gateway.edit_frame_tag.call_args_list
        assert calls[1].args == (new, old)

    def test_no_gateway_raises(self):
        from edit.frontend.utils.undo import (
            EditFrameTagCommand, EditFrameTagSnapshot, FrameTagSnapshot
        )
        from unittest.mock import MagicMock
        from edit.frontend.utils.undo.gateways import GatewayHolder
        old = FrameTagSnapshot("lanechange", 0, 10)
        new = FrameTagSnapshot("overtake", 0, 10)
        cmd = EditFrameTagCommand(EditFrameTagSnapshot(old=old, new=new))
        annotation_gateway = MagicMock()
        annotation_gateway.capture_geometry = MagicMock()
        ctx = GatewayHolder(annotation_gateway=annotation_gateway)
        with pytest.raises(RuntimeError, match="No frame tag gateway"):
            cmd.do(ctx)
