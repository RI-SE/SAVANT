from __future__ import annotations

from types import SimpleNamespace

import pytest

from savant_app.frontend.utils.undo import (
    BBoxGeometrySnapshot,
    CascadeBBoxCommand,
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
