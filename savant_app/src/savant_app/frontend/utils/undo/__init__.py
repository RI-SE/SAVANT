"""Undo/redo toolkit for frontend annotation operations."""

from .manager import UndoRedoContext, UndoRedoManager, UndoableCommand
from .gateways import (
    UndoGatewayError,
    ControllerAnnotationGateway,
    ControllerFrameTagGateway,
    AnnotationGateway,
    FrameTagGateway,
)
from .snapshots import (
    BBoxGeometrySnapshot,
    FrameObjectSnapshot,
    ObjectMetadataSnapshot,
    CreatedObjectSnapshot,
    FrameTagSnapshot,
)
from .commands import (
    AddFrameTagCommand,
    CascadeBBoxCommand,
    CompositeCommand,
    CreateExistingObjectBBoxCommand,
    CreateNewObjectBBoxCommand,
    DeleteBBoxCommand,
    InterpolateAnnotationsCommand,
    LinkObjectIdsCommand,
    RemoveFrameTagCommand,
    ResolveConfidenceCommand,
    UpdateBBoxGeometryCommand,
)

__all__ = [
    "UndoRedoContext",
    "UndoRedoManager",
    "UndoableCommand",
    "UndoGatewayError",
    "ControllerAnnotationGateway",
    "ControllerFrameTagGateway",
    "AnnotationGateway",
    "FrameTagGateway",
    "BBoxGeometrySnapshot",
    "FrameObjectSnapshot",
    "ObjectMetadataSnapshot",
    "CreatedObjectSnapshot",
    "FrameTagSnapshot",
    "AddFrameTagCommand",
    "CascadeBBoxCommand",
    "CompositeCommand",
    "CreateExistingObjectBBoxCommand",
    "CreateNewObjectBBoxCommand",
    "DeleteBBoxCommand",
    "InterpolateAnnotationsCommand",
    "LinkObjectIdsCommand",
    "RemoveFrameTagCommand",
    "ResolveConfidenceCommand",
    "UpdateBBoxGeometryCommand",
]
