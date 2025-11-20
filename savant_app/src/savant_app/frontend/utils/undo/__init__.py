"""Undo/redo toolkit for frontend annotation operations."""

from .manager import UndoRedoManager
from .commands import UndoableCommand
from .gateways import (
    UndoGatewayError,
    ControllerAnnotationGateway,
    ControllerFrameTagGateway,
    AnnotationGateway,
    FrameTagGateway,
    GatewayHolder,
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
    CreateObjectRelationshipCommand,
    DeleteBBoxCommand,
    InterpolateAnnotationsCommand,
    LinkObjectIdsCommand,
    RemoveFrameTagCommand,
    ResolveConfidenceCommand,
    UpdateBBoxGeometryCommand,
)

__all__ = [
    "GatewayHolder",
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
    "CreateObjectRelationshipCommand",
    "DeleteBBoxCommand",
    "InterpolateAnnotationsCommand",
    "LinkObjectIdsCommand",
    "RemoveFrameTagCommand",
    "ResolveConfidenceCommand",
    "UpdateBBoxGeometryCommand",
]
