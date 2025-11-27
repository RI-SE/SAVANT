"""Undo/redo toolkit for frontend annotation operations."""

from .commands import (
    AddFrameTagCommand,
    CascadeBBoxCommand,
    CompositeCommand,
    CreateExistingObjectBBoxCommand,
    CreateNewObjectBBoxCommand,
    CreateObjectRelationshipCommand,
    DeleteBBoxCommand,
    DeleteRelationshipCommand,
    InterpolateAnnotationsCommand,
    LinkObjectIdsCommand,
    RemoveFrameTagCommand,
    ResolveConfidenceCommand,
    UndoableCommand,
    UpdateBBoxGeometryCommand,
)
from .gateways import (
    AnnotationGateway,
    ControllerAnnotationGateway,
    ControllerFrameTagGateway,
    FrameTagGateway,
    GatewayHolder,
    UndoGatewayError,
)
from .manager import UndoRedoManager
from .snapshots import (
    BBoxGeometrySnapshot,
    CreatedObjectSnapshot,
    FrameObjectSnapshot,
    FrameTagSnapshot,
    ObjectMetadataSnapshot,
    RelationshipSnapshot,
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
    "DeleteRelationshipCommand",
]
