from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional


class AnnotationMode(Enum):
    NONE = auto()  # No annotation mode
    EXISTING = auto()  # State when adding bbox to an existing object
    NEW = auto()  # State when adding bbox to a new object


@dataclass
class AnnotationState:
    object_type: str
    mode: AnnotationMode = field(
        default=AnnotationMode.NONE, metadata={"exclude": True}
    )  # Will not include annotation mode in dict conversions.
    object_id: Optional[str] = None
    coordinates: Optional[tuple[float, float, float, float, float]] = None
