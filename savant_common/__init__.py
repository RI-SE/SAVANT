"""
savant_common - Shared utilities for SAVANT toolkit

This package contains common functionality used across multiple SAVANT tools:
- ontology: RDF/Turtle ontology parsing for SAVANT ontology integration

Used by:
- markit: Object detection and video processing
- trainit: YOLO model training
- utils: Various utility tools
"""

__version__ = '1.0.0'

from savant_common.ontology import (
    read_ontology_classes,
    create_class_map,
    get_classes_by_category,
    get_class_by_uid,
    get_class_by_label,
)

__all__ = [
    'read_ontology_classes',
    'create_class_map',
    'get_classes_by_category',
    'get_class_by_uid',
    'get_class_by_label',
]
