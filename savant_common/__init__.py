"""
savant_common - Shared utilities for SAVANT toolkit

This package contains common functionality used across multiple SAVANT tools:
- ontology: RDF/Turtle ontology parsing for SAVANT ontology integration
- openlabel: OpenLabel JSON format reading and writing

Used by:
- markit: Object detection and video processing
- trainit: YOLO model training
- savant_app: Annotation UI
- utils: Various utility tools
"""

__version__ = '1.1.0'

from savant_common.ontology import (
    read_ontology_classes,
    create_class_map,
    get_classes_by_category,
    get_class_by_uid,
    get_class_by_label,
)

from savant_common.openlabel import (
    # Reading
    OpenLabel,
    RotatedBBox,
    FrameObjects,
    FrameLevelObject,
    ObjectMetadata,
    load_openlabel,
    # Writing
    OpenLabelWriter,
    DetectionData,
    NumpyEncoder,
    # Utilities
    normalize_angle_to_2pi_range,
)

__all__ = [
    # Ontology
    'read_ontology_classes',
    'create_class_map',
    'get_classes_by_category',
    'get_class_by_uid',
    'get_class_by_label',
    # OpenLabel Reading
    'OpenLabel',
    'RotatedBBox',
    'FrameObjects',
    'FrameLevelObject',
    'ObjectMetadata',
    'load_openlabel',
    # OpenLabel Writing
    'OpenLabelWriter',
    'DetectionData',
    'NumpyEncoder',
    # Utilities
    'normalize_angle_to_2pi_range',
]
