# savant_common

Shared utilities for SAVANT toolkit components.

## Overview

This package provides common functionality used across multiple SAVANT tools (markit, trainit, utils, etc.).

## Modules

### ontology.py

RDF/Turtle ontology parsing for SAVANT ontology integration.

**Functions:**
- `read_ontology_classes(ttl_path)` - Parse ontology file and extract all classes
- `create_class_map(ttl_path, filter_by_uid=None, filter_by_category=None)` - Create UID to label mapping
- `get_classes_by_category(ttl_path, top_level_category)` - Filter classes by category
- `get_class_by_uid(ttl_path, uid)` - Look up class by UID
- `get_class_by_label(ttl_path, label, case_sensitive=True)` - Look up class by label

### openlabel.py

OpenLabel JSON format reading and writing for SAVANT annotations.

**Reading (Pydantic models):**
- `OpenLabel` - Main model representing complete OpenLabel structure
  - `get_boxes_with_ids_for_frame(frame_idx)` - Extract bounding boxes for a frame
  - `get_frame_indices()` - Get sorted list of all frame indices
- `load_openlabel(json_path)` - Load and validate an OpenLabel JSON file
- `RotatedBBox` - Model for rotated bounding box (x, y, w, h, rotation)

**Writing:**
- `OpenLabelWriter` - Class for creating OpenLabel JSON files
  - `add_metadata(video_path, annotator, ...)` - Add metadata
  - `set_ontology(ontology_uri)` - Set ontology URI
  - `add_frame_objects(frame_idx, detections, class_map)` - Add frame detections
  - `save_to_file(output_path)` - Save to JSON file
- `DetectionData` - Dataclass for passing detection information
- `NumpyEncoder` - JSON encoder for NumPy types

**Utilities:**
- `normalize_angle_to_2pi_range(angle)` - Normalize angle to [0, 2π) range

## Usage Examples

### In markit

```python
from savant_common.ontology import create_class_map

# Load class mapping from ontology
class_map = create_class_map("../ontology/savant.ttl")
print(f"Loaded {len(class_map)} classes")
```

### In trainit

```python
from savant_common.ontology import read_ontology_classes

# Read all ontology classes
classes = read_ontology_classes("../ontology/savant.ttl")
for cls in classes:
    print(f"{cls['uid']}: {cls['label']}")
```

### In utils

```python
from savant_common.ontology import get_classes_by_category

# Get all vehicle classes
vehicles = get_classes_by_category(
    "../ontology/savant.ttl",
    "RoadUserVehicle"
)
```

### OpenLabel Reading

```python
from savant_common.openlabel import load_openlabel

# Load an OpenLabel annotation file
openlabel = load_openlabel("annotations.json")

# Get all frame indices
frames = openlabel.get_frame_indices()
print(f"Annotation has {len(frames)} frames")

# Get bounding boxes for a specific frame
boxes = openlabel.get_boxes_with_ids_for_frame(0)
for obj_id, obj_type, x, y, w, h, rot in boxes:
    print(f"Object {obj_id} ({obj_type}): center=({x}, {y}), size=({w}, {h}), rotation={rot}")
```

### OpenLabel Writing

```python
from savant_common.openlabel import OpenLabelWriter, DetectionData

# Create writer
writer = OpenLabelWriter("../schema/savant_openlabel_subset.schema.json")
writer.add_metadata("video.mp4", annotator="my_tool")
writer.set_ontology("http://example.org/ontology")

# Add detections for a frame
detections = [
    DetectionData(
        object_id=1,
        class_id=0,
        center=(100.0, 200.0),
        width=50.0,
        height=30.0,
        angle=0.5,
        confidence=0.95,
        source_engine="yolo"
    )
]
writer.add_frame_objects(0, detections, class_map={0: "car"})

# Save to file
writer.save_to_file("output.json")
```

## Installation

The package is automatically installed when you install the SAVANT toolkit:

```bash
# Install in development mode from repository root
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

## Dependencies

- `rdflib>=6.0.0` - RDF processing library (ontology.py)
- `pydantic>=2.0.0` - Data validation using Python type annotations (openlabel.py)
- `numpy` - Numerical operations (openlabel.py)

## Adding New Shared Utilities

To add new shared functionality:

1. Create a new module in `savant_common/`
2. Import and expose it in `savant_common/__init__.py`
3. Update this README with usage examples

Example:

```python
# savant_common/new_utility.py
def my_function():
    pass

# savant_common/__init__.py
from savant_common.new_utility import my_function

__all__ = [
    # ... existing exports
    'my_function',
]
```
