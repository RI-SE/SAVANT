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

## Usage Examples

### In markit

```python
from savant_common.ontology import create_class_map

# Load class mapping from ontology
class_map = create_class_map("../Specification/savant_ontology_1.3.0.ttl")
print(f"Loaded {len(class_map)} classes")
```

### In trainit

```python
from savant_common.ontology import read_ontology_classes

# Read all ontology classes
classes = read_ontology_classes("../Specification/savant_ontology_1.3.0.ttl")
for cls in classes:
    print(f"{cls['uid']}: {cls['label']}")
```

### In utils

```python
from savant_common.ontology import get_classes_by_category

# Get all vehicle classes
vehicles = get_classes_by_category(
    "../Specification/savant_ontology_1.3.0.ttl",
    "RoadUserVehicle"
)
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

- `rdflib>=6.0.0` - RDF processing library

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
