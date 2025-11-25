# SAVANT Utils

Utility scripts for working with SAVANT ontology and datasets.

## Overview

This directory contains command-line utilities for ontology inspection and dataset manipulation:

- **print_ontology.py** - Read and display SAVANT ontology classes
- **remap_classes_to_ontology.py** - Remap YOLO class IDs to SAVANT ontology UIDs

## Installation

These utilities are installed as part of the SAVANT package:

```bash
# From repository root
uv sync --dev

# The scripts are available as CLI commands
print-ontology --help
remap-classes-to-ontology --help
```

## Scripts

### print_ontology.py

Reads a SAVANT ontology file (.ttl) and displays the classes in various formats.

**Usage:**

```bash
# Basic usage - list all classes
print-ontology ../Specification/savant_ontology_1.3.0.ttl

# Show detailed information including parent classes
print-ontology ../Specification/savant_ontology_1.3.0.ttl --detailed

# Filter by top-level category
print-ontology ../Specification/savant_ontology_1.3.0.ttl --top-level DynamicObject
```

**Arguments:**

- `ttl_file` (required) - Path to the Turtle ontology file (.ttl)
- `--detailed`, `-d` - Show detailed information including parent and top-level classes
- `--top-level`, `-t` - Filter to show only classes with specific top-level class

**Examples:**

```bash
# List all vehicle classes
print-ontology ../Specification/savant_ontology_1.3.0.ttl --top-level RoadUserVehicle

# Get detailed view of all classes
print-ontology ../Specification/savant_ontology_1.3.0.ttl -d
```

**Output:**

```
Found 41 ontology class(es):

UID  Label                Parent Class
---- -------------------- ----------------------
110  car                  RoadUserVehicle
111  van                  RoadUserVehicle
112  truck                RoadUserVehicle
113  bus                  RoadUserVehicle
...
```

---

### remap_classes_to_ontology.py

Remaps YOLO class IDs in dataset label files and YAML configuration to match SAVANT ontology UIDs. This is essential when converting datasets to use SAVANT's standardized class identifiers.

**Usage:**

```bash
# Preview changes without modifying files (recommended first step)
remap-classes-to-ontology \
    --yaml dataset.yaml \
    --ontology ../Specification/savant_ontology_1.3.0.ttl \
    --labels-dir datasets/labels \
    --dry-run

# Execute remapping (updates both labels and YAML)
remap-classes-to-ontology \
    --yaml dataset.yaml \
    --ontology ../Specification/savant_ontology_1.3.0.ttl \
    --labels-dir datasets/labels

# Only update label files
remap-classes-to-ontology \
    --yaml dataset.yaml \
    --ontology ../Specification/savant_ontology_1.3.0.ttl \
    --labels-dir datasets/labels \
    --labels-only

# Only update dataset YAML
remap-classes-to-ontology \
    --yaml dataset.yaml \
    --ontology ../Specification/savant_ontology_1.3.0.ttl \
    --labels-dir datasets/labels \
    --yaml-only
```

**Arguments:**

- `--yaml` (required) - Path to dataset YAML file
- `--ontology` (required) - Path to SAVANT ontology file (.ttl)
- `--labels-dir` (required) - Base directory for label files (should contain train/ and val/ subdirectories)
- `--dry-run` - Preview changes without modifying files
- `--labels-only` - Only update label files, skip YAML
- `--yaml-only` - Only update YAML, skip label files

**What it does:**

1. Reads class names from the dataset YAML file
2. Matches them against SAVANT ontology classes
3. Creates a mapping from old class IDs to new ontology UIDs
4. Updates all label files in train/ and val/ directories
5. Updates the dataset YAML with new class IDs

**Example Workflow:**

```bash
# Step 1: Inspect the ontology
print-ontology ../Specification/savant_ontology_1.3.0.ttl --detailed

# Step 2: Preview the remapping (check for any issues)
remap-classes-to-ontology \
    --yaml UAV.yaml \
    --ontology ../Specification/savant_ontology_1.3.0.ttl \
    --labels-dir datasets/UAV_yolo_obb/labels \
    --dry-run

# Step 3: If everything looks good, execute
remap-classes-to-ontology \
    --yaml UAV.yaml \
    --ontology ../Specification/savant_ontology_1.3.0.ttl \
    --labels-dir datasets/UAV_yolo_obb/labels
```

**Output Example:**

```
================================================================================
SAVANT Ontology Class Remapper
================================================================================
Loading YAML from UAV.yaml...
Loading ontology from ../Specification/savant_ontology_1.3.0.ttl...
Creating class mapping...

Class ID Mapping:
  0: vehicle         → UID 2 (Vehicle)
  1: car             → UID 3 (Car)
  2: truck           → UID 5 (Truck)
  3: bus             → UID 9 (Bus)

✓ All classes matched successfully
✓ Remapping completed successfully
```

## Running with uv

You can run these scripts directly with uv without activating the virtual environment:

```bash
# From repository root
uv run print-ontology ../Specification/savant_ontology_1.3.0.ttl

uv run remap-classes-to-ontology \
    --yaml dataset.yaml \
    --ontology ../Specification/savant_ontology_1.3.0.ttl \
    --labels-dir datasets/labels
```

## Common Use Cases

### Inspecting an Ontology File

```bash
# See all classes
print-ontology ../Specification/savant_ontology_1.3.0.ttl

# See detailed information
print-ontology ../Specification/savant_ontology_1.3.0.ttl --detailed

# Find all vehicle classes
print-ontology ../Specification/savant_ontology_1.3.0.ttl --top-level RoadUserVehicle
```

### Converting a Dataset to SAVANT UIDs

```bash
# Always start with dry-run to preview changes
remap-classes-to-ontology \
    --yaml my_dataset.yaml \
    --ontology ../Specification/savant_ontology_1.3.0.ttl \
    --labels-dir datasets/my_dataset/labels \
    --dry-run

# If preview looks correct, execute
remap-classes-to-ontology \
    --yaml my_dataset.yaml \
    --ontology ../Specification/savant_ontology_1.3.0.ttl \
    --labels-dir datasets/my_dataset/labels
```

## Dependencies

These utilities use the `savant_common` package for ontology parsing:

```python
from savant_common.ontology import (
    read_ontology_classes,
    create_class_map,
    get_class_by_label,
)
```

## Notes

- **Always use `--dry-run` first** when using `remap-classes-to-ontology.py` to preview changes
- Class names in the dataset YAML must match ontology labels (case-insensitive)
- The remapper expects label files in YOLO format: `<class_id> <x> <y> <w> <h> <angle>`
- Label files should be organized in `train/` and `val/` subdirectories under `--labels-dir`
