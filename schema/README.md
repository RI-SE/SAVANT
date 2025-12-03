# SAVANT OpenLabel Schema

This directory contains the JSON schema for SAVANT's OpenLabel subset.

## Overview

SAVANT uses a subset of the [ASAM OpenLabel](https://www.asam.net/standards/detail/openlabel/) format for video annotation data. This schema defines the structure for storing object detections, tracking data, and behavioural annotations.

**Schema Version:** 1.1
**Format:** JSON Schema

## Structure

### Root

```json
{
  "openlabel": {
    "metadata": { ... },
    "ontologies": { ... },
    "objects": { ... },
    "actions": { ... },
    "frames": { ... }
  }
}
```

### Metadata

Schema version is required. We use `tagged_file` for the source video and `annotator` for person(s) or tool(s) annotating.

```json
"metadata": {
    "schema_version": "1.1.0",
    "tagged_file": "filename.mp4",
    "annotator": "SAVANT markit v0.3.2"
}
```

### Ontologies

References to ontology definitions. Each ontology has a unique UID.

```json
"ontologies": {
    "0": "https://github.com/RI-SE/SAVANT/ontology/savant_ontology_1.3.1.ttl"
}
```

### Objects

Static information about objects in the video. Contains type, name, and optionally frame intervals where the object appears.

```json
"objects": {
    "0": {
        "name": "Car-0",
        "type": "Car",
        "ontology_uid": "0",
        "frame_intervals": [{ "frame_start": 0, "frame_end": 10 }]
    },
    "1": {
        "name": "Person-1",
        "type": "Pedestrian",
        "ontology_uid": "0"
    }
}
```

#### ArUco Marker Objects

ArUco markers include GPS coordinates for one or more corner(s):

```json
"2": {
    "name": "GbgSaroRound_24",
    "type": "ArUco",
    "ontology_uid": "0",
    "object_data": {
        "vec": [
            { "name": "arucoID", "val": ["24a", "24c"] },
            { "name": "long", "val": ["12.010028", "12.010052"] },
            { "name": "lat", "val": ["57.484172", "57.484185"] },
            { "name": "alt", "val": ["75.032", "75.017"] },
            { "name": "description", "val": "GbgSaroRound" }
        ]
    }
}
```

### Actions

Semantically meaningful acts occurring over frame intervals (e.g., overtake, lane change).

```json
"actions": {
    "0": {
        "name": "Action-0",
        "type": "Overtake",
        "ontology_uid": "0",
        "frame_intervals": [{ "frame_start": 5, "frame_end": 8 }]
    }
}
```

### Frames

Dynamic per-frame information, primarily bounding boxes for tracked objects.

```json
"frames": {
    "0": {
        "objects": {
            "0": {
                "object_data": {
                    "rbbox": [{ "name": "shape", "val": [x, y, width, height, angle] }],
                    "vec": [
                        { "name": "annotator", "val": ["SAVANT markit v0.3.2"] },
                        { "name": "confidence", "val": [0.87] }
                    ]
                }
            }
        }
    }
}
```

## Rotated Bounding Box (rbbox)

SAVANT uses rotated bounding boxes for all dynamic objects:

| Parameter | Description |
|-----------|-------------|
| x | Center x-coordinate (pixels) |
| y | Center y-coordinate (pixels) |
| width | Box width (pixels) |
| height | Box height (pixels) |
| angle | Rotation angle (radians, 0 to 2π) |

![Rotated bounding box](docs/OpenLabel_rbbox.png)

## Validation

Validate OpenLabel files against the schema:

```python
import json
import jsonschema

with open("schema/savant_openlabel_subset.schema.json") as f:
    schema = json.load(f)

with open("output.json") as f:
    data = json.load(f)

jsonschema.validate(data, schema)
```

Or using markit with schema validation:

```bash
markit --input video.mp4 --output_json output.json \
       --schema schema/savant_openlabel_subset.schema.json
```

## References

- [ASAM OpenLabel](https://www.asam.net/standards/detail/openlabel/)
- [SAVANT Ontology](../ontology/README.md)
- [markit Documentation](../markit/README.md)
