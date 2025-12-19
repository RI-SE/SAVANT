# Markit

![SAVANT logo](../docs/savant_logo.png)

**Multi-engine object detection and tracking with OpenLabel output**

Markit is a command-line tool for detecting and tracking objects using oriented bounding boxes (OBB), the default YOLO model is tuned for drone footage of road traffic. It combines multiple detection engines, resolves conflicts between them, and exports results in an ASAM OpenLabel JSON compatible format (the tool supports a subset of OpenLabel).

## Features

- **Multi-Engine Detection** - YOLO OBB, optical flow, and ArUco marker detection
- **Conflict Resolution** - IoU-based merging when engines detect the same object
- **Oriented Bounding Boxes** - Proper rotation handling with continuous angle tracking
- **OpenLabel Export** - Schema-validated JSON output
- **Postprocessing Pipeline** - Gap filling, duplicate removal, rotation smoothing, static object handling
- **Provenance Tracking** - W3C PROV-JSON format via dataprov
- **Video Rendering** - Optional annotated output video with drawn detections

## Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detection Engines](#detection-engines)
- [Postprocessing](#postprocessing)
- [Output Format](#output-format)
- [ArUco Markers](#aruco-markers)
- [Configuration Reference](#configuration-reference)
- [Testing](#testing)
- [Coordinate Conventions](#coordinate-conventions)

## Installation

Markit is part of the SAVANT repository. Install from the repository root:

```bash
# Clone and install with uv (recommended)
git clone git@github.com:RI-SE/SAVANT.git
cd SAVANT
uv sync

# Or with pip
pip install -e .
```

Verify installation:

```bash
markit --help
```

### Dependencies

- Python >= 3.10
- ultralytics (YOLO OBB models)
- opencv-contrib-python >= 4.5.0
- numpy
- lap (tracking)
- rdflib (ontology parsing)
- dataprov (provenance tracking)

## Quick Start

Basic usage with YOLO detection:

```bash
markit --input video.mp4 --output_json output.json
```

With custom weights and output video:

```bash
markit --input video.mp4 --output_json output.json \
       --weights custom_model.pt --output_video annotated.mp4
```

With postprocessing enabled:

```bash
markit --input video.mp4 --output_json output.json --housekeeping
```

Using uv from repository root:

```bash
uv run markit --input markit/video.mp4 --output_json output.json
```

## Detection Engines

Markit supports three detection engines that can be used individually or combined.

### YOLO OBB Engine (default)

Uses Ultralytics YOLO with oriented bounding box support. Provides object classification and tracking.

```bash
markit --detection-method yolo --weights model.pt --input video.mp4 --output_json output.json
```

### Optical Flow Engine

Motion-based detection using background subtraction and optical flow. Useful for detecting moving objects without a trained model.

```bash
markit --detection-method optical_flow --input video.mp4 --output_json output.json \
       --motion-threshold 0.5 --min-object-area 200
```

### Combined Detection

Run both engines with IoU-based conflict resolution. YOLO takes precedence for overlapping detections.

```bash
markit --detection-method both --weights model.pt --input video.mp4 --output_json output.json \
       --iou-threshold 0.3
```

Disable conflict resolution to keep all detections:

```bash
markit --detection-method both --weights model.pt --input video.mp4 --output_json output.json \
       --disable-conflict-resolution
```

### ArUco Marker Detection

Enabled when providing a CSV file with marker positions:

```bash
markit --input video.mp4 --output_json output.json \
       --aruco-csv markers.csv --aruco-dict DICT_4X4_50
```

See [ArUco Markers](#aruco-markers) for CSV format details.

## Postprocessing

Enable all postprocessing passes with `--housekeeping`:

```bash
markit --input video.mp4 --output_json output.json --housekeeping
```

### Available Passes

| Pass | Description |
|------|-------------|
| Gap Detection | Identifies gaps in object tracking sequences |
| Gap Filling | Interpolates detections across small gaps |
| Duplicate Removal | Removes overlapping detections using IoU thresholds |
| First Detection Refinement | Refines initial detection angles using lookahead |
| Rotation Adjustment | Smooths rotation using movement direction |
| Sudden Detection | Flags objects appearing/disappearing far from frame edges |
| Frame Interval | Calculates frame intervals for each object |
| Static Object Removal | Removes or marks objects that don't move |

### Postprocessing Options

```bash
markit --input video.mp4 --output_json output.json --housekeeping \
       --duplicate-avg-iou 0.7 \
       --duplicate-min-iou 0.3 \
       --rotation-threshold 0.1 \
       --min-movement-pixels 5.0 \
       --temporal-smoothing 0.3 \
       --edge-distance 200 \
       --static-threshold 20 \
       --static-mark  # Mark instead of remove
```

## Output Format

Markit exports detections in OpenLabel (subset) JSON format, including information on annotator and confidence in annotation accuracy.

### JSON Structure

```json
{
  "openlabel": {
    "metadata": {
      "schema_version": "1.1",
      "tagged_file": "Saro_roundabout.mp4",
      "annotator": "SAVANT Markit 2.0.2",
      "name": "SAVANT Markit 2.0.0 Analysis",
      "comment": "Multi-engine object detection and tracking analysis of Saro_roundabout.mp4",
      "tags": [
        "object_detection",
        "tracking"
      ]
    },
    "ontologies": {},
    "objects": {
      "1": {
        "name": "Object-1",
        "type": "Car",
        "ontology_uid": "0",
        "frame_intervals": [
          {
            "frame_start": 0,
            "frame_end": 299
          }
        ]
      }
    }
    "frames": {
      "0": {
        "objects": {
          "1": {
            "object_data": {
              "rbbox": [
                {
                  "name": "shape",
                  "val": [
                    3154,
                    1876,
                    144,
                    71,
                    3.698
                  ]
                }
              ],
              "vec": [
                {
                  "name": "annotator",
                  "val": [
                    "markit_yolo"
                  ]
                },
                {
                  "name": "confidence",
                  "val": [
                    0.8921
                  ]
                }
              ]
            }
          }
        }
      }
   }
}
```

### Output Video

Generate an annotated video with drawn bounding boxes:

```bash
markit --input video.mp4 --output_json output.json --output_video annotated.mp4
```

### Provenance Tracking

Track processing provenance in W3C PROV-JSON format:

```bash
markit --input video.mp4 --output_json output.json --provenance provenance.json
```

The provenance file records inputs, outputs, parameters, and processing steps.

## ArUco Markers

ArUco markers can be used as ground control points with known GPS positions. When detected, markers are added to the OpenLabel output with their associated coordinates (see TestVids/Saro_roundabout for example).

### CSV Format

```csv
ID,long,lat,alt,horiz SD,vert SD,Location name
aruco_24a,47.3769,8.5417,410,0.02,0.03,Gothenburg
aruco_24c,47.3771,8.5419,410,0.02,0.03,Gothenburg
```

| Column | Description |
|--------|-------------|
| ID | Marker identifier in format `aruco_[num][a-d]` where letter indicates corner |
| long | Longitude of the marker corner |
| lat | Latitude of the marker corner |
| alt | Altitude in meters |
| horiz SD | Horizontal standard deviation (use `inf` if unknown) |
| vert SD | Vertical standard deviation (use `inf` if unknown) |
| Location name | Human-readable location identifier |

### Corner Notation

Each ArUco marker has 4 corners labeled a, b, c, d:

![ArUco coordinates](docs/coords_aruca.png)

Marker position is included in the OpenLabel output as objects with additional object_data including longitude, latitude, and altitude from the corner(s) where position is measured (from the csv file):

```json
     "2000017": {
        "name": "GbgSaroRound_17",
        "type": "ArUco",
        "ontology_uid": "0",
        "object_data": {
          "vec": [
            {
              "name": "arucoID",
              "val": [
                "17a",
                "17c"
              ]
            },
            {
              "name": "long",
              "val": [
                "12.00977788073061",
                "12.009746301733916"
              ]
            },
            {
              "name": "lat",
              "val": [
                "57.48451372894236",
                "57.484504456642185"
              ]
            },
            {
              "name": "alt",
              "val": [
                "72.75258941650391",
                "72.54821319580078"
              ]
            },
            {
              "name": "description",
              "val": "GbgSaroRound"
            }
          ]
        }
      }
```


### Usage

```bash
markit --input video.mp4 --output_json output.json \
       --aruco-csv markers.csv --aruco-dict DICT_4X4_50
```

Supported ArUco dictionaries: `DICT_4X4_50`, `DICT_4X4_100`, `DICT_4X4_250`, `DICT_4X4_1000`, `DICT_5X5_50`, etc., where default is DICT_4X4_50.

## Configuration Reference

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--input` | Path to input video file |
| `--output_json` | Path to output OpenLabel JSON file |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--weights` | `markit_yolo.pt` | YOLO weights file (.pt) |
| `--schema` | `../schema/savant_openlabel_subset.schema.json` | OpenLabel JSON schema |
| `--ontology` | `../ontology/savant.ttl` | SAVANT ontology file |
| `--ontology-uri` | extracted from file | Ontology URI for OpenLabel output |
| `--output_video` | - | Output annotated video path |
| `--aruco-csv` | - | CSV with ArUco marker positions |
| `--visual-markers` | - | CSV with visual marker positions (same format as ArUco) |
| `--provenance` | - | Provenance chain file path |

### Detection Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--detection-method` | `yolo` | `yolo`, `optical_flow`, or `both` |
| `--motion-threshold` | `0.5` | Optical flow motion threshold |
| `--min-object-area` | `200` | Minimum object area (pixels²) |
| `--aruco-dict` | `DICT_4X4_50` | ArUco dictionary type |

### Conflict Resolution

| Argument | Default | Description |
|----------|---------|-------------|
| `--iou-threshold` | `0.3` | IoU threshold for conflict detection |
| `--verbose-conflicts` | false | Enable verbose conflict logging |
| `--disable-conflict-resolution` | false | Keep all detections without merging |

### Postprocessing

| Argument | Default | Description |
|----------|---------|-------------|
| `--housekeeping` | false | Enable all postprocessing passes |
| `--duplicate-avg-iou` | `0.7` | Average IoU for duplicate detection |
| `--duplicate-min-iou` | `0.3` | Minimum IoU for duplicate detection |
| `--rotation-threshold` | `0.1` | Rotation adjustment threshold (radians) |
| `--min-movement-pixels` | `5.0` | Minimum movement for rotation calculation |
| `--temporal-smoothing` | `0.3` | Temporal smoothing factor (0-1) |
| `--edge-distance` | `200` | Edge distance for sudden detection (pixels) |
| `--static-threshold` | `20` | Static object movement threshold (pixels) |
| `--static-mark` | false | Mark static objects instead of removing |

### Logging

| Argument | Description |
|----------|-------------|
| `--verbose` | Enable detailed angle and detection logging |
| `--version` | Show version and exit |

## Testing

Run the test suite from the markit directory:

```bash
# All tests
pytest

# Unit tests only (fast)
pytest -m "not integration"

# Integration tests only
pytest -m integration

# With coverage report
pytest --cov=markitlib --cov-report=html

# Specific test file
pytest tests/test_geometry.py

# Verbose output
pytest -vv -s
```

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_geometry.py         # IoU and polygon operations
├── test_config.py           # Configuration and ontology
├── test_postprocessing.py   # Postprocessing passes
├── test_openlabel.py        # JSON generation and validation
├── test_integration.py      # End-to-end tests
└── fixtures/                # Test data
    ├── Kraklanda_short.mp4
    └── best.pt
```

## Coordinate Conventions

### Bounding Box Representation

Markit uses oriented bounding boxes represented as:
- **OBB corners**: 4 points `[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]`
- **XYWHR format**: center (x, y), dimensions (width, height), rotation (r) in radians

### Angle Convention

- Internal storage uses continuous unbounded angles to handle YOLO's π/2 ambiguity
- Output angles are normalized to `[0, 2π)` range
- Positive x-axis = 0 radians, rotation increases counterclockwise
- Semantic dimensions: width is always the longer axis

### Image Coordinates

- Origin (0, 0) at top-left
- x increases rightward
- y increases downward

## License

SAVANT is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
