[![CD](https://github.com/fwrise/SAVANT/actions/workflows/cd.yml/badge.svg)](https://github.com/fwrise/SAVANT/actions/workflows/cd.yml)
[![CI](https://github.com/fwrise/SAVANT/actions/workflows/ci.yml/badge.svg)](https://github.com/fwrise/SAVANT/actions/workflows/ci.yml)

# SAVANT
Development repository for RISE SAVANT - Semi-automated video annotation toolkit
![SAVANT logo](./savant_app/src/savant_app/frontend/assets/savant_logo.png)

```diff
- This repository is for development so we can add references and other stuff used temporarily. Will move the relevant parts to a new repo in the Rise org when there is something working which we want to release later.
```

## Installation

### Prerequisites
- Python 3.10 or higher
- Git (with SSH access to fwrise repositories)
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Install from Repository

**Using uv (recommended):**

```bash
# Clone the repository
git clone git@github.com:fwrise/SAVANT.git
cd SAVANT

# Sync dependencies and create virtual environment
uv sync

# Or sync with development dependencies
uv sync --dev

# Activate the virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows
```

**Alternative using pip:**

```bash
# Clone the repository
git clone git@github.com:fwrise/SAVANT.git
cd SAVANT

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Verify Installation

```bash
# Check that CLI tools are available
markit --help
train-yolo-obb --help

# Test imports
python -c "from savant_common import read_ontology_classes; print('✓ savant_common installed')"
```

### Running Commands with uv

You can run commands directly with uv without activating the virtual environment:

```bash
# Run markit
uv run markit --input video.mp4 --output_json output.json

# Run training
uv run train-yolo-obb --data dataset.yaml --epochs 50

# Run Python scripts
uv run python -c "from savant_common import read_ontology_classes; print('OK')"
```

## Quick Start

### Object Detection with markit
```bash
# With uv (from project root)
uv run markit --input markit/video.mp4 --output_json output.json

# Or activate venv and run from markit directory
source .venv/bin/activate
cd markit
markit --input video.mp4 --output_json output.json
```

### Train YOLO Model with trainit
```bash
# With uv (from project root)
uv run train-yolo-obb --data trainit/dataset.yaml --epochs 50

# Or activate venv and run from trainit directory
source .venv/bin/activate
cd trainit
train-yolo-obb --data dataset.yaml --epochs 50
```

## Repository Structure

- **savant_common/** - Shared utilities (ontology parsing, etc.)
- **markit/** - Object detection and video annotation tool
- **trainit/** - YOLO model training and dataset preparation tools
- **utils/** - Command-line utilities (ontology inspection, dataset remapping)
- **Specification/** - Ontology and OpenLabel schema files
- **savant_app/** - Qt6 desktop application (separate installation)

## License

SAVANT is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

This project uses dependencies with the following licenses:
- **Ultralytics YOLO** (AGPL-3.0) - Required for object detection
- **PyQt6** (GPL-3.0) - Used in savant_app
- **OpenCV** (Apache 2.0)
- **NumPy** (BSD-3-Clause)
- **RDFlib** (BSD-3-Clause)
- **Pydantic** (MIT)

All dependencies are compatible with AGPL-3.0 licensing.

### What does AGPL-3.0 mean?

- You are free to use, modify, and distribute this software
- If you distribute modified versions, you must share the source code under AGPL-3.0
- If you use this software to provide a network service (e.g., web API), you must make the source code available to users of that service
- Commercial use is permitted, but the source code must remain available

For the complete license text, see the [LICENSE](LICENSE) file. For more information about AGPL-3.0, visit https://www.gnu.org/licenses/agpl-3.0.html

See [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) for detailed usage examples.
