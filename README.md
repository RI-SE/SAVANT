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

See [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) for detailed usage examples.
