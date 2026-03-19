# SAVANT Copilot Instructions

SAVANT is a semi-automated video annotation toolkit for UAV/drone footage. It has three main applications (`markit`, `edit`, `trainit`) plus a shared library (`savant_common`) and CLI utilities (`utils`).

## Build & Install

Uses `uv` (not pip directly). Python 3.12 in CI, requires >=3.10.

```bash
# Full install (all apps)
uv sync --group all-apps --group dev

# Per-component installs
uv sync --group markit --group dev
uv sync --group edit --group dev
uv sync --group trainit --group dev
```

## Test Commands

```bash
# All tests
uv run pytest

# Single component
uv run pytest markit/tests
uv run pytest edit/tests
uv run pytest trainit/tests

# Single test file
uv run pytest markit/tests/test_foo.py

# Single test function
uv run pytest markit/tests/test_foo.py::TestClass::test_name

# Edit has a dedicated entry point (runs headless)
uv run test-edit
```

Test markers: `@pytest.mark.unit` and `@pytest.mark.integration`. Run only unit tests: `uv run pytest -m unit`.

## Lint

```bash
uv tool run flake8 markit/
uv tool run flake8 trainit/
uv tool run flake8 edit/
# Format (not enforced by CI, but available)
uv tool run black .
```

## Architecture

```
savant_common/   ← shared library: OpenLabel I/O, ontology parsing (Pydantic + rdflib)
ontology/        ← savant.ttl  (ASAM-compatible traffic ontology, RDF/Turtle)
schema/          ← savant_openlabel_subset.schema.json (JSON Schema for output validation)
markit/          ← CLI tool: multi-engine detection → OpenLabel JSON
edit/src/edit/   ← PyQt6 desktop app: manual annotation / QA
trainit/         ← CLI tools + PyQt6 GUI: dataset management and YOLO training
utils/           ← standalone CLI utilities (ontology inspection, class remapping)
```

**Data flow:** `markit` produces OpenLabel JSON validated against `schema/`. `edit` reads/writes that same format. `trainit` consumes OpenLabel + ontology to build YOLO training datasets.

### `savant_common` — the shared core

All apps import from here:
- `savant_common.openlabel` — `OpenLabel` (Pydantic read model), `OpenLabelWriter` (write), `load_openlabel()`, `DetectionData` dataclass
- `savant_common.ontology` — RDF/Turtle parsing, `create_class_map()`, `read_ontology_classes()`

### `edit` — MVC-style PyQt6 app

- `models/` — Pydantic/dataclass data models (own `OpenLabel.py`, not `savant_common`)
- `services/` — pure business logic (no Qt); `AnnotationService`, `ProjectState`, `VideoReader`, `InterpolationService`
- `controllers/` — bridge services↔frontend; wrap calls with `@error_handler` decorator from `error_handler_middleware.py`
- `frontend/` — Qt widgets and views

### `trainit` — MVC-style PyQt6 app (same pattern as `edit`)

Follows the same `models/` / `services/` / `controllers/` / `frontend/` split under `trainit/trainit_gui/`.

### `markit` — pipeline CLI

`run_markit.py` → `markitlib/` pipeline stages. Multi-engine detection (YOLO OBB, optical flow, ArUco), IoU-based conflict resolution, postprocessing, OpenLabel export.

## Key Conventions

- **Pydantic v2** everywhere for data models. `@model_validator(mode="before")` is used for list-to-model deserialization (e.g., `RotatedBBox` from a 5-element list).
- **`@error_handler` decorator** on all controller methods in `edit` — catches service exceptions and surfaces them to the UI rather than crashing.
- **`Path` objects** throughout; avoid raw string paths.
- **Rotated bounding boxes** are `[x_center, y_center, width, height, rotation_radians]`. Angles normalized to `[0, 2π)` for OpenLabel output via `normalize_angle_to_2pi_range()`.
- **Ontology UIDs** (integers) are the canonical class identifiers; human-readable labels are mapped via `create_class_map()`.
- **OpenLabel JSON** is always schema-validated on write. The schema is at `schema/savant_openlabel_subset.schema.json`.
- **Entry points** are declared in `pyproject.toml` under `[project.scripts]`: `markit`, `edit`, `trainit`, `train-yolo-obb`, `split-train-val`, `extract-yolo-training`, etc.
- **`uv run <entry-point>`** is the standard way to invoke tools during development.
- **Dependency groups** are per-app (`markit`, `trainit`, `edit`). `savant_common` has no group — it's always included as the core package.
- Docstrings: single-line summary only. No `Args`/`Returns` unless non-obvious.
