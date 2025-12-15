# Savant App Documentation

## Audience & Scope
This document covers everything you need to run the Savant desktop annotator (`savant_app`) and everything you need to maintain or extend it. It is split into two parts:

1. **User Guide** – installing, running, and using the app to review videos and adjust SAVANT OpenLabel annotations.
2. **Maintainer Guide** – architecture, directory layout, extension points, and validation workflows for contributors.

---

## 1. User Guide

### 1.1 Prerequisites and Installation
1. Install [uv](https://docs.astral.sh/uv) if you have not already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Clone the repository and enter the Savant app directory:
   ```bash
   git clone git@github.com:fwrise/SAVANT.git
   cd SAVANT/savant_app
   ```
3. Install dependencies (PyQt6, OpenCV, Pydantic, etc.) via uv:
   ```bash
   uv sync --all-groups
   ```

### 1.2 Running Tests and Quality Tools
Use the uv workflows documented in `savant_app/README.md`:
```bash
# Run the entire pytest suite (unit tests live under savant_app/tests)
uv run pytest

# Run a specific module with verbose output
uv run pytest savant_app/tests/unit/services/test_annotation_service.py -vv

# Run lint / formatting helpers
uv run flake8
uv run black
```

### 1.3 Launching the Application
1. Start from `savant_app/src/savant_app`.
2. Run the packaged entry point:
   ```bash
   uv run python -m savant_app.main
   ```
3. On launch the app:
   - Sets up rotating log handlers (`savant_app.log` alongside an OS-specific log dir).
   - Creates a shared `ProjectState` to hold metadata and the loaded OpenLabel config.
   - Instantiates service and controller layers before opening the PyQt6 `MainWindow`.

### 1.4 Typical Project Workflow
1. **Load video and OpenLabel config** using the _File_ menu.
2. The seek bar, playback controls, keyboard shortcuts (Undo/Redo, Ctrl+Y) and the overlay let you scrub frames and visualize interpolated vs manual boxes.
3. **Create or edit annotations** via menu actions or sidebar buttons:
   - `Create New Bounding Box` prompts for object type, draws the initial rotated bbox, and records the annotator.
   - `Add BBox to Existing Object` attaches another frame to an existing object track.
   - `Cascade Edit` applies width/height/rotation changes from a starting frame across subsequent frames.
   - `Move/Resize` drags handles directly on the overlay; each operation logs the annotator plus sets confidence back to 1.0.
4. **Manage tags and metadata**:
   - Frame tags come from ontology “Action” labels; use `Create Frame Tag` to add intervals and `Remove` to delete an interval.
   - Object metadata (name/type) is editable from the sidebar; types are sourced from ontology Dynamic/Static classes.
5. **Interpolate**: pick start/end frames and let the interpolation service fill in bounding boxes in between.
6. **Relationships**: the relationship dialog lets you link two objects with an ontology-defined relation. The app automatically scopes relations to frame intervals in which both objects exist.
7. **Undo / Redo**: almost every service call is wrapped by the undo manager for quick correction.
8. **Save**: `Quick Save` or `Save As` writes the updated OpenLabel JSON by calling `ProjectState.save_openlabel_config`.

### 1.5 Settings and Ontology
- The ontology path (TTL file), namespace, tag visibility toggles, movement/rotation sensitivity, frame history count, and zoom rate are stored via `frontend/utils/settings_store.py`.
- Select an ontology under the Settings dialog before tagging; actor types, frame tags, object types, and relation labels all come from this TTL.
- Warning/error confidence thresholds can be configured in Settings and are shown directly on the seek bar.

### 1.6 Logging and Error Handling
- Logs are written both to `savant_app.log` (repo root) and the OS log directory retrieved via `appdirs.user_log_dir`.
- All controller methods are wrapped with `error_handler_middleware`, ensuring:
  - Domain errors (e.g., invalid frame range, missing object) are shown as warnings.
  - Unexpected issues are converted to a generic “An unexpected error occurred” dialog and logged as `InternalException`.
- Qt’s `sys.excepthook` is replaced to ensure any unhandled exception still triggers a UI dialog and a traceback in the log file.

---

## 2. Maintainer & Contributor Guide

### 2.1 Architecture Overview
- The architecture follows clean layering:
  1. **Frontend (PyQt6)** – widgets, menus, playback controls, overlay (under `savant_app/src/savant_app/frontend`).
  2. **Controllers** – `AnnotationController`, `ProjectStateController`, `VideoController` translate UI calls into service calls.
  3. **Services** – business logic for annotations (`annotation_service.py`), state persistence (`project_state.py`), video access (`video_reader.py`), and interpolation (`interpolation_service.py`).
  4. **Models** – `OpenLabel.py` holds the Pydantic schema plus helpers for manipulating bounding boxes, frame tags, and relationships.
  5. **Data** – persisted SAVANT OpenLabel JSON files produced by upstream pipelines (e.g., Markit).
- `Doc_images/Architecture.png` and `Doc_images/DataCommunication.png` visualize this separation and data flow.

### 2.2 Key Directories
```
savant_app/
├── README.md          – quick start commands (uv install, testing)
├── src/savant_app
│   ├── main.py        – entry point
│   ├── controllers/   – UI-facing controller layer
│   ├── services/      – annotation, state, video, interpolation services
│   ├── models/        – OpenLabel schema and helpers
│   ├── frontend/      – PyQt widgets, states, utils, themes
│   ├── global_exception_handler.py / logger_config.py
│   └── utils.py       – shared JSON file helper
└── tests/             – pytest suites for services, models, and controllers
```

### 2.3 Runtime Flow (main.py)
1. `setup_logger()` configures file (rotating) + console logging.
2. Instantiate `ProjectState` (shared annotation/video metadata).
3. Spin up `QApplication`, install the custom menu styler.
4. Build services: `VideoReader` (needs `ProjectState` for metadata) and `AnnotationService`.
5. Build controllers that wrap the services.
6. Create `MainWindow`, injecting controllers and shared state.
7. Replace `sys.excepthook` with `global_exception_handler`.
8. Start the Qt event loop (`app.exec()`).

### 2.4 Service Responsibilities
- **AnnotationService**
  - CRUD for bounding boxes, interpolations, frame tags, and metadata updates.
  - Actor lists, bbox type lists, and frame tag labels are cached and derived from the ontology path managed by the settings store.
  - Relationship helpers compute frame interval intersections before writing relation metadata.
  - Raises domain-specific exceptions defined in `services/exceptions.py`; controllers rely on these classes to display user-friendly errors.
- **ProjectState**
  - Loads/saves OpenLabel JSON by delegating to `utils.read_json` and `OpenLabel.model_dump`.
  - Aggregates metadata for the UI: boxes per frame, object names/types, frame/object tags, and per-frame confidence issues.
  - Maintains auxiliary state such as the active video metadata and which frames contain interpolated annotations.
  - Validates frame tag intervals before saving (start/end presence, start ≤ end).
- **VideoReader**
  - Wraps OpenCV’s `VideoCapture` to provide iteration, random access, skip/back, and metadata population.
  - Uses typed `VideoMetadata` dataclass for FPS, frame count, and resolution.
- **InterpolationService**
  - Stateless helper that linearly interpolates bbox centers, width, height, and rotation values between two keyframes.

### 2.5 Model Layer (OpenLabel.py)
- Pydantic models for:
  - `RotatedBBox`, `GeometryData`, `ObjectData`, `FrameLevelObject`, `FrameObjects`.
  - `ActionMetadata` (frame tags), `RelationMetadata`, `FrameInterval`.
  - RDF references (subjects/objects) and ontology metadata.
- Helper methods:
  - `append_object_bbox`, `update_bbox`, `delete_bbox`, `restore_bbox`.
  - `add_object_relationship`, `restore_relationship`, `delete_relationship`.
  - Annotator/confidence bookkeeping ensures each edit records the latest editor with a confidence default of 1.0 unless downstream detectors set differently.
- `model_dump(exclude_none=True)` keeps persisted files clean of unused fields.

### 2.6 Frontend Overview
- `frontend/main_window.py` constructs:
  - Video display stack (`VideoDisplay`, overlay, seek bar, playback controls).
  - Sidebar (object list, metadata editing, relationships).
  - Menu actions covering project flows, bbox/tag creation, interpolation, relationships, annotator switching, settings, and about dialog.
  - Undo/redo manager (`frontend/utils/undo`) that uses controller gateways to capture reversible commands.
- Supporting modules in `frontend/utils`:
  - `annotation_ops`, `navigation`, `playback`, `render`, `zoom` wire controller calls to UI widgets.
  - `project_io` handles file dialogs and ensures state + controllers are synchronized when loading/saving projects.
  - `settings_store` persists user preferences (see §1.5).
  - `ontology_utils` parses TTL using `rdflib` and caches ontology-derived label sets.

### 2.7 Error Handling & Logging
- Centralized exception hierarchy in `services/exceptions.py` distinguishes domain vs internal errors.
- `controllers/error_handler_middleware.py`:
  - Logs domain errors at WARNING with context.
  - Converts unexpected exceptions into `InternalException`, preserving the original stack trace in logs.
- `global_exception_handler.py` intercepts Qt-level exceptions and shows QMessageBoxes for domain, frontend, or unexpected errors.
- `logger_config.py` ensures all logs propagate to both console and rotating files for easier debugging on user machines.

### 2.8 Testing & QA
- Unit tests live under `savant_app/tests/unit`, grouped by layer:
  - `services/test_annotation_service.py` and `test_annotation_service_relations.py`.
  - `services/test_project_state.py`, `test_video_reader.py`, `test_interpolation_service.py`.
  - Controller/front-end utility tests (under `tests/unit/controllers` and `tests/unit/frontend`) cover error handling and helper logic.
- Use `uv run pytest` for the whole suite; target modules with `uv run pytest path/to/test.py -vv`.
- When adding features, include regression tests that:
  - Exercise service-side validation paths.
  - Confirm controller middleware surfaces domain errors (mock services where appropriate).
  - Cover schema changes via new fixtures or pydantic validations.

### 2.9 Contribution Workflow
1. Create a feature branch from `main`.
2. Add or update documentation (this file, `savant_app/README.md`, or doc images) when introducing meaningful feature changes.
3. Run `uv sync --all-groups` (or target groups as needed) if dependencies change and update `pyproject.toml` / lockfiles.
4. Run `uv run pytest`, `uv run flake8`, and `uv run black`.
5. Commit with descriptive messages and open a PR summarizing the change, affected modules, and QA steps.
6. Include screenshots or short clips for UI-heavy changes when possible.

---

For quick reference, keep `savant_app/README.md` for onboarding commands, and use `docs/savant_app.md` (this file) as the canonical in-repo handbook for both users and contributors.
