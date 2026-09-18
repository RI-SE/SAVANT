# Changelog

> [!NOTE]
> **This changelog was reconstructed retroactively** from git tags, commit history, and `pyproject.toml` version bumps — no changelog was maintained during development (added 2026-09-18). Entries summarize commit subjects and dates; they were **not curated at release time** and may be incomplete, mis-categorized, or omit changes that weren't reflected in a commit message. Treat this as a best-effort historical index, not an authoritative release record.
>
> **Versioning discontinuity:** versions `1.0.0`–`1.3.2` belong to an earlier, pre-monorepo project (a single `savant_app`, no root `pyproject.toml`). The restructure into today's `markit` / `edit` / `trainit` / `savant_common` layout (commit `acde3ea`, 2025-11-25) introduced the current root `pyproject.toml`, which restarted its own version count at `1.0.0` and then jumped straight to `2.0.0` twelve days later ("Bump versions", commit `1092f13`) to mark the restructure. So `2.0.0` is the first version of the project *as it exists today*; the `1.x` entries below describe the predecessor codebase.

## [Unreleased]
_Commits since `v2.4.1` (2026-06-17), including the untagged `2.4.2` version bump._

- Object/frame tag and confidence-warning UI overhaul in `edit`: new tag/warning navigator widget, moved object tags to sidebar with removal support, corrected confidence-warning display, auto-generation of additional standard object tags.
- Fixed a markit bug leaving orphaned frame objects behind when cascading duplicate-frame removal.
- Added "inspection mode" and pre-selection for the "Fix range" action in `edit`.
- Fix to persist the bbox centring setting.
- General postprocessing fixes; `uv.lock` refresh; lint cleanup; removed accidentally committed files.

## [2.4.1] - 2026-06-17
- Added spline-based angle interpolation for bounding-box rotation, both as a markit postprocessing pass (`AngleSplineInterpolationPass`, documented) and as an `edit` context-menu action.
- Fixed a degrees/radians mixup that broke linear interpolation.
- Added "retrack entire range for an object", and sidebar improvements: jump to an object's first/last frame, faster frame jumping, better frame-range display.
- Added lock-to-center functionality for bounding boxes.
- Fixed a PyQt teardown bug that could segfault at exit.

## [2.4.0] - 2026-04-08
- Added per-frame timing info and a "measure" function.
- Added stream metadata to OpenLabel output.
- Increased fidelity of rotation adjustments.

## [2.3.1] - 2026-03-23
- Refactoring pass: split large functions/modules for maintainability, restructuring around test/import-dependency issues.
- Multiple tracking-exception and UI-freeze fixes; fixed link-objects behavior for frame overlaps and a bookmark integer/string bug.
- Improved dependency license information; dependency updates.

## [2.3.0] - 2026-03-06
- Better ghost-object removal for optical flow; tracker fixes after OpenCV 4.11.0 removed some tracker implementations.
- Added caching in `edit` for smoother navigation/faster tracking; introduced re-track functionality.
- Fixed a 90-degree angle bug affecting both markit and edit.

## [2.2.0] - 2026-02-11
- Added bookmarks (with notes), spacebar frame-advance, go-to-frame, zoom improvements, and tab-switching between bounding boxes in `edit`.
- Added right-click "copy previous frame bbox params".
- Documentation and lint fixes; cascade warning improvements.

## 2.1.0 - 2026-02-09 _(untagged, superseded by 2.2.0 two days later)_
- Major optical-flow engine rework: new algorithms/config, new smoothing postprocessing pass, extensive rotation-adjustment and duplicate-removal bug fixes, IoU-min metric for duplicate detection, configurable housekeeping.
- Added a tracking feature to `edit`.
- Updated default-file discovery (schema, ontology, YOLO weights bundled in the repo).

## [2.0.3] - 2026-01-23
- Added a first VLM (vision-language model) scene-tagging feature to markit, including prompt/response-parsing iteration, confidence/annotator fields, and `edit` support for displaying VLM tags.
- Unified versioning: all SAVANT modules now share a single version number (previously divergent).

## [2.0.2] - 2025-12-19
- Release fix (hotfix on 2.0.1).

## [2.0.1] - 2025-12-18
- Fixed CI issues with the PyQt dependency and a hardcoded test asset path.
- Removed the old `Specification` folder in favor of the unified `pyproject.toml` `all-apps` target.
- Documentation reorganization (separate dev/user READMEs for `edit`).

## [2.0.0] - 2025-12-17
**Monorepo restructure** — `savant_app` split into `markit` / `edit` / `trainit`, plus new shared `savant_common` library; unified root `pyproject.toml`/CI (moved from rye to uv).
- Added object relationships in `edit`: create/display/delete relationships between objects, with undo/redo support.
- Added `dataprov`-based provenance tracking to markit and `train_yolo_obb`.
- Relicensed to AGPL-3.0 (required by the `ultralytics` dependency).
- Added a first `trainit-gui` implementation.
- Added object/frame tag warning markers, an "About" page, and various lint/test fixes from the restructure.

## [1.3.2] - 2025-11-19
- Added the ArUco marker processing engine to markit.
- Added confidence/annotator tracking end-to-end: per-change annotator attribution, confidence display, and a warning/error sidebar list for confidence issues (with sorting and right-click resolution).
- Added straight-road annotation interpolation (spline groundwork; later superseded).
- Added bbox movement/rotation via arrow keys, adjustable movement sensitivity, and undo/redo for user changes.
- Renamed/restructured YOLO training tooling into the new `trainit` tool; added more YOLO training parameters.

## [1.2.1] - 2025-10-23
- Added cascading edits (size, rotation, center) across frames, with user-adjustable cascade behavior.
- Added a GNU GPLv3 license; modularized/refactored the markit tool and updated the SAVANT ontology.
- Added a static-object detection pass and ontology integration in markit.
- Removed macOS from the deployment workflow.

## [1.1.0] - 2025-10-17
- Added unified error handling (custom exceptions, global exception handler) and structured logging to file.
- Added bbox selection ↔ active-object-list linking, and object ID/type display on annotations.
- Added frame tags (with cascading delete) and settings-panel reorganization.
- Various ontology-loading and import-error fixes.

## [1.0.0] - 2025-09-29
Initial tagged release of the (pre-restructure) `savant_app` project.
- Initial UI, CI/CD pipeline, and project architecture.
- First version of the markit tool: YOLO-OBB based annotation script, OpenLabel subset output.
- First version of the SAVANT OpenLabel spec/ontology model.
- Bounding-box creation, editing, resizing, moving, and rotation in the UI; project state save/load; video split and playback controls.
