# Changelog

> [!NOTE]
> **This changelog was reconstructed retroactively** from git tags, commit history, and `pyproject.toml` version bumps — no changelog was maintained during development (added 2026-09-18). Entries summarize commit subjects and dates; they were **not curated at release time** and may be incomplete, mis-categorized, or omit changes that weren't reflected in a commit message. Treat this as a best-effort historical index, not an authoritative release record.
>
> **Versioning discontinuity:** versions `1.0.0`–`1.3.2` belong to an earlier, pre-monorepo project (a single `savant_app`, no root `pyproject.toml`). The restructure into today's `markit` / `edit` / `trainit` / `savant_common` layout (commit `acde3ea`, 2025-11-25) introduced the current root `pyproject.toml`, which restarted its own version count at `1.0.0` and then jumped straight to `2.0.0` twelve days later ("Bump versions", commit `1092f13`) to mark the restructure. So `2.0.0` is the first version of the project *as it exists today*; the `1.x` entries below describe the predecessor codebase.

## [2.5.0] - 2026-10-20
_Changes since `v2.4.1` (2026-06-17), including the untagged `2.4.2` version bump._

- Object/frame tag and confidence-warning UI overhaul in `edit`: new tag/warning navigator widget, moved object tags to sidebar with removal support, corrected confidence-warning display, auto-generation of additional standard object tags.
- Fixed a markit bug leaving orphaned frame objects behind when cascading duplicate-frame removal.
- Added "inspection mode" and pre-selection for the "Fix range" action in `edit`.
- Fix to persist the bbox centring setting.
- General postprocessing fixes; `uv.lock` refresh; lint cleanup; removed accidentally committed files.
- YOLO model choice (YOLO11/YOLO26) in `trainit`; `--conf` threshold in markit; richer provenance (library versions, model architecture).

### Scenario ontology and VLM output (2026-10)
- `savant_scenario.ttl` 2.0.0: URI now `https://ri-se.github.io/SAVANT/ontology/savant_scenario#`; classes mirror ASAM OpenLABEL scenario tagging 1:1 (same names/hierarchy, only those used) with SAVANT extension subclasses; uids moved to 1000+ (disjoint from `savant.ttl`); unused classes removed.
- VLM tags/contexts are now typed by scenario-ontology class, one tag per field value; the field→class mapping is shared by markit and edit (`savant_common.scenario_tags`).
- Relations and VLM output reference the scenario ontology by its key in the `ontologies` section (was a hardcoded/invalid `ontology_uid`); edit's VLM dialog re-types tags when values are edited.
- The third-party OpenLABEL ontology file is no longer tracked (gitignored, not redistributed).

### Code review fixes (2026-10-20)
Whole-repo review; details and remaining items in `development.md`.

**Behaviour changes to be aware of**
- markit: object `frame_intervals` are now one interval per contiguous run of frames (gaps no longer reported as presence).
- markit: ArUco objects use the ontology label `"Aruco"` (was `"ArUco"`).
- markit: YOLO weights are never downloaded; uses `--weights` or the repo's `markit/markit_yolo.pt`, else stops with an error.
- markit: VLM weather/traffic contexts cover the whole video (a sampled value holds until the next sample); single-sample segments are merged as noise; majority-vote ties are deterministic.
- markit: provenance records every CLI option (never the API key) and all input files.
- markit: the VLM API key can be given as `$SAVANT_VLM_API_KEY`; warning when a key would be sent over plain HTTP.
- markit: a detection engine failing on 10 consecutive frames aborts the run instead of silently writing empty detections.
- edit: on save, object `frame_intervals` are rebuilt from the boxes; objects with no boxes (and relations to them) are not written.
- edit: ontology paths referenced by an OpenLABEL file are only used if they lie inside the project folder.
- trainit: train/val re-split holds out whole 600-frame chunks (20 s at 30 fps) per dataset/sequence and excludes a 120-frame gap around them, removing leakage of neighbouring frames (GUI and `split-train-val`; new `--chunk-frames`/`--gap-frames`, gap frames go to `excluded/`).
- trainit: `split-train-val --remove-orphans` no longer also splits.
- trainit: explicit CLI options override `--config` values (also when equal to the default); the GUI's device choice is now honoured.
- trainit: `settings.yaml` disables Ultralytics analytics (`sync`) and all tracker integrations.
- trainit: extraction clips boxes at the image edge to the visible part (instead of clamping corners) and drops boxes fully outside the image.

**Fixed — data loss and crashes**
- markit: a VLM failure no longer loses the whole run (detections are still saved); `scene_change`/`keyframes` sampling no longer crashes; malformed VLM replies are validated/coerced instead of crashing or producing false tags.
- edit: "mark confidence resolved" crashed; undo history from a previous project could modify a newly opened one; the new-project dialog could delete the selected file; saving is now atomic.
- trainit: saving a config dropped the split settings; regenerating left stale links that leaked val images into train; `project.json` saves are atomic.

**Fixed — wrong results**
- markit: one optical-flow track could take two detections in a frame; ArUco boxes were always 50×50; chained duplicate merges could lose frames; ontology uid `"0"` could collide with the scenario ontology; fallback config defaults disagreed with the CLI.
- markit/edit: oriented-box IoU assumed counter-clockwise corners (wrong for clockwise boxes such as Ultralytics'); now one shared OpenCV-based implementation.
- edit: relationships were sent to the overlay with a wrong id; video-buffer setting never applied; annotator check matched substrings; non-numeric relation ids broke relation creation; the unsaved-changes prompt is now based on the actual model state.
- edit undo: tracking undo removed manual boxes and partial tracking was not rolled back; geometry undo rewrote annotator/confidence history; a failing undo dropped the command; deleting/restoring relations lost their intervals; invalid frame-tag edits lost the original; linking objects dropped object tags and left dangling relations; undo of tags not in the ontology failed.
- edit: failed ontology checks left a half-switched project; inspection silently skipped failed tags.
- trainit: split was not reproducible across machines; all extracted datasets were pooled as one sequence; optimizer `"auto"` in project defaults broke new configs; empty model/project fields were saved as `""`; `warmup_epochs` could not take decimals; unsaved changes were never detected.
- trainit: names containing path separators (`../`) are rejected; manifest generation fails instead of silently omitting files it can't hash.

**Performance**
- markit: VLM scene-change sampling reads frames sequentially (~27× faster); duplicate removal ~4.7× faster with identical results; ontology parsing cached (was re-parsed per object); conflict decisions only kept when `--decision-log` is given.
- edit: undo history capped at 500 steps; frame-tag lookup no longer caches stale data.
- trainit: generate preview debounced; datasets read once per analysis; generation shows a cancellable progress dialog.

**Internal**
- VLM multi-frame aggregation and trainit training parameters are each driven by one list, with tests guarding against drift; duplicated geometry, `NumpyEncoder` and angle helpers consolidated in `savant_common`.
- `run_markit.main` split into stage functions; edit frontend no longer reaches into services or other widgets' private members; dead code removed; broad exception handlers that hid failures now log.
- Many new regression tests (suite: 665 passed).

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
