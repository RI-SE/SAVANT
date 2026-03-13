# SAVANT Codebase Refactoring Plan

## Summary

Analysis of `edit/src/` and `markit/markitlib/` against an 80-line function soft limit.
52 functions exceed the limit. This document categorises every finding into four tiers:

- 🔴 **High priority** — will cause real maintenance pain or bugs; do these
- 🟡 **Medium priority** — worthwhile improvements to do when touching the area
- 🟢 **Nice but not necessary** — cleaner but low ROI; only if time allows
- ⚪ **Not recommended** — the cure is worse than the disease

**Nothing here should be implemented without explicit approval.**

---

## Raw Data

### Function size violations (52 functions > 80 lines)

| Tier | Lines | Branches | File | Function |
|------|------:|---------:|------|----------|
| 🟡 | 282 | 2 | `widgets/settings.py` | `SettingsDialog.__init__` |
| 🔴 | 263 | 30 | `services/tracking_service.py` | `_track` |
| 🟡 | 260 | 34 | `markit/.../response_parser.py` | `to_openlabel_tags` |
| ⚪ | 230 | 1 | `widgets/sidebar.py` | `Sidebar.__init__` |
| 🔴 | 202 | 21 | `widgets/overlay.py` | `paintEvent` |
| 🟡 | 199 | 28 | `markit/.../passes.py` | `process` (SmoothingPass) |
| 🟡 | 196 | 27 | `markit/.../engines.py` | `process_frame` (YoloEngine) |
| 🟢 | 194 | 20 | `utils/zoom.py` | `wire` |
| 🟡 | 183 | 40 | `markit/.../response_parser.py` | `_aggregate_results` |
| 🟢 | 169 | 17 | `utils/annotation_ops.py` | `wire` |
| ⚪ | 159 | 1 | `widgets/playback_controls.py` | `PlaybackControls.__init__` |
| 🔴 | 155 | 23 | `widgets/overlay.py` | `mouseMoveEvent` |
| 🟡 | 150 | 15 | `widgets/sidebar.py` | `_run_retrack_range` |
| ⚪ | 145 | 2 | `main_window.py` | `MainWindow.__init__` |
| 🟡 | 146 | 14 | `markit/.../passes.py` | `_smooth_object_trajectory` |
| 🔴 | 132 | 36 | `utils/annotation_ops.py` | `_on_overlay_context_menu` |
| 🟡 | 130 | 26 | `widgets/vlm_analysis_dialog.py` | `_apply_field_edit` |
| 🟡 | 125 | 12 | `markit/.../passes.py` | `_calculate_smoothed_rotation` |
| 🟢 | 120 | 19 | `markit/outputvideo.py` | `render_output_video` |
| 🟡 | 114 | 9 | `utils/annotation_ops.py` | `_start_tracking` |
| 🔴 | 111 | 12 | `widgets/overlay.py` | `hit_test` |
| ⚪ | 108 | 0 | `widgets/menu.py` | `Menu.__init__` |
| 🟢 | 105 | 3 | `widgets/cascade_dropdown.py` | `_setup_ui` |
| ⚪ | 100 | 0 | `widgets/overlay.py` | `Overlay.__init__` |

(+ 28 more functions in the 80–100 line range, mostly in markit; omitted for brevity)

### Oversized classes

| Lines | Methods | File | Class |
|------:|--------:|------|-------|
| 1404 | 63 | `widgets/sidebar.py` | `Sidebar` |
| 1238 | 55 | `widgets/overlay.py` | `Overlay` |
| 964 | 40 | `services/annotation_service.py` | `AnnotationService` |
| 599 | 9 | `services/tracking_service.py` | `TrackingService` |
| 586 | 18 | `widgets/settings.py` | `SettingsDialog` |

### Oversized modules

| Lines | File |
|------:|------|
| 2964 | `markit/markitlib/postprocessing/passes.py` |
| 1834 | `edit/.../frontend/utils/annotation_ops.py` |
| 1480 | `edit/.../frontend/widgets/sidebar.py` |
| 1468 | `markit/markitlib/processing/engines.py` |
| 1261 | `edit/.../frontend/widgets/overlay.py` |

### Code duplication (cascade function pairs)

| Pair | Lines | Shared |
|------|------:|-------:|
| `_apply_cascade_all_frames` / `_apply_cascade_next_frames` | 60 + 86 | ~53% |
| `_apply_cascade_delta_all_frames` / `_apply_cascade_delta_next_frames` | 68 + 87 | ~68% |
| `_apply_rotate90_all_frames` / `_apply_rotate90_next_frames` | 49 + 71 | ~55% |

---

## 🔴 High Priority

These are the items that will most improve AI-agent comprehension, reduce bug risk, and
pay for themselves quickly. Recommended to actually do.

### H1. Split `TrackingService._track` (263 lines, 30 branches)

This is the single highest-complexity function in edit/. It has three clear phases:
1. Initialisation (tracker setup, validation)
2. Per-frame loop (read frame → track → check guards → emit bbox)
3. Cleanup / return results

**Action**: Extract `_init_tracker()`, `_track_single_frame()`, and `_finalize_results()`.
The loop body in phase 2 becomes ~30 lines that calls out to the helpers.

**Risk**: Medium — this is core tracking logic with many edge cases. Must be tested before
and after with the existing test suite + manual verification on a real video.

### H2. Extract `paintEvent` helpers in `overlay.py` (202 lines, 21 branches)

`paintEvent` renders bboxes, handles, rotation indicators, drag previews, and tracking
overlays all in one method. Each section is already visually separated by comments.

**Action**: Extract `_paint_rotated_boxes()`, `_paint_resize_handles()`,
`_paint_drag_preview()`, `_paint_tracking_overlay()`. `paintEvent` becomes a ~30-line
dispatcher.

**Risk**: Low — painting is output-only with no state mutation. Visually verifiable.

### H3. Split `_on_overlay_context_menu` (132 lines, 36 branches)

Highest branch-to-line ratio in the codebase. Builds the entire right-click menu in one
function with deeply nested lambdas.

**Action**: Extract `_build_cascade_submenu()`, `_build_delete_submenu()`,
`_build_tracking_submenu()`, `_build_relationship_submenu()`. Each returns a QMenu or
list of QActions.

**Risk**: Low — pure menu construction, easily tested visually.

### H4. Split `mouseMoveEvent` in `overlay.py` (155 lines, 23 branches)

Dispatches on annotation mode (hover, drag-move, drag-resize, drag-rotate, pan) with
~30 lines per branch.

**Action**: Extract `_handle_move_drag()`, `_handle_resize_drag()`,
`_handle_rotate_drag()` as private methods on Overlay.

**Risk**: Low — each branch is independent.

### H5. Add tests for pure-logic functions (no Qt dependency needed)

These functions have zero test coverage and are fully deterministic:

| Function | File | Why |
|----------|------|-----|
| `_frames_to_ranges` | annotation_ops.py | Converts `[1,2,3,7,8]` → `[(1,3),(7,8)]` |
| `_compress_frame_ranges` | annotation_ops.py | Merges adjacent ranges |
| `interpolate_annotations` | tracking_service.py | Pure geometry interpolation |
| All `Command.do()/undo()` | undo/commands.py | Deterministic with mock gateway |

**Action**: Write unit tests. These are the easiest tests in the codebase to add and
they protect the most fragile logic (cascade frame ranges, undo/redo correctness).

**Risk**: Zero — additive, no code changes.

### H6. De-duplicate cascade function pairs

Three pairs of cascade functions share 53–68% of their code. The "all frames" and
"next N frames" variants differ only in how the frame range is computed.

**Action**: Extract a shared `_apply_cascade_core(frame_range, ...)` that both variants
call. Each public function becomes a thin wrapper that computes the frame range and
delegates.

**Risk**: Low-medium — these functions already have undo/redo, so the existing undo tests
validate correctness.

---

## 🟡 Medium Priority

Worthwhile when you're already editing these files. Not worth a dedicated refactoring PR.

### M1. Split `annotation_ops.py` into sub-modules (1834 lines)

The module mixes 6 unrelated concerns:
- Cascade logic (~500 lines)
- Tracking logic (~200 lines)
- Context menu (~170 lines)
- Delete operations (~140 lines)
- Relationship CRUD (~100 lines)
- Generic helpers (misc)

**Action**: Create sub-modules under `frontend/utils/annotation/`:
```
__init__.py               ← re-exports wire()
wire.py                   ← the wire() function
cascade_ops.py            ← cascade + delta cascade + rotate90
tracking_ops.py           ← _start_tracking, _start_tracking_to_frame
delete_ops.py             ← cascade_delete, directional delete
context_menu_ops.py       ← right-click menu builders
relationship_ops.py       ← relationship CRUD
helpers.py                ← _apply_geometry_update, _refresh_after_*, _frames_to_ranges
```

**Why medium, not high**: The functions inside are already reasonably sized (mostly <90
lines). The module-level split helps AI agents find things faster but doesn't reduce
per-function complexity.

### M2. Split `_start_tracking` (114 lines, 9 branches)

Mixes input validation, progress dialog setup, and actual tracking invocation.

**Action**: Extract `_validate_tracking_preconditions()` and `_create_tracking_dialog()`.

### M3. Resolve `render.py` ↔ `annotation_ops.py` coupling

`render.py` line 68 has a self-documented TODO acknowledging it calls back into model
code that annotation_ops also depends on. Not circular yet, but fragile.

**Action**: Move `_update_overlay_from_model()` out of render.py into a dedicated
`frame_sync.py` utility. Both render and annotation_ops import from it.

### M4. Fix inverse signal naming (annotation_ops.py TODOs at lines 63, 70)

Two signal/function pairs are named inversely, which confuses AI agents tracing signal
flow.

**Action**: Rename atomically (both emit site and connect site in same commit).

### M5. Split `_run_retrack_range` in sidebar.py (150 lines, 15 branches)

**Action**: Extract `_setup_retrack_dialog()` and `_execute_retrack_loop()`.

### M6. Extract `hit_test` coordinate math from overlay.py (111 lines, 12 branches)

`hit_test` does both coordinate transforms and hit-test geometry. The coordinate
transform portion is reusable and testable.

**Action**: Extract `_screen_to_image_coords()` and `_point_in_rotated_rect()`.

### M7. Type-safe signal payloads (overlay.py TODO at line 34)

Overlay signals currently emit raw tuples. A `BBoxMoveEvent(object_id, dx, dy)` dataclass
makes the contract explicit and catches mismatches at construction time.

**Action**: Define dataclasses in `frontend/types.py` and update emit/connect sites.

**Why medium**: Functional correctness is not affected — this is about catching bugs
during development rather than in production.

### M8. Split `markit/postprocessing/passes.py` into per-pass files (2964 lines)

7+ `Pass` classes in one file. Each class is already self-contained.

**Action**: Move each pass to its own file under `postprocessing/passes/`.
Add `__init__.py` with re-exports for backward compatibility.

### M9. Split `_track` guard logic into named methods

Even after H1, the per-frame body will still contain inline guard checks
(boundary, stationary, confidence). Each guard is 10-15 lines and would be
clearer as `_check_boundary_guard()`, `_check_stationary_guard()`.

---

## 🟢 Nice But Not Necessary

Clean-up items that make the code slightly nicer but have low practical impact.

### N1. Split `wire()` functions into themed sub-functions

`wire()` in zoom.py (194 lines), annotation_ops.py (169 lines), and playback.py (102
lines) are long but have **zero branching complexity** — they're sequential lists of
`.connect()` calls. Breaking them up adds function call overhead and indirection without
reducing complexity.

**Action if desired**: Group into `_wire_keyboard()`, `_wire_cascade()`, etc. But the
current flat list is actually easy for AI to parse since it's linear.

### N2. Split `SettingsDialog.__init__` (282 lines, 2 branches)

Despite being the longest function, it's 99% UI construction (`setFont`, `addWidget`,
etc.) with almost no logic. AI agents handle this fine because it's repetitive and linear.

**Action if desired**: Extract `_build_appearance_tab()`, `_build_tracking_tab()`, etc.

### N3. Split `Sidebar.__init__` (230 lines, 1 branch)

Same pattern as N2 — 99% UI layout, 1 branch. Readable as-is.

### N4. Split `PlaybackControls.__init__` (159 lines, 1 branch)

Pure button construction. Not confusing.

### N5. `cascade_dropdown._setup_ui` (105 lines, 3 branches)

Slightly over the limit but entirely UI button layout. Not worth the extraction overhead.

### N6. Split markit `_aggregate_results` (183 lines, 40 branches)

High branching but in markit, which is edited less frequently and has good test coverage.

### N7. Add pytest-qt tests for widget signals

Testing signal emission from `CascadeDropdown`, `Overlay`, etc. Provides some confidence
but has high setup cost (QApplication, event loop) for moderate value.

### N8. FrontendContext dataclass to reduce import fan-in

`main_window.py` has 21 internal imports. A `FrontendContext(controllers, states, ...)` 
dataclass could reduce this, but the current pattern works and import count isn't causing
real problems.

---

## ⚪ Not Recommended

These were considered and deliberately rejected.

### X1. Split `Sidebar` into sub-panel widgets

While `Sidebar` has 63 methods and 1404 lines, splitting it into `ObjectListPanel`,
`ConfidencePanel`, etc. would:
- Require threading signals through parent/child widget boundaries
- Break the existing signal wiring in annotation_ops.py and main_window.py
- Introduce a new inter-widget communication pattern that doesn't exist yet

The risk-to-reward ratio is poor. The class is large but each method is small and
well-named. AI agents can find methods by name without understanding the whole class.

### X2. Split `AnnotationService` into Read/Write services

The service has 40 methods but they share internal state (`self._openlabel_model`).
Splitting into ReadService/WriteService creates artificial boundaries and requires
either shared state or dependency injection patterns that add complexity.

### X3. Split `Overlay` into Painter/HitTester classes

Extracting an `OverlayPainter` class sounds clean but `paintEvent` needs access to
Overlay's state (selected object, annotation mode, zoom level, etc.). This means either
passing 10+ parameters or holding a reference to the parent, which makes the extraction
purely cosmetic.

**Better alternative**: Extract helper methods on Overlay itself (H2, H4, M6) rather
than creating new classes.

### X4. Split `MainWindow.__init__` (145 lines, 2 branches)

This is the composition root. It's *supposed* to be long — it wires the entire
application together. Splitting it just moves the wiring into helper methods that have
no standalone meaning.

### X5. Split `Menu.__init__` (108 lines, 0 branches)

Pure menu bar construction with zero logic. Splitting adds indirection with no benefit.

### X6. Convert `wire()` free-functions to `Wirer` classes

The `wire()` pattern works, is consistent across modules, and doesn't need OO wrapping.
Adding classes would increase boilerplate without improving testability (you still can't
unit-test signal wiring without a QApplication).

### X7. Split markit `engines.py` into per-engine files

The engines share internal helpers. Splitting them requires either duplicating the helpers
or creating a `_common.py` module. Low payoff for a module that's rarely edited.

---

## Recommended Execution Order

If approved, the recommended order maximises safety (safest first) and impact:

1. **H5** — Add tests for pure functions (zero risk, foundational)
2. **H6** — De-duplicate cascade pairs (protected by new tests from H5)
3. **H2** — Extract paintEvent helpers (output-only, visually verifiable)
4. **H3** — Split context menu builder (output-only, visually verifiable)
5. **H4** — Split mouseMoveEvent (independent branches)
6. **H1** — Split `_track` (highest impact but needs careful testing)
7. **M1** — Split annotation_ops.py into sub-modules (mechanical move)
8. **M4** — Fix inverse signal naming (quick, reduces confusion)

Each item is independently committable and should be its own PR (or at least its own
commit) to make rollback easy.
