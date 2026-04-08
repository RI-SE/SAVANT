# Savant Edit User Guide

Savant Edit is the desktop reviewer/annotator that lets you inspect OpenLabel data, adjust rotated bounding boxes, tag frames, and manage ontology-backed relationships. This guide walks through setting up the environment, understanding the new project layout, and using the editor’s workflows, settings, and error reporting.

---

## 1. Environment Setup

1. **Install `uv`** (if it is not already available):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Clone the repository and install the Edit dependencies**:
   ```bash
   git clone git@github.com:RI-SE/SAVANT.git
   cd SAVANT/edit
   uv sync --group edit --group dev
   ```
   The `edit` dependency group installs the PyQt6, OpenCV, numpy, rdflib, and appdirs packages that the GUI uses, while `dev` adds the optional tooling (flake8, black, pytest).
3. **Launch the application** from the package source directory so Python can import `edit`:
   ```bash
   # The tool is available as CLI commands
   edit
   
   # Or start from the edit folder
   cd /path/to/SAVANT/edit/src
   uv run python -m edit.main
   ```
   `uv run` reuses the synced environment, runs `edit/src/edit/main.py`, and initialises logging before the UI opens.

### 1.1 Running GitHub Release Builds

You can skip the source setup entirely by using the prebuilt PyInstaller packages attached to every GitHub release:

1. Download the asset that matches your platform (`edit-windows.exe` or `edit-linux`) from the latest release tag.
2. **Windows**: double-click `edit-windows.exe` or run it from PowerShell with `.\edit-windows.exe`. The binary is self-contained, so you can keep it anywhere that still has access to your project folders.
3. **Linux**: mark the file executable once (`chmod +x edit-linux`) and launch it with `./edit-linux`. Running it from a terminal keeps the log output visible if you need to troubleshoot.
4. Open or create projects the same way you would in the source build—your OpenLabel JSON, ontology files, and videos stay outside the executable alongside your project folders.

The release binaries ship with the bundled assets used by the UI. When a new tag is published the CI workflow rebuilds both executables, so checking GitHub releases is the quickest way to grab an updated version.

---

## 2. Feature List

- **Project onboarding**: folder scanner, guided video/config import, OpenLabel template generator, and automatic ontology fallback.
- **Annotator awareness**: login prompt, quick annotator switching, and per-project history so previous names autofill.
- **Video playback & navigation**: instant seek jumps, skip/play controls with FPS-aware playback, spacebar frame advance, go-to-frame (`Ctrl+G`), bookmarks with notes, and next/previous issue jumps.
- **Bounding box editing**: rotated boxes with drag handles, keyboard nudging, zoom/pan, rectangle zoom, bbox review cycling, cascade edits, undo/redo, right-click context actions, and Delete-to-remove.
- **Measure tool**: ephemeral pixel-distance measurements overlaid on the video (press `M` to toggle; not saved to the project).
- **Repeat last adjustment**: press `R` to re-apply the same geometry delta (position, size, rotation) from the previous frame edit to the currently selected object — useful when correcting many consecutive annotations with similar offsets.
- **Object management**: Active Objects list, object name/type editing, relationship viewer, and link-to-existing-ID workflow for both dynamic and static objects.
- **Interpolation & relationships**: **Fix Range** wizard (formerly "Interpolate") with three methods — *Linear interpolation*, *Re-track forward*, and *Re-track backward* — plus ontology-backed relationship creation, deletion, restoration, and overlay visualisation.
- **Tagging**: ontology-powered frame tags with configurable default ranges, object tag discovery, tag toggles that surface as markers and status notes, and Delete to remove tags.
- **Confidence controls**: configurable warning/error thresholds, seek-bar/overlay markers, sortable issue list with “Mark as resolved,” and playback issue summaries.
- **Saving & persistence**: quick save with validation, per-project settings snapshot (zoom, thresholds, tag toggles, bookmarks, ontology namespace), and automatic restoration on reopen.
- **Settings & theming**: zoom defaults, movement/rotation sensitivity, bbox zoom padding, frame history depth, ontology namespace, action interval offset, tag toggles, and warning/error visibility toggles.
- **Logging & error handling**: on-screen dialogs for user errors plus rotating log files for deeper troubleshooting.

---

## 3. Launching & Managing Projects

1. **Open the New or Load flow** from `File → New project`, `File → Load project`, or the matching toolbar buttons. The staged New Project dialog scans the chosen folder, reports whether exactly one video and one OpenLabel JSON are present, and lets you:
   - Copy in a video via *Select video…* (the file is validated and copied into the folder).
   - Import an existing OpenLabel file or ask the editor to generate a template (useful when you only have a video).
   - Rename the project before loading.
2. **Ontology resolution** happens automatically. The app looks for ontology references inside the OpenLabel config, searches relative to the JSON and project directories, and falls back to the bundled file if nothing can be found. If that file is missing you will see an error asking you to restore it.
3. **Finish loading** once both the video and OpenLabel files are ready. The app validates the JSON, refreshes `savant_project_config.json`, and then opens the video so the seek bar, FPS display, and frame count stay in sync.
4. **Annotator tracking**: the first time you open a project in a session, the app prompts you to identify yourself. Your choice is remembered for undo/redo command metadata, written back to the OpenLabel annotator fields (e.g., when resolving warnings), and stored into the per-project config so the name appears as a future suggestion. You can change users at any time from `Edit → Change annotator`.

---

## 4. Working in the Editor

### 4.1 Navigation & Playback
- Click anywhere on the seek bar to jump to that frame instantly. Warning/error markers, bookmark markers, and any enabled tag markers sit below the slider for quick reference.
- Use the playback bar to step one frame (`◀`, `▶`), skip ±30 frames, play/pause, or jump between the next/previous flagged issue. Press `Space` to advance one frame.
- `Ctrl+G` opens a go-to-frame dialog for direct frame number entry.
- **Bookmarks**: `Ctrl+B` toggles a bookmark on the current frame. `Ctrl+Shift+N` / `Ctrl+Shift+P` jump to the next/previous bookmark. Each bookmark can carry an editable text note. Open `View → Bookmarks` to manage all bookmarks and their notes.
- The right side of the control bar shows live center/size/rotation values for the active bounding box so you can see how edits affect it.

### 4.2 Bounding Boxes & Object Details
- `New BBox` lets you:
  - Create a **new** object type using the ontology labels.
  - Link a **bounding box to an existing ID**. Pick from recent dynamic objects or all static objects, or type an ID.
- The **Active Objects** list shows everything on the current frame. Selecting one highlights it, unlocks the **Object details** panel (rename, change type, view relationships), and synchronises the relationship list with the overlay. `Shift+click` an item to select it **and** zoom in to its bounding box.
- Overlay controls:
  - Drag handles/edges to resize, drag the box to move, drag the rotation handle to rotate.
  - Arrow keys nudge the box; hold `Shift` with ←/→ to rotate in the configured step. Hold `Ctrl+Shift` with ←/→ to rotate at 1/8 of that step for fine-tuning. When no box is selected and the view is zoomed in, arrow keys pan the view.
  - The rotation handle (cyan circle above the top edge) has a longer stem for a greater lever arm, giving finer mouse-rotation control. Hold `Ctrl` while dragging the rotation handle to slow down rotation by 8× for precision work.
  - `Tab` / `Shift+Tab` cycles to the next/previous bounding box and zooms in to it, letting you review each bbox in turn without manual panning. `Ctrl+0` zooms back out.
  - `Delete` removes the selected box (undo restores it).
  - **Repeat last adjustment**: press `R` to re-apply the geometry delta (dx, dy, dw, dh, d-rotation) recorded from your most recent edit to the same object. The delta is the compound change made during the last frame visit (not just a single step), so multiple nudge/rotate steps are accumulated and can be replayed with one key press.
  - **Zoom & pan**: `Ctrl` + mouse wheel zooms at the cursor position. Middle-click drag or `Ctrl` + left-click drag pans. `Z` toggles rectangle zoom mode (draw a rectangle to zoom into that area). `Ctrl+0` or the reset-view button resets to the default zoom. When selecting an object near the image border the view now keeps the bounding box fully visible rather than centering past the edge.
  - **Right-click context menu**: right-click a bbox for actions including delete, cascade delete, copy from previous frame, tracking, link IDs, and relationship management. Right-click empty space to copy a missing object's bbox from the previous frame.
- Cascade edits: select a box, open the cascade dropdown, and choose whether to apply size, rotation, or center changes to all future frames or only a frame range for that object.
- Undo/redo: `Ctrl+Z` / `Ctrl+Shift+Z` or **Edit → Undo / Redo** reverses most actions, including bbox edits, tag changes, Fix Range, linking, and relationship updates. The Edit menu items are enabled and disabled automatically as the undo/redo history changes.

### 4.3 Fix Range, Linking & Relationships
- **Fix Range** (formerly "Interpolate") corrects or fills bounding box annotations across a frame range for a selected object. Open it from **Edit → Fix Range** or the sidebar button, then choose a method:
  - **Linear interpolation** — generates intermediate boxes by linearly blending position, size, and rotation between the two boundary frames (start+1 … end-1). Overwrites any existing annotations in that range.
  - **Re-track forward / Re-track backward** — deletes the existing annotations in the range and re-runs the optical-flow tracker from one boundary frame toward the other, then linearly interpolates the rotation between the two anchors to suppress cumulative drift. Accepts either frame order; enter the range in whichever direction feels natural and the dialog normalises it.
- **Linking** adds a bounding box for an existing object. Static objects automatically gain boxes in any frames where they were missing; dynamic objects stay unique per frame.
- **Relationships** let you describe interactions (e.g., “vehicle follows person”). Choose the subject, relation, and object from the dialog. The editor limits the relationship to the frames where both objects exist and displays the link both in the overlay and the object details list.

### 4.4 Tags & Metadata
- **Frame tags**: select `New frame tag`, choose an Action label, and pick start/end frames. By default the dialog suggests a window centered on the current frame based on the "Action interval offset" setting. Tags appear in the sidebar list and can be removed (select row → `Delete`).
- **Object tags**: when you enable a tag in the Settings dialog, its frames are used as additional warning markers and show up in the playback issue panel when the corresponding object is visible.

### 4.5 VLM Analysis
If the OpenLabel file contains VLM-generated scene analysis (from markit `--vlm`), you can view and edit it via **View → VLM Analysis...**.

The dialog shows two sections:
- **Video-Level Tags** – Aggregated metadata for the entire video (editable)
- **Frame-Bound Contexts** – Time-bounded scene conditions with frame intervals (read-only)

To edit a tag:
1. Click **Edit** on the tag you want to modify
2. Change the text, number, or boolean fields as needed
3. Click **Save** to commit or **Cancel** to discard

When you save an edit, the system automatically:
- Prepends your annotator name to the annotator history
- Sets confidence to 1.0 (human ground truth)
- Preserves the original VLM annotation in the history

The `annotator` and `confidence` vec fields are auto-managed and shown as read-only during editing. Edits support undo/redo (`Ctrl+Z` / `Ctrl+Shift+Z`).

For details on confidence values and multi-annotator tracking, see the [Schema documentation](../schema/README.md#annotator-and-confidence-fields).

### 4.6 Measure Tool

Press **`M`** to toggle measure mode. The cursor changes to a crosshair and annotation editing is suspended.

- **Left-click** places the first point; a second left-click places the endpoint and completes one measurement segment, showing a dashed white line and the pixel distance at the midpoint.
- You can place multiple independent measurement pairs simultaneously — each completed pair stays visible until you exit.
- **Right-click** (or `Escape` with one point already placed) cancels the in-progress pair without clearing completed ones.
- **`Escape`** when no pair is in progress exits measure mode and clears all measurements.
- Distances are reported in video pixels (`"1234 px"`). Measurements are ephemeral and are never saved to the project.

### 4.7 Confidence Issues
- Confidence markers are drawn when a bounding box’s stored confidence value falls inside the Warning or Error range you configured. Warnings show amber icons, errors show red icons, and both ranges also appear under the seek bar.
- The **Confidence Issues** list in the sidebar shows every active warning/error near the current frame. Sort by frame or ID, multi-select rows, and right-click → *Mark as resolved* to confirm you have reviewed the issue.
- The issue panel in the playback controls mirrors the same data and adds any enabled tag notes. Use the `Next/Previous issue` buttons to jump along the timeline.

For details on how confidence values are generated and what they mean for different annotator types (YOLO, VLM, human), see the [Schema documentation](../schema/README.md#annotator-and-confidence-fields).

### 4.8 Keyboard Shortcuts
Open `Help → Keyboard Shortcuts` to see a table of all available keyboard shortcuts.

### 4.9 Saving Projects
- `Ctrl+S`, `File → Save project`, or the Save toolbar icon writes the OpenLabel JSON back to disk. Before saving, the app validates action tags to ensure each interval has a valid start/end.
- After saving annotations you are asked whether to store the current settings (zoom, warning ranges, tag toggles, namespace, bookmarks, etc.) inside `savant_project_config.json`. Choosing "Yes" means next time the project opens it will look exactly the same without further tweaks.

---

## 5. Settings & Preferences

Open `File → Settings` to fine-tune the experience:

- **Default zoom rate** – the rate at which the video is zoomed in to fit the video display area.
- **Frame history** – how many earlier frames the “Link to existing ID” dialog inspects while suggesting dynamic objects.
- **Movement/rotation sensitivity** – arrow-key increments for nudging and rotating. Rotation sensitivity can be set as low as 0.01 rad (~0.57°) for fine work; `Ctrl+Shift+←/→` always rotates at 1/8 of the configured step regardless of this setting.
- **BBox zoom padding** – how much context to show around a bounding box when zooming to it (via `Shift+click` or `Tab`). Higher values show more surrounding area.
- **Ontology namespace** – the namespace written when new entries are created.
- **Action interval offset** – extends `New frame tag` start/end defaults equally before and after the current frame.
- **Frame/Object tag toggles** – turn specific tags into seek-bar markers and playback notes.
- **Warning/Error ranges & visibility** – choose the confidence thresholds and whether each set of markers is shown. If both are visible, the ranges must not overlap.
- **Show warnings/errors** – quickly hide all warning markers without changing the numeric thresholds.

### 5.1 Settings Persistence
Saving a project gives you the option to store the session’s configuration in that folder’s `savant_project_config.json`, making the settings preferences specific to that project. Accepting the prompt means Savant Edit will restore the previous settings choices automatically the next time you open it, so each project remembers its own preferred setup.

---

## 6. Troubleshooting & Logs

- **Error popups** explain what went wrong (missing files, invalid frame ranges, etc.). Fix the input and retry; if you are unsure, undo the previous action and repeat the workflow slowly.
- **Application logs** are stored in two places:
  - `edit.log` in the directory where you launched the app (useful for quick checks).
  - A rotating log named `edit.log` under your OS’s application log folder (for example `~/.local/share/SAVANT/log/` on Linux). These files keep the last few sessions so you can share them when reporting bugs.
- **Crashes or unexpected behavior**: reopen Savant Edit, load the project, and review the log file for detailed messages. Most issues can be resolved by reinstalling dependencies (`uv sync --group edit --group dev`) or double-checking that the OpenLabel JSON still has the expected structure.

With these essentials you can install Savant Edit, open a project, review annotations, and keep everything in sync without needing to know the internal code layout.

---

## 7. For developers

For more information on contributing to Savant Edit, refer to the [Developer guide](DEV_README.md).

---
