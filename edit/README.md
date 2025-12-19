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
- **Video playback & navigation**: instant seek jumps, skip/play controls with FPS-aware playback, and next/previous issue jumps.
- **Bounding box editing**: rotated boxes with drag handles, keyboard nudging, zoom/pan, cascade edits, undo/redo, and Delete-to-remove.
- **Object management**: Active Objects list, object name/type editing, relationship viewer, and link-to-existing-ID workflow for both dynamic and static objects.
- **Interpolation & relationships**: frame-range interpolation wizard plus ontology-backed relationship creation, deletion, restoration, and overlay visualisation.
- **Tagging**: ontology-powered frame tags with configurable default ranges, object tag discovery, tag toggles that surface as markers and status notes, and Delete to remove tags.
- **Confidence controls**: configurable warning/error thresholds, seek-bar/overlay markers, sortable issue list with “Mark as resolved,” and playback issue summaries.
- **Saving & persistence**: quick save with validation, per-project settings snapshot (zoom, thresholds, tag toggles, ontology namespace), and automatic restoration on reopen.
- **Settings & theming**: zoom defaults, movement/rotation sensitivity, frame history depth, ontology namespace, action interval offset, tag toggles, and warning/error visibility toggles.
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
- Click anywhere on the seek bar to jump to that frame instantly. Warning/error markers (and any enabled tag markers) sit below the slider for quick reference.
- Use the playback bar to step one frame (`◀`, `▶`), skip ±30 frames, play/pause, or jump between the next/previous flagged issue.
- The right side of the control bar shows live center/size/rotation values for the active bounding box so you can see how edits affect it.

### 4.2 Bounding Boxes & Object Details
- `New BBox` lets you:
  - Create a **new** object type using the ontology labels.
  - Link a **bounding box to an existing ID**. Pick from recent dynamic objects or all static objects, or type an ID.
- The **Active Objects** list shows everything on the current frame. Selecting one highlights it, unlocks the **Object details** panel (rename, change type, view relationships), and synchronises the relationship list with the overlay.
- Overlay controls:
  - Drag handles/edges to resize, drag the box to move, drag the rotation handle to rotate.
  - Arrow keys nudge the box; hold `Shift` with ←/→ to rotate in small steps.
  - `Delete` removes the selected box (undo restores it). `Ctrl` + mouse wheel zooms; `Ctrl` + drag pans when zoomed. `Ctrl+0` resets to the default zoom.
- Cascade edits: select a box, open the cascade dropdown, and choose whether to apply size, rotation, or center changes to all future frames or only a frame range for that object.
- Undo/redo: `Ctrl+Z` / `Ctrl+Shift+Z` (or the Edit menu) reverses most actions, including bbox edits, tag changes, interpolation, linking, and relationship updates.

### 4.3 Interpolation, Linking & Relationships
- **Interpolation** fills gaps between two frames of the same object. Pick the object plus start/end frames (with at least one frame between them) and the tool generates intermediate boxes.
- **Linking** adds a bounding box for an existing object. Static objects automatically gain boxes in any frames where they were missing; dynamic objects stay unique per frame.
- **Relationships** let you describe interactions (e.g., “vehicle follows person”). Choose the subject, relation, and object from the dialog. The editor limits the relationship to the frames where both objects exist and displays the link both in the overlay and the object details list.

### 4.4 Tags & Metadata
- **Frame tags**: select `New frame tag`, choose an Action label, and pick start/end frames. By default the dialog suggests a window centered on the current frame based on the “Action interval offset” setting. Tags appear in the sidebar list and can be removed (select row → `Delete`).
- **Object tags**: when you enable a tag in the Settings dialog, its frames are used as additional warning markers and show up in the playback issue panel when the corresponding object is visible.

### 4.5 Confidence Issues
- Confidence markers are drawn when a bounding box’s stored confidence value falls inside the Warning or Error range you configured. Warnings show amber icons, errors show red icons, and both ranges also appear under the seek bar.
- The **Confidence Issues** list in the sidebar shows every active warning/error near the current frame. Sort by frame or ID, multi-select rows, and right-click → *Mark as resolved* to confirm you have reviewed the issue.
- The issue panel in the playback controls mirrors the same data and adds any enabled tag notes. Use the `Next/Previous issue` buttons to jump along the timeline.

### 4.6 Saving Projects
- `Ctrl+S`, `File → Save project`, or the Save toolbar icon writes the OpenLabel JSON back to disk. Before saving, the app validates action tags to ensure each interval has a valid start/end.
- After saving annotations you are asked whether to store the current settings (zoom, warning ranges, tag toggles, namespace, etc.) inside `savant_project_config.json`. Choosing “Yes” means next time the project opens it will look exactly the same without further tweaks.

---

## 5. Settings & Preferences

Open `File → Settings` to fine-tune the experience:

- **Default zoom rate** – the rate at which the video is zoomed in to fit the video display area.
- **Frame history** – how many earlier frames the “Link to existing ID” dialog inspects while suggesting dynamic objects.
- **Movement/rotation sensitivity** – arrow-key increments for nudging and rotating.
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
