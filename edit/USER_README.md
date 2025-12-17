# Savant Edit User Guide

Welcome to the Savant desktop annotator! This guide provides everything you need to get started with installing, running, and using the application to review and edit video annotations.

---

## 1. Prerequisites and Installation

To get started, you'll need to install the `uv` package manager and clone the project repository.

1.  **Install `uv`** (if you don't have it already):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Clone the SAVANT repository** and navigate to the application directory:
    ```bash
    git clone git@github.com:fwrise/SAVANT.git
    cd SAVANT/edit
    ```

3.  **Install dependencies** (like PyQt6, OpenCV, and Pydantic) using `uv`:
    ```bash
    uv sync --all-groups
    ```

---

## 2. Launching the Application

Once the installation is complete, you can launch the app from within the `edit` directory.

1.  Navigate to the source directory:
    ```bash
    cd src/edit
    ```
2.  Run the application using the packaged entry point:
    ```bash
    uv run python -m edit.main
    ```
Upon launch, the application sets up logging, creates a shared project state, and opens the main window.

---

## 3. Typical Workflow and Features

The Savant App provides a comprehensive set of tools for video annotation.

1.  **Load Data**: Use the **File** menu to load a video and its corresponding SAVANT OpenLabel JSON configuration.
2.  **Navigate and Review**: Use the seek bar, playback controls, and keyboard shortcuts (like Undo/Redo with `Ctrl+Z`/`Ctrl+Y`) to scrub through frames and visualize annotations. The overlay distinguishes between interpolated and manually created bounding boxes.
3.  **Create and Edit Annotations**:
    *   **Create New Bounding Box**: Draw a new rotated bounding box for an object and specify its type.
    *   **Add BBox to Existing Object**: Add a new keyframe annotation to an existing object track.
    *   **Cascade Edit**: Apply changes in width, height, or rotation from a starting frame across all subsequent frames for an object.
    *   **Move/Resize**: Click and drag handles directly on the video overlay to adjust a bounding box.
4.  **Manage Tags and Metadata**:
    *   **Frame Tags**: Apply ontology-defined "Action" labels to time intervals.
    *   **Object Metadata**: Edit the name and type of tracked objects from the sidebar.
5.  **Interpolate**: Automatically generate bounding boxes between two keyframes for smoother tracking.
6.  **Define Relationships**: Link two objects together with a predefined relationship from the ontology (e.g., "follows"). The app automatically scopes the relationship to the frames where both objects are present.
7.  **Undo/Redo**: Correct mistakes easily, as nearly every action is tracked in the undo history.
8.  **Save Your Work**: Use **Quick Save** (`Ctrl+S`) or **Save As** to write your changes back to an OpenLabel JSON file.

---

## 4. Settings and Configuration

*   **Ontology**: The application uses a TTL file for its ontology. You can select the ontology file via the **Settings** dialog. This file populates the available object types, frame tags, and relationship labels.
*   **Preferences**: Adjust display settings, confidence thresholds for warnings, and interaction sensitivity from the **Settings** dialog. These preferences are stored locally on your machine.

---

## 5. Logging and Error Handling

*   The application writes logs to a `edit.log` file in the `edit` directory and to a system-specific log directory.
*   If you encounter an error, the application will display a user-friendly dialog. For unexpected issues, a generic error message is shown, and detailed information is recorded in the log file for debugging.