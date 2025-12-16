# Savant App Developer Guide

This guide is for developers who want to contribute to, maintain, or extend the Savant desktop annotator. It covers the application's architecture, development workflows, and contribution guidelines.

---

## 1. Architecture Overview

The Savant App is built on a clean, layered architecture designed for maintainability and separation of concerns.

![Architecture Diagram](./Doc_images/Architecture.png)

1.  **Frontend (PyQt6)**: Manages the user interface, including widgets, menus, playback controls, and the annotation overlay. Located under `savant_app/src/savant_app/frontend`.
2.  **Controllers**: Translate UI events (like button clicks) into actions for the business logic layer. These act as a bridge between the UI and the services.
3.  **Services**: Contain the core business logic for features like annotation management, state persistence, video processing, and interpolation.
4.  **Models**: Pydantic models that define the data structures, primarily for the SAVANT OpenLabel format. This ensures data consistency and provides a clear schema.
5.  **Data**: The persisted data layer, consisting of SAVANT OpenLabel JSON files.

![Data Communication Diagram](./Doc_images/DataCommunication.png)

Data flows from the JSON files into the Pydantic models, which are managed by the services. The controllers orchestrate calls to the services, and the frontend presents the data to the user.

---

## 2. Project Setup and Tooling

### 2.1 Prerequisites
Ensure you have `uv` installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2.2 Installation
Clone the repository and install all dependencies, including those for development and testing.

```bash
git clone git@github.com:fwrise/SAVANT.git
cd SAVANT/savant_app
uv sync --all-groups
```

### 2.3 Running Tests and Quality Checks

We use `pytest` for testing, `flake8` for linting, and `black` for code formatting. All are configured to be run via `uv`.

```bash
# Run the entire pytest suite
uv run pytest

# Run a specific test module with verbose output
uv run pytest savant_app/tests/unit/services/test_annotation_service.py -vv

# Run the linter
uv run flake8

# Format code
uv run black
```

---

## 3. Key Directories and Files

```
savant_app/
├── README.md              # Original quick-start guide
├── USER_README.md         # Guide for end-users
├── DEV_README.md          # This guide
├── pyproject.toml         # Project metadata and dependencies for uv
├── src/savant_app/
│   ├── main.py            # Application entry point
│   ├── controllers/       # Controller layer
│   ├── services/          # Service layer (business logic)
│   ├── models/            # Pydantic data models (OpenLabel.py)
│   ├── frontend/          # PyQt6 UI code
│   ├── global_exception_handler.py # Top-level error handling
│   └── logger_config.py   # Logging setup
└── tests/
    └── unit/              # Unit tests for services, models, etc.
```

---

## 4. Service Layer Responsibilities

-   **AnnotationService**: Handles CRUD operations for bounding boxes, frame tags, relationships, and other metadata. It enforces business rules and raises domain-specific exceptions.
-   **ProjectState**: Manages the application's in-memory state, including loading, saving, and providing access to the OpenLabel data.
-   **VideoReader**: A wrapper around OpenCV's `VideoCapture` that provides a consistent interface for reading video frames and metadata.
-   **InterpolationService**: A stateless helper for performing linear interpolation of bounding box attributes between keyframes.

---

## 5. Error Handling and Logging

-   A custom exception hierarchy is defined in `services/exceptions.py` to distinguish between domain errors (expected) and internal errors (unexpected).
-   The `controllers/error_handler_middleware.py` decorator wraps controller methods to catch these exceptions, log them appropriately, and ensure the UI displays a user-friendly message.
-   The `global_exception_handler.py` acts as a final safety net, catching any unhandled exceptions from the Qt event loop to prevent silent crashes.
-   Logs are configured in `logger_config.py` to output to both the console and a rotating file, making debugging easier.

---

## 6. Contribution Workflow

1.  **Branch**: Create a new feature branch from the `main` branch.
2.  **Code**: Implement your changes, adhering to the existing architecture and coding style.
3.  **Test**: Add or update unit tests under `tests/unit` to cover your changes. Ensure all tests pass by running `uv run pytest`.
4.  **Lint & Format**: Run `uv run flake8` and `uv run black` to ensure your code meets our quality standards.
5.  **Document**: Update `USER_README.md`, this `DEV_README.md`, or other documentation if you are introducing significant changes.
6.  **Pull Request**: Open a pull request to `main`. In the description, summarize your changes, explain the "why," and list the steps you took to test your work. Include screenshots for any UI changes.
