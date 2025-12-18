# Savant App Developer Guide

This guide is for developers who want to contribute to, maintain, or extend the Savant desktop annotator. It covers the application's architecture, development workflows, and contribution guidelines.

---

## 1. Architecture Overview



The Savant App is built on a clean, layered architecture designed for maintainability and separation of concerns. It follows a pattern similar to Model-View-Controller, with a clear distinction between data, business logic, and the user interface.




```mermaid

graph TD

    subgraph Data Layer

        A1[JSON Files]

    end



    subgraph Service Layer (Model)

        B1[ProjectState Service]

        B2[Pydantic OpenLabel Models]

        A1 -- Loads/Saves --> B1

        B1 -- Manages --> B2

    end



    subgraph Controller Layer

        C1[Annotation Controller]

        C2[Project State Controller]

        C3[Video Controller]

        B1 -- Exposed via --> C2

    end



    subgraph Frontend (View)

        D1[MainWindow (PyQt6)]

        D2[Widgets]

        D3[UI Ops Files (_ops.py)]

        D4[Undo/Redo System]

        D5[Gateways]



        D1 -- Contains --> D2

        D1 -- Contains --> D4

        D3 -- Creates --> D4

        D4 -- Uses --> D5

    end

    

    subgraph User

        E1[User Interaction]

    end



    E1 -- Interacts with --> D2

    D2 -- Triggers --> D3

    D5 -- Talks to --> C1

    D5 -- Talks to --> C2

    D5 -- Talks to --> C3



    C1 -- Modifies --> B1

    C2 -- Reads from --> B1

    C3 -- Controls --> D1



```



1.  **Frontend (PyQt6)**: Manages the user interface, including widgets, menus, playback controls, and the annotation overlay. Located under `edit/src/edit/frontend`.
2.  **Controllers**: Translate UI events (like button clicks) into actions for the business logic layer. These act as a bridge between the UI and the services.
3.  **Services**: Contain the core business logic for features like annotation management, state persistence, video processing, and interpolation.
4.  **Models**: Pydantic models that define the data structures, primarily for the SAVANT OpenLabel format. This ensures data consistency and provides a clear schema.
5.  **Data**: The persisted data layer, consisting of SAVANT OpenLabel JSON files.



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
cd SAVANT/edit
uv sync --all-groups
```

### 2.3 Running Tests and Quality Checks

We use `pytest` for testing, `flake8` for linting, and `black` for code formatting. All are configured to be run via `uv`.

```bash
# Run the entire pytest suite
uv run pytest

# Run a specific test module with verbose output
uv run pytest edit/tests/unit/services/test_annotation_service.py -vv

# Run the linter
uv run flake8

# Format code
uv run black
```

---

## 3. Key Directories and Files

```
edit/
├── README.md              # Original quick-start guide
├── USER_README.md         # Guide for end-users
├── DEV_README.md          # This guide
├── pyproject.toml         # Project metadata and dependencies for uv
├── src/edit/
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

## 4. Architectural Components

-   **Controllers**: This layer is responsible for handling user interactions with the frontend. It acts as a bridge between the UI and the application's core logic, translating UI events (like button clicks) into specific actions. It ensures that user input is processed correctly and that the appropriate services are called to handle the request.

-   **Services**: This layer contains the core business logic of the application. It encapsulates the primary functionalities, such as managing annotations, handling video processing, and managing the application's state. The services are designed to be independent of the UI, which allows for easier testing and maintenance.

-   **Models**: The models define the data structures used throughout the application, primarily for handling data in the SAVANT OpenLabel format. By using Pydantic models, we ensure data consistency, validation, and a clear schema. This layer is crucial for maintaining data integrity as it flows through the application.

    > **Note:** Relationships in the data model are stored as individual entries. For instance, if car A tows both car B and car C, these would be two separate relationship entries rather than a single entry with an array of towed cars.

-   **Frontend**: The frontend is built with PyQt6 and is responsible for everything the user sees and interacts with. It is designed to be modular and maintainable, with a clear separation of concerns between different parts of the UI. The main components of the frontend are:
    -   **main_window.py**: This is the entry point for the UI. It initializes the main application window and brings together all the different UI components, such as the menu, video display, and sidebar.
    -   **widgets**: This directory contains all the custom UI elements used in the application. These are reusable components like the video display, playback controls, seek bar, and the sidebar, which encapsulates a significant portion of the application's interactivity.
    -   **states**: This directory manages the state of the UI. It holds data related to the frontend's current status, such as the selected theme, visibility of certain elements, and other UI-specific configurations.
    -   **utils**: This directory contains utility functions and helpers for the frontend. These can include anything from handling user input and managing settings to performing rendering operations. It is a collection of tools that support the functionality of the widgets and the main window.
        -   **ops files**: Within the `utils` directory, you'll find files with an `_ops.py` suffix, such as `annotation_ops.py` and `confidence_ops.py`. These files play a crucial role in connecting the UI to the application's core logic. They act as a dedicated layer for handling specific UI operations, encapsulating the logic required to translate user interactions into calls to the appropriate backend services. For example, `annotation_ops.py` manages everything related to creating, modifying, and deleting annotations, while `confidence_ops.py` handles the display of confidence scores. This separation of concerns helps to keep the UI components clean and focused on presentation, while the `ops` files manage the orchestration of complex UI-driven workflows.
        -   **undo**: A significant part of the application's robustness comes from its undo/redo functionality, located in the `utils/undo` directory. This system is built on the Command pattern and consists of several key components:
            -   **manager.py**: The `UndoRedoManager` is the core of the system. It maintains two stacks: one for commands that can be undone and one for commands that can be redone.
            -   **commands.py**: This file defines the `UndoableCommand` protocol and a set of concrete command classes. Each class represents a specific user action, such as creating a bounding box or deleting a frame tag, and contains the logic to both `do` and `undo` that action.
            -   **gateways.py**: To keep the commands decoupled from the application's controllers, we use a gateway pattern. This file defines gateway protocols that the commands use to interact with the application's data and services. It also contains concrete gateway implementations that adapt the application's controllers to these protocols.
            -   **snapshots.py**: These are simple data classes that capture the state of an object at a specific moment. Commands use these snapshots to save the state of the application before they make a change, so they can restore it during an `undo` operation.

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