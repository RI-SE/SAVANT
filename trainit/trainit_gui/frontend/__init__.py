"""Frontend UI components for trainit_gui."""

# Lazy imports to avoid requiring PyQt6 at module load time
__all__ = ["MainWindow"]


def __getattr__(name):
    """Lazy import to defer PyQt6 dependency."""
    if name == "MainWindow":
        from .main_window import MainWindow

        return MainWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
