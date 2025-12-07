"""UI state management for trainit_gui."""

# Lazy imports to avoid requiring PyQt6 at module load time
__all__ = ['AppState']


def __getattr__(name):
    """Lazy import to defer PyQt6 dependency."""
    if name == 'AppState':
        from .app_state import AppState
        return AppState
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
