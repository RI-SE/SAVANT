from pathlib import Path
import sys

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys._MEIPASS)
else:
    BASE_DIR = Path(__file__).resolve().parent.parent

ASSETS_DIR = BASE_DIR / "assets"


def asset_path(*parts: str) -> str:
    """File-relative path into /frontend/assets."""
    return str(ASSETS_DIR.joinpath(*parts))


def icon(name: str):
    """Load an icon from /frontend/assets, fallback to empty icon."""
    from PyQt6.QtGui import QIcon  # Lazy import to avoid headless CI failures
    p = ASSETS_DIR / name
    return QIcon(str(p)) if p.exists() else QIcon()
