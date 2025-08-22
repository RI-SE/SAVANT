# frontend/utils/assets.py
from pathlib import Path
from PyQt6.QtGui import QIcon

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def asset_path(*parts: str) -> str:
    """File-relative path into /frontend/assets."""
    return str(ASSETS_DIR.joinpath(*parts))


def icon(name: str) -> QIcon:
    """Load an icon from /frontend/assets, fallback to empty icon."""
    p = ASSETS_DIR / name
    return QIcon(str(p)) if p.exists() else QIcon()
