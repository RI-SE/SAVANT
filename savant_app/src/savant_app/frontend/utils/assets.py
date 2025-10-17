from PyQt6.QtGui import QIcon
from savant_app.frontend.utils.asset_paths import ASSETS_DIR


def icon(name: str) -> QIcon:
    """Load an icon from /frontend/assets, fallback to empty icon."""
    p = ASSETS_DIR / name
    return QIcon(str(p)) if p.exists() else QIcon()
