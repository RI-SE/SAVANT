# edit/frontend/utils/_annotation_pure.py
"""Pure-logic helpers with no external dependencies (safe for headless import)."""
from __future__ import annotations


def frames_to_ranges(frames: list[int]) -> str:
    """Convert a list of frame numbers into contiguous ranges as a string."""
    if not frames:
        return ""
    ranges = []
    start = prev = frames[0]
    for f in frames[1:]:
        if f == prev + 1:
            prev = f
        else:
            ranges.append((start, prev))
            start = prev = f
    ranges.append((start, prev))
    range_strs = [f"{s}-{e}" if s != e else f"{s}" for s, e in ranges]
    return ", ".join(range_strs)


def cascade_property_description(center_x, center_y, width, height, rotation) -> str:
    """Build a human-readable list of properties being cascaded."""
    parts = []
    if center_x is not None or center_y is not None:
        parts.append("position")
    if width is not None or height is not None:
        parts.append("size")
    if rotation is not None:
        parts.append("rotation")
    return ", ".join(parts) or "properties"
