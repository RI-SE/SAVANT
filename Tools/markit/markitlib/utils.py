"""
utils - Utility functions for markit

Contains helper functions for angle normalization and other common operations.
"""

from typing import Tuple
import numpy as np


def normalize_angle_to_pi(angle: float) -> float:
    """Normalize angle to [-π, π] range.

    Used for general angle differences and wraparound handling.

    Args:
        angle: Angle in radians

    Returns:
        Normalized angle in [-π, π] range
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return float(angle)


def normalize_angle_to_half_pi(angle: float) -> float:
    """Normalize angle to [-π/2, π/2] range.

    Used for OpenLabel rotation angles where width is always horizontal extent
    and rotation represents the angle of the width axis from horizontal.
    This ensures rotation stays within ±90° range.

    Args:
        angle: Angle in radians

    Returns:
        Normalized angle in [-π/2, π/2] range
    """
    while angle > np.pi / 2:
        angle -= np.pi
    while angle < -np.pi / 2:
        angle += np.pi
    return float(angle)


def normalize_bbox_representation(width: float, height: float, rotation: float) -> Tuple[float, float, float]:
    """Normalize oriented bounding box to canonical representation.

    Ensures width is always the more horizontal dimension by adjusting width/height
    and rotation so that the rotation angle stays in [-π/4, π/4] range where width
    represents the more horizontal extent.

    This maintains the OpenLabel convention where width should be closer to horizontal
    than vertical. If the rotation is outside [-45°, 45°], it means the height dimension
    has become more horizontal, so we swap dimensions and adjust the angle.

    Args:
        width: Current width value
        height: Current height value
        rotation: Current rotation angle in radians (should be in [-π/2, π/2])

    Returns:
        Tuple of (adjusted_width, adjusted_height, adjusted_rotation) where:
        - adjusted_rotation is in [-π/4, π/4] range
        - adjusted_width is the more horizontal dimension
        - adjusted_height is the more vertical dimension

    Example:
        >>> # Object rotated 80° (nearly vertical)
        >>> w, h, r = normalize_bbox_representation(100, 60, np.radians(80))
        >>> # Returns (60, 100, -0.175) meaning swap dimensions and angle is now -10°
    """
    # First normalize rotation to [-π/2, π/2] range
    rot = normalize_angle_to_half_pi(rotation)

    # Check if rotation magnitude exceeds π/4 (45°)
    # Beyond this threshold, the other dimension becomes more horizontal
    if rot > np.pi / 4:
        # Rotation is between 45° and 90° - height is now more horizontal
        # Swap dimensions and subtract 90° to bring angle back to canonical range
        return float(height), float(width), float(rot - np.pi / 2)
    elif rot < -np.pi / 4:
        # Rotation is between -45° and -90° - height is now more horizontal
        # Swap dimensions and add 90° to bring angle back to canonical range
        return float(height), float(width), float(rot + np.pi / 2)
    else:
        # Rotation is in [-45°, 45°] - width is already more horizontal
        # No adjustment needed
        return float(width), float(height), float(rot)
