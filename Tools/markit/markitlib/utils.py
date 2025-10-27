"""
utils - Utility functions for markit

Contains helper functions for angle normalization and other common operations.
"""

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
