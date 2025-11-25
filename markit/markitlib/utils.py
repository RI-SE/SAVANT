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


def rebase_angle_if_needed(angle: float) -> float:
    """Rebase angle by ±2π only if magnitude exceeds 2π.

    Maintains continuity by keeping rotations unlimited, only wrapping
    when necessary to prevent numerical issues with very large values.

    Args:
        angle: Angle in radians

    Returns:
        Rebased angle (still continuous, just shifted by 2π if needed)
    """
    if angle > 2 * np.pi:
        return angle - 2 * np.pi
    elif angle < -2 * np.pi:
        return angle + 2 * np.pi
    return angle


def normalize_angle_to_2pi_range(angle: float) -> float:
    """Normalize angle to [0, 2π) range for OpenLabel output.

    Args:
        angle: Angle in radians (can be any value)

    Returns:
        Normalized angle in [0, 2π) range
    """
    result = angle % (2 * np.pi)
    if result < 0:
        result += 2 * np.pi
    return float(result)


def find_continuous_angle(new_angle: float, previous_angle: float,
                          ambiguity_period: float = np.pi/2) -> float:
    """Find continuous rotation closest to previous angle.

    Used when converting from YOLO's ambiguous [0, π/2) format.
    YOLO gives the same angle for orientations differing by π/2, so we find which
    multiple of the ambiguity period maintains continuity with the previous frame.

    Args:
        new_angle: Raw angle from YOLO (in [0, π/2))
        previous_angle: Previous frame's continuous angle
        ambiguity_period: YOLO's π/2 ambiguity period

    Returns:
        Continuous angle closest to previous_angle

    Example:
        >>> # Previous angle was 1.8 rad (≈103°), new YOLO angle is 0.2 rad (≈11°)
        >>> # YOLO angle 0.2 could represent 0.2, 0.2+π/2, 0.2+π, 0.2+3π/2, etc.
        >>> find_continuous_angle(0.2, 1.8, np.pi/2)
        1.77...  # Returns ~1.77 (0.2 + π/2) as it's closest to 1.8
    """
    k = round((previous_angle - new_angle) / ambiguity_period)
    return new_angle + k * ambiguity_period
