"""
id_manager - Unified object ID management for all detection engines.

Provides sequential ID assignment starting from 0:
- ArUco markers: IDs 0 to N-1 (sorted by physical marker number)
- Visual markers: IDs N to M-1 (sorted by marker number)
- Dynamic detections (YOLO, Optical Flow): IDs M onwards
"""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ObjectIDManager:
    """Manages unified sequential object IDs across all detection engines.

    ArUco markers get IDs 0 to N-1 (sorted by physical marker ID).
    Visual markers get IDs N to M-1 (sorted by marker ID).
    Dynamic detections (YOLO, Optical Flow) continue from M onwards.
    """

    def __init__(self):
        # ArUco mapping: physical_marker_id -> sequential_id
        self._aruco_mapping: Dict[int, int] = {}

        # Visual marker mapping: marker_id -> sequential_id
        self._visual_marker_mapping: Dict[int, int] = {}

        # Dynamic ID remapping: (engine, native_id) -> sequential_id
        self._dynamic_mapping: Dict[Tuple[str, int], int] = {}

        # Next available ID for dynamic detections
        self._next_dynamic_id: int = 0

        # Flags to track initialization state
        self._aruco_finalized: bool = False
        self._visual_markers_finalized: bool = False

    def initialize_aruco_mapping(self, aruco_marker_ids: List[int]) -> Dict[int, int]:
        """Initialize ArUco ID mapping from list of physical marker IDs.

        Must be called before any dynamic ID assignment.

        Args:
            aruco_marker_ids: List of physical ArUco marker IDs from CSV

        Returns:
            Mapping dict: physical_marker_id -> sequential_id
        """
        if self._aruco_finalized:
            raise RuntimeError("ArUco mapping already initialized")

        # Sort marker IDs and assign sequential IDs starting from 0
        sorted_ids = sorted(aruco_marker_ids)
        self._aruco_mapping = {
            physical_id: seq_id
            for seq_id, physical_id in enumerate(sorted_ids)
        }

        # Dynamic IDs start after ArUco IDs
        self._next_dynamic_id = len(sorted_ids)
        self._aruco_finalized = True

        logger.info(f"ArUco ID mapping initialized: {len(sorted_ids)} markers, "
                    f"dynamic IDs start at {self._next_dynamic_id}")

        return self._aruco_mapping

    def initialize_visual_marker_mapping(self, marker_ids: List[int]) -> Dict[int, int]:
        """Initialize visual marker ID mapping from list of marker IDs.

        Must be called after initialize_aruco_mapping and before any dynamic ID assignment.

        Args:
            marker_ids: List of visual marker IDs from CSV

        Returns:
            Mapping dict: marker_id -> sequential_id
        """
        if not self._aruco_finalized:
            raise RuntimeError("ArUco mapping must be initialized first")
        if self._visual_markers_finalized:
            raise RuntimeError("Visual marker mapping already initialized")

        # Sort marker IDs and assign sequential IDs continuing from ArUco
        sorted_ids = sorted(marker_ids)
        start_id = self._next_dynamic_id
        self._visual_marker_mapping = {
            marker_id: start_id + seq_id
            for seq_id, marker_id in enumerate(sorted_ids)
        }

        # Dynamic IDs start after visual markers
        self._next_dynamic_id = start_id + len(sorted_ids)
        self._visual_markers_finalized = True

        logger.info(f"Visual marker ID mapping initialized: {len(sorted_ids)} markers "
                    f"(IDs {start_id} to {self._next_dynamic_id - 1}), "
                    f"dynamic IDs start at {self._next_dynamic_id}")

        return self._visual_marker_mapping

    def get_visual_marker_id(self, marker_id: int) -> int:
        """Get sequential ID for a visual marker.

        Args:
            marker_id: Visual marker ID from CSV

        Returns:
            Sequential object ID
        """
        if marker_id not in self._visual_marker_mapping:
            logger.warning(f"Visual marker {marker_id} not in mapping, "
                           "assigning dynamic ID")
            return self.get_dynamic_id('visual_marker_unknown', marker_id)

        return self._visual_marker_mapping[marker_id]

    def get_aruco_id(self, physical_marker_id: int) -> int:
        """Get sequential ID for an ArUco marker.

        Args:
            physical_marker_id: Physical ArUco marker ID

        Returns:
            Sequential object ID (0 to N-1)
        """
        if physical_marker_id not in self._aruco_mapping:
            logger.warning(f"ArUco marker {physical_marker_id} not in mapping, "
                           "assigning dynamic ID")
            return self.get_dynamic_id('aruco_unknown', physical_marker_id)

        return self._aruco_mapping[physical_marker_id]

    def get_dynamic_id(self, engine: str, native_id: int) -> int:
        """Get sequential ID for a dynamic detection (YOLO or Optical Flow).

        Remaps engine-native IDs to sequential IDs starting from N.

        Args:
            engine: Engine name ('yolo' or 'optical_flow')
            native_id: Engine's native tracking ID

        Returns:
            Sequential object ID (N onwards)
        """
        key = (engine, native_id)

        if key not in self._dynamic_mapping:
            self._dynamic_mapping[key] = self._next_dynamic_id
            self._next_dynamic_id += 1

        return self._dynamic_mapping[key]

    @property
    def aruco_mapping(self) -> Dict[int, int]:
        """Get the ArUco physical-to-sequential ID mapping."""
        return self._aruco_mapping.copy()

    @property
    def dynamic_start_id(self) -> int:
        """Get the starting ID for dynamic detections."""
        return self._next_dynamic_id

    @property
    def aruco_count(self) -> int:
        """Get the number of ArUco markers."""
        return len(self._aruco_mapping)

    @property
    def visual_marker_mapping(self) -> Dict[int, int]:
        """Get the visual marker ID mapping."""
        return self._visual_marker_mapping.copy()

    @property
    def visual_marker_count(self) -> int:
        """Get the number of visual markers."""
        return len(self._visual_marker_mapping)
