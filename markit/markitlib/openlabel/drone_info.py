"""
drone_info - FlightRecord parser and OpenLabel streams entry builder.

Parses a FlightRecord*.video_stats.json file and builds the streams block
for the OpenLabel output. Uses the sequence with the highest frame count.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class DroneInfo:
    """Extracted metadata from a FlightRecord video_stats file."""

    drone_type: str
    lens_type: str
    recording_start: str
    recording_end: str
    latitude_mean_deg: float
    longitude_mean_deg: float
    altitude_msl_mean_m: float
    height_agl_mean_m: float
    gimbal_pitch_mean_deg: float


def parse_drone_info(path: str) -> DroneInfo:
    """Parse a FlightRecord*.video_stats.json and return metadata from the longest sequence."""
    with open(path) as f:
        data = json.load(f)

    sequences = data.get("sequences", [])
    if not sequences:
        raise ValueError(f"No sequences found in {path}")

    longest = max(sequences, key=lambda s: s.get("frame_count", 0))

    osd = longest["stats"]["osd"]
    gimbal = longest["stats"]["gimbal"]
    time_range = longest["time_range"]

    return DroneInfo(
        drone_type=data.get("drone_type", "unknown"),
        lens_type=data.get("lens_type", "unknown"),
        recording_start=time_range["start"],
        recording_end=time_range["end"],
        latitude_mean_deg=osd["latitude"]["mean"],
        longitude_mean_deg=osd["longitude"]["mean"],
        altitude_msl_mean_m=osd["altitude_msl"]["mean"],
        height_agl_mean_m=osd["height_agl"]["mean"],
        gimbal_pitch_mean_deg=gimbal["pitch"]["mean"],
    )


def build_streams_entry(
    video_path: str, fps: float, info: DroneInfo
) -> Dict[str, Any]:
    """Build the OpenLabel streams dict for a drone camera recording."""
    description = f"{info.drone_type} {info.lens_type} lens"
    return {
        "drone_camera": {
            "type": "camera",
            "description": description,
            "uri": os.path.basename(video_path),
            "stream_properties": {
                "sync": {"frame_rate": fps},
                "drone_type": info.drone_type,
                "lens_type": info.lens_type,
                "recording_start": info.recording_start,
                "recording_end": info.recording_end,
                "position": {
                    "latitude_mean_deg": round(info.latitude_mean_deg, 6),
                    "longitude_mean_deg": round(info.longitude_mean_deg, 6),
                    "altitude_msl_mean_m": round(info.altitude_msl_mean_m, 2),
                    "height_agl_mean_m": round(info.height_agl_mean_m, 2),
                },
                "gimbal_pitch_mean_deg": round(info.gimbal_pitch_mean_deg, 1),
            },
        }
    }
