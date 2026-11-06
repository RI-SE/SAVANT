"""Per-box features for the learned box-uncertainty model (development.md, "Learned per-box uncertainty").

Shared by trainit (study and fitting) and, later, markit (run time). Boxes are [cx, cy, w, h, r] in px and rad.
Track features come from the boxes in the OpenLABEL file; markit's smoothing hides part of the detector jitter,
so a study and its later use must take the tracks from runs with the same housekeeping options.
"""

import math
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

TRACK_WINDOW = 5  # frames before and after the box used for track features
PATCH_MARGIN = 0.15  # share of the box size added around the patch for image features
CANNY_THRESHOLDS = (50, 150)

BOX_FEATURES = ("size", "aspect")
POSITION_FEATURES = ("radial", "edge_dist")
TRACK_FEATURES = ("speed", "jitter", "w_std", "h_std", "r_std", "track_age", "track_left")
PATCH_FEATURES = ("brightness", "contrast", "edge_density", "sharpness")
FLOW_FEATURES = ("flow_mag", "flow_std")


def box_features(box: Sequence[float]) -> Dict[str, float]:
    """Box size (sqrt of the area, px) and aspect ratio (long/short side)."""
    w, h = max(box[2], 1e-6), max(box[3], 1e-6)
    return {"size": math.sqrt(w * h), "aspect": max(w, h) / min(w, h)}


def position_features(box: Sequence[float], width: int, height: int) -> Dict[str, float]:
    """Distance of the box centre from the image centre (0 centre, 1 corner) and from the nearest edge (px)."""
    dx, dy = box[0] - width / 2, box[1] - height / 2
    return {"radial": math.hypot(dx, dy) / math.hypot(width / 2, height / 2),
            "edge_dist": float(min(box[0], box[1], width - box[0], height - box[1]))}


def track_index(openlabel: dict) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Per object: (sorted frame numbers, boxes N x 5) from the frames of an OpenLABEL dict."""
    tracks: Dict[str, List[Tuple[int, List[float]]]] = {}
    for frame_key, frame in (openlabel.get("frames") or {}).items():
        for obj_id, frame_obj in (frame.get("objects") or {}).items():
            shapes = (frame_obj.get("object_data") or {}).get("rbbox") or []
            shape = next((g.get("val") for g in shapes if g.get("name") == "shape"), None)
            if shape and len(shape) >= 5:
                tracks.setdefault(obj_id, []).append((int(frame_key), [float(v) for v in shape[:5]]))
    index = {}
    for obj_id, items in tracks.items():
        items.sort(key=lambda item: item[0])
        index[obj_id] = (np.array([f for f, _ in items]), np.array([b for _, b in items]))
    return index


def _angle_std(angles: np.ndarray) -> float:
    """Std of box angles, which are equivalent modulo pi (doubled-angle circular std)."""
    doubled = 2 * angles
    length = min(1.0, math.hypot(np.cos(doubled).mean(), np.sin(doubled).mean()))
    return math.sqrt(max(-2 * math.log(max(length, 1e-12)), 0.0)) / 2


def track_features(frames: np.ndarray, boxes: np.ndarray, frame: int, window: int = TRACK_WINDOW) -> Dict[str, float]:
    """Motion and stability of a track around one frame.

    speed: median centre displacement per frame (px); jitter: std of the centre's second differences (px);
    w_std/h_std/r_std: variation of size and angle in the window; track_age/track_left: frames since the first
    and until the last box of the track.
    """
    near = np.abs(frames - frame) <= window
    f, b = frames[near], boxes[near]
    features = {"track_age": float(frame - frames[0]), "track_left": float(frames[-1] - frame)}
    if len(f) < 2:
        return {**features, **dict.fromkeys(("speed", "jitter", "w_std", "h_std", "r_std"), math.nan)}
    steps = np.diff(f).astype(float)
    velocity = np.diff(b[:, :2], axis=0) / steps[:, None]
    jitter = float(np.linalg.norm(np.diff(velocity, axis=0), axis=1).std()) if len(velocity) > 2 else math.nan
    features.update({
        "speed": float(np.median(np.linalg.norm(velocity, axis=1))), "jitter": jitter,
        "w_std": float(b[:, 2].std()), "h_std": float(b[:, 3].std()), "r_std": _angle_std(b[:, 4]),
    })
    return features


def patch_bounds(box: Sequence[float], shape: Tuple[int, int], margin: float = PATCH_MARGIN):
    """Clipped axis-aligned pixel bounds (x0, y0, x1, y1) around a rotated box, or None if outside the image."""
    corners = cv2.boxPoints(((box[0], box[1]), (box[2], box[3]), math.degrees(box[4])))
    pad = margin * math.sqrt(max(box[2] * box[3], 0.0))
    x0, y0 = np.floor(corners.min(axis=0) - pad).astype(int)
    x1, y1 = np.ceil(corners.max(axis=0) + pad).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, shape[1]), min(y1, shape[0])
    return (x0, y0, x1, y1) if x1 - x0 >= 3 and y1 - y0 >= 3 else None


def patch_features(gray: np.ndarray, box: Sequence[float]) -> Dict[str, float]:
    """Brightness, contrast (std), Canny edge density and sharpness (variance of the Laplacian) around a box."""
    bounds = patch_bounds(box, gray.shape)
    if bounds is None:
        return dict.fromkeys(PATCH_FEATURES, math.nan)
    x0, y0, x1, y1 = bounds
    patch = gray[y0:y1, x0:x1]
    return {
        "brightness": float(patch.mean()), "contrast": float(patch.std()),
        "edge_density": float((cv2.Canny(patch, *CANNY_THRESHOLDS) > 0).mean()),
        "sharpness": float(cv2.Laplacian(patch, cv2.CV_64F).var()),
    }


def flow_features(previous: Optional[np.ndarray], gray: np.ndarray, box: Sequence[float]) -> Dict[str, float]:
    """Mean and std of the dense optical-flow magnitude (px/frame, Farneback) around a box."""
    bounds = None if previous is None else patch_bounds(box, gray.shape)
    if bounds is None:
        return dict.fromkeys(FLOW_FEATURES, math.nan)
    x0, y0, x1, y1 = bounds
    flow = cv2.calcOpticalFlowFarneback(previous[y0:y1, x0:x1], gray[y0:y1, x0:x1], None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude = np.linalg.norm(flow, axis=2)
    return {"flow_mag": float(magnitude.mean()), "flow_std": float(magnitude.std())}


def video_size(video_path) -> Tuple[int, int]:
    """(width, height) of a video."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()


def iter_gray_frames(video_path, frames: Sequence[int]) -> Iterator[Tuple[int, Optional[np.ndarray], np.ndarray]]:
    """(frame, previous grey frame or None, grey frame) for the wanted frames, reading the video sequentially.

    Sequential reading is slower than seeking but frame-exact for all codecs.
    """
    wanted = sorted(set(int(f) for f in frames))
    capture = cv2.VideoCapture(str(video_path))
    try:
        index, previous, position = 0, None, 0
        while position < len(wanted):
            ok, image = capture.read()
            if not ok:
                break
            if index + 1 == wanted[position] or index == wanted[position]:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if index == wanted[position]:
                    yield index, previous, gray
                    position += 1
                previous = gray
            else:
                previous = None
            index += 1
    finally:
        capture.release()
