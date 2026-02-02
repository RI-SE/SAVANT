# Postprocessing Pipeline - Technical Reference

This document describes the postprocessing passes available in markit and their algorithms. For usage instructions, see the main [README](../README.md#postprocessing).

## Overview

The postprocessing pipeline runs when `--housekeeping` is enabled. Passes execute in a fixed order, each transforming the OpenLabel data structure.

**Pipeline Order:**
```
GapDetection → GapFilling → DuplicateRemoval → StaticObjectRemoval →
FirstDetectionRefinement → BboxSmoothing → RotationAdjustment →
SuddenDetection → FrameInterval → AngleNormalization
```

---

## Pass Descriptions

### GapDetectionPass

**Purpose:** Identifies gaps in object tracking sequences where an object disappears for one or more frames then reappears.

**Algorithm:**
1. Build a map of frame indices for each object ID
2. For each object, sort frames and find consecutive frame pairs where `next_frame - current_frame > 1`
3. Record gap locations (start frame, end frame, gap size)

**Output:** Detection only - does not modify data. Logs warnings for each gap found.

**Statistics:**
- `objects_with_gaps`: Count of objects containing gaps
- `total_gaps_detected`: Total number of gaps across all objects

---

### GapFillingPass

**Purpose:** Fills gaps by linearly interpolating bbox parameters between the frames before and after each gap.

**Algorithm:**
1. For each gap, extract rbbox values from boundary frames
2. Calculate deltas: `delta = (after - before) / (gap_size + 1)`
3. For each missing frame, interpolate: `value = before + delta * step`
4. Create new frame entries with interpolated rbbox values

**Interpolated Parameters:** x, y, w, h, r (all five bbox parameters)

**Annotator:** Gap-filled frames are marked with `markit_housekeeping(gap)` and confidence `0.6666`.

**Statistics:**
- `gaps_filled`: Number of gaps filled
- `frames_added`: Total interpolated frames created

---

### DuplicateRemovalPass

**Purpose:** Removes duplicate objects that represent the same physical entity (e.g., when both YOLO and optical flow detect the same vehicle).

**Algorithm:**
1. For each pair of objects, find frames where both appear
2. Check shared-frame ratio: shared frames must be ≥ `min_shared_ratio` × shorter object's length
3. Calculate IoU for each shared frame
4. Objects are duplicates if:
   - Average IoU > `avg_iou_threshold` (default: 0.3)
   - Minimum IoU > `min_iou_threshold` (default: 0.2)
5. Delete the object with fewer frames (or lower average confidence if tied)

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `avg_iou_threshold` | 0.3 | Average IoU across shared frames to consider duplicate |
| `min_iou_threshold` | 0.2 | Minimum IoU in any shared frame |
| `min_shared_ratio` | 0.5 | Minimum ratio of shared frames to shorter object's total frames |

**Statistics:**
- `duplicate_pairs_found`: Number of duplicate pairs identified
- `objects_deleted`: Objects removed
- `frames_modified`: Frame entries cleaned up

---

### StaticObjectRemovalPass

**Purpose:** Removes or marks DynamicObject instances (vehicles, pedestrians, etc.) that don't move beyond a threshold. Useful for filtering parked cars or stationary pedestrians.

**Algorithm:**
1. For each object, look up its class in the ontology
2. Skip objects not classified as `DynamicObject`
3. Calculate movement: `delta_x = max(x) - min(x)`, `delta_y = max(y) - min(y)`
4. If both deltas are below threshold, remove or mark the object

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `static_threshold` | 20 | Maximum movement in pixels to be considered static |
| `mark_only` | false | If true, add `staticdynamic` annotation instead of removing |

**Marking:** When `mark_only=true`, adds `{"name": "staticdynamic", "val": [first_frame]}` to the object's vec data.

**Statistics:**
- `objects_checked`: DynamicObjects evaluated
- `objects_removed` or `objects_marked`: Count of static objects handled

---

### FirstDetectionRefinementPass

**Purpose:** Refines the initial rotation angle for newly detected objects by looking at their movement direction in subsequent frames.

**Problem Solved:** When an object first appears, its orientation may be ambiguous (especially for optical flow detections). This pass uses future movement to determine the correct heading.

**Algorithm:**
1. For each object, find its first frame
2. Look ahead up to `lookahead_frames` frames
3. Calculate movement vector from first frame to each future frame
4. Use the frame with maximum movement (if > `min_movement_pixels`) to determine direction
5. Adjust first frame's rotation to align with movement direction (snapping to nearest π)

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookahead_frames` | 5 | Number of future frames to examine |
| `min_movement_pixels` | 5.0 | Minimum movement to use for angle refinement |

**Statistics:**
- `objects_refined`: Objects with adjusted initial angles
- `objects_kept_base`: Objects kept at original angle (insufficient movement)

---

### BboxSmoothingPass

**Purpose:** Applies temporal smoothing to bbox position and size parameters (x, y, w, h) to reduce frame-to-frame jitter. Rotation is handled separately by RotationAdjustmentPass.

**Problem Solved:** Detection algorithms produce noisy bboxes with small variations between frames. This is especially noticeable for slow-moving or stationary objects where position jitter of several pixels and size fluctuations of ±10% occur.

**Algorithm:**

1. **Bidirectional EMA** (eliminates lag):
   - Forward pass: EMA from start to end
   - Backward pass: EMA from end to start
   - Average the two passes for each frame

2. **Size smoothing** (w, h): Uses fixed smoothing factor.

3. **Position smoothing** (x, y): Uses velocity-adaptive factor:
   - Calculate velocity: `v = sqrt((x[t] - x[t-1])² + (y[t] - y[t-1])²)`
   - `v < min_velocity`: full smoothing factor (maximum denoising for stationary/slow objects)
   - `v > max_velocity`: factor × 0.5 (reduced smoothing for fast objects)
   - Between: linear interpolation
   - Can be disabled with `--no-position-smoothing`

4. **Edge-Aware Size Handling:**
   - Detect when object center is near frame edge (within `edge_margin` pixels)
   - In "freeze" mode: use nearest interior (non-edge) size values
   - Prevents size jumps as objects enter/leave the frame

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `smoothing_factor` | 0.7 | Base EMA retention factor (0-1, higher = smoother) |
| `smooth_position` | true | Enable velocity-adaptive position smoothing (disable with `--no-position-smoothing`) |
| `min_velocity` | 2.0 | Below this velocity (px/frame), use maximum position smoothing |
| `max_velocity` | 20.0 | Above this velocity (px/frame), use minimum position smoothing |
| `edge_margin` | 100 | Pixels from frame edge for special handling |
| `edge_size_mode` | "freeze" | Edge size handling: "freeze" or "normal" |

**Annotator:** Smoothed frames are marked with `markit_housekeeping(smooth)`.

**Statistics:**
- `objects_smoothed`: Number of objects processed
- `frames_smoothed`: Total frames with smoothing applied
- `edge_frames_handled`: Frames where edge-aware size handling was used

**Note:** This pass does NOT smooth rotation angles. Rotation smoothing is handled by `RotationAdjustmentPass` using its own `temporal_smoothing` parameter.

---

### RotationAdjustmentPass

**Purpose:** Adjusts rotation values based on movement direction with temporal smoothing. Ensures objects face their direction of travel.

**Algorithm:**

1. **Movement Direction Calculation:**
   - Look backward 1-4 frames and forward 1-8 frames
   - Calculate angle from position deltas: `angle = atan2(delta_y, delta_x)`
   - Apply minimum movement threshold to filter noise
   - Weight forward frames higher than backward frames
   - Use circular averaging with `np.unwrap` to handle angle wraparound

2. **Aspect Ratio Handling:**
   - For elongated objects (aspect ratio > 1.5): align long axis with movement
   - For circular/square objects: use movement direction directly

3. **Temporal Smoothing (EMA):**
   ```
   smoothed_angle = factor * prev_angle + (1 - factor) * current_angle
   ```
   Handles angle wraparound by normalizing differences to [-π, π]

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `rotation_threshold` | 0.1 | Minimum angle difference (radians) to trigger adjustment |
| `min_movement_pixels` | 5.0 | Minimum movement to consider for rotation calculation |
| `rotation_smoothing` | 0.5 | EMA factor for rotation smoothing (0-1, higher = smoother) |

**Annotator:** Adjusted frames are marked with `markit_housekeeping(rot)` and confidence `0.8888`.

**Statistics:**
- `objects_processed`: Objects evaluated
- `rotations_adjusted`: Angles modified based on movement
- `rotations_kept`: Angles left unchanged (within threshold)
- `rotations_copied`: Angles propagated from previous frame (insufficient movement)

---

### SuddenPass

**Purpose:** Detects objects that suddenly appear or disappear far from frame edges, which may indicate tracking errors or unusual events.

**Algorithm:**
1. For each object, find first and last frames of appearance
2. Check if the object's bbox is near any frame edge (within `edge_distance`)
3. If object appears/disappears NOT near an edge, flag as sudden event

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `edge_distance` | 200 | Distance in pixels from edge to consider "near edge" |

**Output:** Adds `suddenappear` and/or `suddendisappear` vec entries to affected objects in the OpenLabel objects section.

**Statistics:**
- `objects_with_events`: Objects with sudden events
- `sudden_appear_count`: Sudden appearances detected
- `sudden_disappear_count`: Sudden disappearances detected

---

### FrameIntervalPass

**Purpose:** Calculates and adds `frame_intervals` to each object based on their actual frame appearances.

**Algorithm:**
1. For each object, collect all frame indices where it appears
2. Set `frame_intervals` to `[{frame_start: min, frame_end: max}]`

**Output:** Adds `frame_intervals` array to each object in the OpenLabel objects section.

**Statistics:**
- `intervals_added`: Objects with frame intervals added
- `intervals_skipped_existing`: Objects that already had intervals
- `intervals_skipped_no_frames`: Objects with no frame appearances

---

### AngleNormalizationPass

**Purpose:** Normalizes all rotation angles to the [0, 2π) range for OpenLabel output compliance.

**Algorithm:**
```python
normalized = angle % (2 * pi)
if normalized < 0:
    normalized += 2 * pi
```

**Note:** This is a mandatory final pass. Internal processing uses continuous unbounded angles to handle YOLO's π/2 ambiguity, but OpenLabel requires [0, 2π) range.

**Statistics:**
- `angles_normalized`: Number of angles adjusted

---

## Annotator Markers

Each pass that modifies data adds an annotator marker to track provenance:

| Marker | Pass | Confidence |
|--------|------|------------|
| `markit_housekeeping(gap)` | GapFillingPass | 0.6666 |
| `markit_housekeeping(smooth)` | BboxSmoothingPass | (unchanged) |
| `markit_housekeeping(rot)` | RotationAdjustmentPass | 0.8888 |

---

## Tuning Guidelines

### Reducing Position/Size Jitter
- Increase `smoothing_factor` in BboxSmoothingPass (try 0.4-0.5)
- For stationary vehicles, the velocity-adaptive smoothing automatically increases smoothing

### Reducing Rotation Jitter
- Increase `rotation_smoothing` in RotationAdjustmentPass (try 0.6-0.7)
- Increase `min_movement_pixels` to ignore small movements

### Edge Artifacts
- Increase `edge_margin` if objects show size jumps when entering/leaving frame
- Set `edge_size_mode="normal"` if you prefer consistent smoothing behavior everywhere

### Static Object Handling
- Decrease `static_threshold` to be more aggressive about removing parked vehicles
- Use `--static-mark` to annotate rather than remove (useful for review)
