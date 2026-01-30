#!/usr/bin/env bash
# Example markit commands using TestVids/Saro_roundabout test data.
# Run from markit/examples/.
set -euo pipefail

INPUT_DIR="../../TestVids/Saro_roundabout"
INPUT="${INPUT_DIR}/Saro_roundabout.mp4"
MASK="${INPUT_DIR}/Saro_roundabout_mask.png"

# Yolo only
markit --input "$INPUT" --output_video Saro_yolo.mp4 --output_json Saro_yolo.json --detection-method yolo
markit --input "$INPUT" --output_video Saro_yolo_hk.mp4 --output_json Saro_yolo_hk.json --detection-method yolo --housekeeping

# Optical flow only
markit --input "$INPUT" --output_video Saro_of.mp4 --output_json Saro_of.json --detection-method optical_flow --debug-flow --exclusion-mask "$MASK"
markit --input "$INPUT" --output_video Saro_of_hk.mp4 --output_json Saro_of_hk.json --detection-method optical_flow --housekeeping --debug-flow --exclusion-mask "$MASK"

# Yolo and optical flow
markit --input "$INPUT" --output_video Saro_both.mp4 --output_json Saro_both.json --detection-method both --housekeeping --debug-flow --exclusion-mask "$MASK"
markit --input "$INPUT" --output_video Saro_both_hk.mp4 --output_json Saro_both_hk.json --detection-method both --housekeeping --debug-flow --exclusion-mask "$MASK"
