#!/usr/bin/env bash
# Example markit commands using TestVids/Saro_roundabout test data.
# Run from markit/examples/.
set -euo pipefail

INPUT_DIR="../../TestVids/Saro_roundabout"
INPUT="${INPUT_DIR}/Saro_roundabout.mp4"
MASK="${INPUT_DIR}/Saro_roundabout_mask.png"

recompress() {
  ffmpeg -y -i "$1" -c:v hevc_nvenc -cq 28 -c:a copy "${1%.mp4}_tmp.mp4"
  mv "${1%.mp4}_tmp.mp4" "$1"
}

# Yolo only
mkdir -p Saro_yolo
markit --input "$INPUT" --output_video Saro_yolo/Saro_yolo.mp4 --output_json Saro_yolo/Saro_yolo.json --detection-method yolo
recompress Saro_yolo/Saro_yolo.mp4

mkdir -p Saro_yolo_hk
markit --input "$INPUT" --output_video Saro_yolo_hk/Saro_yolo_hk.mp4 --output_json Saro_yolo_hk/Saro_yolo_hk.json --detection-method yolo --housekeeping
recompress Saro_yolo_hk/Saro_yolo_hk.mp4

# Optical flow only
mkdir -p Saro_of
markit --input "$INPUT" --output_video Saro_of/Saro_of.mp4 --output_json Saro_of/Saro_of.json --detection-method optical_flow --debug-flow --exclusion-mask "$MASK"
recompress Saro_of/Saro_of.mp4

mkdir -p Saro_of_hk
markit --input "$INPUT" --output_video Saro_of_hk/Saro_of_hk.mp4 --output_json Saro_of_hk/Saro_of_hk.json --detection-method optical_flow --housekeeping --debug-flow --exclusion-mask "$MASK"
recompress Saro_of_hk/Saro_of_hk.mp4

# Yolo and optical flow
mkdir -p Saro_both
markit --input "$INPUT" --output_video Saro_both/Saro_both.mp4 --output_json Saro_both/Saro_both.json --detection-method both --housekeeping --debug-flow --exclusion-mask "$MASK"
recompress Saro_both/Saro_both.mp4

mkdir -p Saro_both_hk
markit --input "$INPUT" --output_video Saro_both_hk/Saro_both_hk.mp4 --output_json Saro_both_hk/Saro_both_hk.json --detection-method both --housekeeping --debug-flow --exclusion-mask "$MASK"
recompress Saro_both_hk/Saro_both_hk.mp4
