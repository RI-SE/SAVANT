#!/usr/bin/env bash
# Example markit commands using TestVids/Saro_roundabout test data.
# Run from markit/examples/.
#
# Usage:
#   ./create_examples.sh            # JSON output only
#   ./create_examples.sh --videos   # JSON + annotated video output (recompressed with HEVC)
set -euo pipefail

INPUT_DIR="../../TestVids/Saro_roundabout"
INPUT="${INPUT_DIR}/Saro_roundabout.mp4"
MASK="${INPUT_DIR}/Saro_roundabout_mask.png"

VIDEOS=false
if [[ "${1:-}" == "--videos" ]]; then
  VIDEOS=true
fi

recompress() {
  ffmpeg -y -i "$1" -c:v hevc_nvenc -cq 28 -c:a copy "${1%.mp4}_tmp.mp4"
  mv "${1%.mp4}_tmp.mp4" "$1"
}

run_markit() {
  local dir="$1"
  shift
  mkdir -p "$dir"
  if $VIDEOS; then
    markit --output_video "$dir/$dir.mp4" "$@"
    recompress "$dir/$dir.mp4"
  else
    markit "$@"
  fi
}

# Yolo only
run_markit Saro_yolo --input "$INPUT" --output_json Saro_yolo/Saro_yolo.json --detection-method yolo
run_markit Saro_yolo_hk --input "$INPUT" --output_json Saro_yolo_hk/Saro_yolo_hk.json --detection-method yolo --housekeeping

# Optical flow only
run_markit Saro_of --input "$INPUT" --output_json Saro_of/Saro_of.json --detection-method optical_flow --debug-flow --exclusion-mask "$MASK"
run_markit Saro_of_hk --input "$INPUT" --output_json Saro_of_hk/Saro_of_hk.json --detection-method optical_flow --housekeeping --debug-flow --exclusion-mask "$MASK"

# Yolo and optical flow
run_markit Saro_both --input "$INPUT" --output_json Saro_both/Saro_both.json --detection-method both --housekeeping --debug-flow --exclusion-mask "$MASK"
run_markit Saro_both_hk --input "$INPUT" --output_json Saro_both_hk/Saro_both_hk.json --detection-method both --housekeeping --debug-flow --exclusion-mask "$MASK"
