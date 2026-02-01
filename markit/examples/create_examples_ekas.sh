#!/usr/bin/env bash
# Example markit commands using TestVids/Ekas_T test data.
# Run from markit/examples/.
#
# Usage:
#   ./create_examples_ekas.sh            # JSON output only
#   ./create_examples_ekas.sh --videos   # JSON + annotated video output (recompressed with HEVC)
set -euo pipefail

INPUT_DIR="../../TestVids/Ekas_T"
INPUT="${INPUT_DIR}/Ekas_T.mp4"
MASK="${INPUT_DIR}/Ekas_T_mask.png"

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
run_markit Ekas_yolo --input "$INPUT" --output_json Ekas_yolo/Ekas_yolo.json --detection-method yolo
run_markit Ekas_yolo_hk --input "$INPUT" --output_json Ekas_yolo_hk/Ekas_yolo_hk.json --detection-method yolo --housekeeping

# Optical flow only
run_markit Ekas_of --input "$INPUT" --output_json Ekas_of/Ekas_of.json --detection-method optical_flow --debug-flow --exclusion-mask "$MASK"
run_markit Ekas_of_hk --input "$INPUT" --output_json Ekas_of_hk/Ekas_of_hk.json --detection-method optical_flow --housekeeping --debug-flow --exclusion-mask "$MASK"

# Yolo and optical flow
run_markit Ekas_both --input "$INPUT" --output_json Ekas_both/Ekas_both.json --detection-method both --housekeeping --debug-flow --exclusion-mask "$MASK"
run_markit Ekas_both_hk --input "$INPUT" --output_json Ekas_both_hk/Ekas_both_hk.json --detection-method both --housekeeping --debug-flow --exclusion-mask "$MASK"
