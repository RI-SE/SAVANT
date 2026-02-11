#!/usr/bin/env bash
# Example markit commands using TestVids/Ekas_T test data.
# Run from markit/examples/.
#
# Usage:
#   ./create_examples_ekas.sh                    # JSON output only
#   ./create_examples_ekas.sh --videos           # JSON + annotated video output (recompressed with HEVC)
#   ./create_examples_ekas.sh --debug            # JSON + verbose/debug logging saved to <dir>/debug.log
#   ./create_examples_ekas.sh --videos --debug   # Both
set -euo pipefail

INPUT_DIR="../../TestVids/Ekas_T"
INPUT="${INPUT_DIR}/Ekas_T.mp4"
MASK="${INPUT_DIR}/Ekas_T_mask.png"

VIDEOS=false
DEBUG=false
for arg in "$@"; do
  case "$arg" in
    --videos) VIDEOS=true ;;
    --debug)  DEBUG=true ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

recompress() {
  ffmpeg -y -i "$1" -c:v hevc_nvenc -cq 28 -c:a copy "${1%.mp4}_tmp.mp4"
  mv "${1%.mp4}_tmp.mp4" "$1"
}

run_markit() {
  local dir="$1"
  shift
  mkdir -p "$dir"

  local debug_flags=()
  if $DEBUG; then
    debug_flags=(--verbose --verbose-conflicts)
  fi

  if $VIDEOS; then
    if $DEBUG; then
      markit --output_video "$dir/$dir.mp4" "${debug_flags[@]}" "$@" 2>&1 | tee "$dir/debug.log"
    else
      markit --output_video "$dir/$dir.mp4" "$@"
    fi
    recompress "$dir/$dir.mp4"
  else
    if $DEBUG; then
      markit "${debug_flags[@]}" "$@" 2>&1 | tee "$dir/debug.log"
    else
      markit "$@"
    fi
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
