#!/usr/bin/env bash
# Re-run markit (all features + VLM) on the three test videos; outputs get a _<version> suffix next to the inputs
# (OpenLABEL as <name>_<version>_openlabel.json, the name edit prefers when a folder has several JSON files).
# Required env: VLM_URL (OpenAI-compatible server), VLM_MODEL (vision model name), SAVANT_VLM_API_KEY if the server
# needs one. Optional: WEIGHTS (default markit/markit_yolo.pt, which has a fitted error_model.json next to it),
# SUFFIX (default 2.6.4).
set -euo pipefail

: "${VLM_URL:?set VLM_URL, e.g. http://localhost:8000}"
: "${VLM_MODEL:?set VLM_MODEL to a vision-capable model name on that server}"
SUFFIX=${SUFFIX:-2.6.4}

cd "$(dirname "$0")/.."  # SAVANT repo root
WEIGHTS=${WEIGHTS:-markit/markit_yolo.pt}
[[ -f ${WEIGHTS%/*}/error_model.json ]] || echo "WARNING: no error_model.json next to $WEIGHTS (default sigma)" >&2

common=(--weights "$WEIGHTS" --detection-method both --housekeeping --static-mark --jitter-mark
    --vlm --vlm-url "$VLM_URL" --vlm-model "$VLM_MODEL"
    --vlm-timeout 600 --vlm-max-resolution 1080 --vlm-rationale)

run() {
    local name=$1; shift
    local dir=TestVids/$name
    local out=$dir/${name}_$SUFFIX
    echo "=== $name ($(date +%T))"
    uv run markit --input "$dir/$name.mp4" --output_json "${out}_openlabel.json" \
        --provenance "$out.prov.json" --decision-log "$out.decisions.json" \
        "${common[@]}" "$@" 2>&1 | tee "$out.log"
    scrub "$out.prov.json"
}

# Provenance records user, host, home paths and the VLM URL; replace them so the example can be shared.
scrub() {
    python3 - "$1" "$HOME" "$VLM_URL" "${VLM_URL#*://}" "$(id -un)" "$(hostname)" <<'EOF'
import sys
path, home, url, url_host, user, host = sys.argv[1:]
text = open(path).read()
for old, new in [(f"_{user}_{host}", "_<redacted>"), (home, "<redacted>"), (url, "<redacted>"), (url_host, "<redacted>"),
                 (f'"{user}"', '"<redacted>"'), (f'"{host}"', '"<redacted>"')]:
    text = text.replace(old, new)
open(path, "w").write(text)
EOF
}

run Kraklanda_short --vlm-interval 15 --vlm-max-frames 6
run Saro_roundabout --exclusion-mask TestVids/Saro_roundabout/Saro_roundabout_mask.png \
    --aruco-csv TestVids/Saro_roundabout/Saro_aruco.csv \
    --visual-markers TestVids/Saro_roundabout/Saro_visualmarkers.csv \
    --vlm-interval 30 --vlm-max-frames 10
run Ekas_T --exclusion-mask TestVids/Ekas_T/Ekas_T_mask.png --vlm-interval 30 --vlm-max-frames 10
echo "=== done ($(date +%T))"
