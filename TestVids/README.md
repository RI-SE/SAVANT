# Test Videos

Test videos for SAVANT (OpenLabel files created by *markit* and can be loaded in *edit*).

All OpenLABEL files were made with markit 2.6.4 by `run_markit_testvids.sh` (see the script for the exact
options): YOLO + optical flow, housekeeping, static/jitter marking, measured error model (box σ), and VLM
scene tags. Per video: `<name>_2.6.4_openlabel.json` (edit prefers JSON names containing "openlabel"),
`.prov.json` (provenance) and `.decisions.json` (housekeeping/detection decision log).

## Contents

### Kraklanda_short/
Short clip (89 frames) for basic detection testing.

### Saro_roundabout/
Roundabout scene (300 frames) with ArUco markers and visual markers for GPS coordinate testing
(`Saro_aruco.csv`, `Saro_visualmarkers.csv`) and an exclusion mask.

### Ekas_T/
T-junction (300 frames) with an exclusion mask; cars in tree shadow are found by optical flow only.
