# Test Videos

Test videos for SAVANT (OpenLabel files created by *markit* and can be loaded in *edit*).

## Contents

### Kraklanda_short/
Short test clip for basic YOLO detection testing.

Processed with:
```bash
./run_markit --input ../TestVids/Kraklanda_short/Kraklanda_short.mp4 --output_json ../TestVids/Kraklanda_short/Kraklanda_short.json
```

### Saro_roundabout/
Roundabout scene with ArUco markers for GPS coordinate testing.

Processed with:
```bash
./run_markit --input ../TestVids/Saro_roundabout/Saro_roundabout.mp4 --output_json ../TestVids/Saro_roundabout/Saro_roundabout.json --aruco-csv ../TestVids/Saro_roundabout/GbgSaroRound_coords.csv --housekeeping --static-mark
```
