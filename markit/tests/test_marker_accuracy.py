"""RTK accuracy (SD) columns of the marker CSVs are kept in the OpenLABEL marker objects."""

import math

from markit.markitlib.openlabel.handler import marker_gps_vec
from markit.markitlib.processing.engines import ArUcoGPSData, VisualMarkerGPSData

SURVEY_HEADER = "point_name,latitude,longitude,altitude,north,east,elevation,point_type,horizontal SD,vertical SD\n"


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def _vec(vec):
    return {entry["name"]: entry["val"] for entry in vec}


def test_survey_export_sd_columns_reach_the_marker_vec(tmp_path):
    csv = _write(tmp_path, "Saro_markers.csv", SURVEY_HEADER
                 + "lamppost_1a,57.4845,12.0089,75.24,1,2,3,Check Point,0.00199,0.00346\n"
                 + "lamppost_1c,57.4846,12.0090,75.10,1,2,3,Check Point,0.00210,0.00350\n")
    markers = VisualMarkerGPSData(csv)
    assert markers.gps_data[1]["a"]["sigma_h_m"] == 0.00199

    vec = _vec(marker_gps_vec("markerID", 1, markers.gps_data[1], "Saro"))
    assert vec["markerID"] == ["1a", "1c"]
    assert vec["sigma_h_m"] == ["0.00199", "0.0021"]
    assert vec["sigma_v_m"] == ["0.00346", "0.0035"]


def test_short_column_names_and_missing_values(tmp_path):
    csv = _write(tmp_path, "Gbg_aruco.csv", "point_name,latitude,longitude,altitude,horiz SD,vert SD\n"
                 "aruco_24a,57.48,12.00,72.7,0.02,inf\n"
                 "aruco_24c,57.48,12.00,72.5,,\n")
    markers = ArUcoGPSData(csv)
    corners = markers.get_gps_data(24)
    assert corners["a"]["sigma_h_m"] == 0.02
    assert math.isinf(corners["a"]["sigma_v_m"])
    assert "sigma_h_m" not in corners["c"]

    vec = _vec(marker_gps_vec("arucoID", 24, corners, "Gbg"))
    assert vec["sigma_h_m"] == ["0.02", "inf"]  # unknown corner stays unknown, not 0


def test_csv_without_sd_columns_gives_the_old_vec(tmp_path):
    csv = _write(tmp_path, "Gbg_aruco.csv", "point_name,latitude,longitude,altitude\naruco_7a,57.48,12.00,72.7\n")
    vec = marker_gps_vec("arucoID", 7, ArUcoGPSData(csv).get_gps_data(7), "Gbg")
    assert [entry["name"] for entry in vec] == ["arucoID", "long", "lat", "alt", "description"]
