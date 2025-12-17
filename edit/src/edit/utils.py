"""Module containing useful util functions"""

import json


def read_json(file_path: str) -> dict:
    """Reads a JSON file and returns its contents as a dictionary."""
    with open(file_path, "r") as f:
        return json.load(f)
