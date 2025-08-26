from pathlib import Path
import json


def read_json(path: str) -> dict:
    """retrieve JSON file from TestVids directory"""
    with open(path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    return test_data