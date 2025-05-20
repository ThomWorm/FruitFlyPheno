import yaml
from pathlib import Path


def load_config(file_path: str) -> dict:
    path = Path(__file__).parent.parent / file_path
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return raw
