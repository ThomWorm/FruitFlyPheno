import yaml
from pathlib import Path
from pydantic import BaseModel


class AppConfig(BaseModel):
    weather: dict
    output: dict


def load_config(file_path: str = "config/settings.yaml") -> AppConfig:
    path = Path(__file__).parent.parent.parent / file_path
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return AppConfig(**raw)
