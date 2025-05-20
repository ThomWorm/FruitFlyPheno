import os
import json
from pathlib import Path


def load_species_params(species):
    # load json file of pre-defined species parameters
    # data_path = os.path.join(data_path, "fly_models.json")
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    data_path = os.path.join(PROJECT_ROOT, "config", "fly_models.json")
    with open(data_path) as f:
        data = json.load(f)
    species = data.get(species)
    if not species:
        return None

    return species.get("stages")
    # Handle default stage
