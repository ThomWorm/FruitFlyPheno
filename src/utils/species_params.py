import os
import json
def load_species_params(species, data_path=None):
    # load json file of pre-defined species parameters
    data_path = os.path.join(data_path, "fly_models.json")
    with open(data_path) as f:
        data = json.load(f)
    species = data.get(species)
    if not species:
        return None

    return species.get("stages")
    # Handle default stage
