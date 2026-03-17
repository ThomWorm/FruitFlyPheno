# FruitFlyPheno

## Setup Instructions

Follow these steps to set up the environment and get started with the FruitFlyPheno project:

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd FruitFlyPheno
   ```

2. **Install `uv`**
   ```bash
   pip install uv
   ```

3. **Create and Activate a Virtual Environment with uv**
   - On Linux/macOS:
     ```bash
     uv venv
     source .venv/bin/activate
     ```
   - On Windows:
     ```bash
     uv venv
     .venv\Scripts\activate
     ```

4. **Install the Project and Dependencies in Editable Mode**
   ```bash
   pip install -e .
   ```

5. **Verify Installation**
   Run the following command to ensure everything is set up correctly:
   ```bash
   fflies-SAFARIS --help
   ```

You are now ready to use the FruitFlyPheno pipeline!

## Running the Model

### CLI Entry Points

The package provides two CLI commands:

#### `fflies-SAFARIS` — Run the degree-day model

```bash
fflies-SAFARIS --input path/to/input.json [OPTIONS]
```

**Options:**

| Flag | Description |
|------|-------------|
| `--input PATH` | **(Required)** Path to input JSON file |
| `--output-path DIR` | Directory to save output files (default: `outputs`) |
| `--print-results` | Print results to the terminal as a formatted table |
| `--save-exec-dashboard` | Also save a NetCDF file for use with the dashboard |
| `--predict-from-date YYYY-MM-DD` | Truncate weather data at this date and run a forward prediction |

**Example:**
```bash
fflies-SAFARIS \
    --input config/sample_user_input_single.json \
    --output-path outputs \
    --print-results \
    --save-exec-dashboard
```

#### `fflies-dashboard` — Visualise results

```bash
fflies-dashboard path/to/results.nc [--port PORT]
```

**Options:**

| Argument | Description |
|----------|-------------|
| `netcdf_path` | **(Required)** Path to the NetCDF file produced by `fflies-SAFARIS --save-exec-dashboard` |
| `--port PORT` | Port to serve the dashboard on (default: `5006`) |

**Example:**
```bash
fflies-dashboard outputs/medfly_off_santa_clara_results.nc --port 5006
# Then open http://localhost:5006 in your browser
```

### Input JSON Format

The model takes a JSON array where each element describes a single detection event. Example (`config/sample_user_input_single.json`):

```json
[{
    "user_id": "user_002",
    "unique_id": "off_santa_clara",
    "latitude": 37.344173,
    "longitude": -121.990698,
    "detection_date": "2023-07-20",
    "species": "off",
    "generations": 3
}]
```

**Required fields:**

| Field | Type | Description |
|-------|------|-------------|
| `unique_id` | string | Unique identifier used in output filenames |
| `latitude` | float | Latitude of the detection site (North America coverage) |
| `longitude` | float | Longitude of the detection site |
| `detection_date` | string | Date of first detection (`YYYY-MM-DD`, between 2000-01-01 and today) |
| `species` | string | Species code (e.g. `medfly`, `mexfly`, `off`, `queensland`) |
| `generations` | int | Number of generations to model (positive integer) |

### Supported Species

| Code | Common Name |
|------|-------------|
| `medfly` | Mediterranean Fruit Fly |
| `mexfly` | Mexican Fruit Fly |
| `off` | Oriental Fruit Fly |
| `queensland` | Queensland Fruit Fly |
| `z_tau` | Zeugodacus tau |
| `z_cucurbitae` | Zeugodacus cucurbitae (Melon Fly) |
| `peach` | Bactrocera zonata (Peach Fruit Fly) |
| `guava` | Bactrocera correcta (Guava Fruit Fly) |

### Output Files

Running `fflies-SAFARIS` produces:

- `outputs/results_<input_filename>.json` — Predicted generation completion dates in JSON format.
- `outputs/<species>_<unique_id>_results.nc` — NetCDF dataset (only when `--save-exec-dashboard` is used); required by `fflies-dashboard`.

For more details, see `fflies/main.py`.
