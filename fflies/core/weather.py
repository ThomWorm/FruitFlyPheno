import pickle
import xarray as xr
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class WeatherServerClass:

    credentials: dict

    def connect(self):
        """Placeholder method to simulate server connection."""
        if not self.credentials.get("api_key"):
            raise ValueError("API key is missing in credentials.")
        return "Connected to Weather Server"

    def fetch_data(self, bbox: tuple, variables: List[str], format: str) -> str:
        """Placeholder method to simulate data fetching."""
        return "Simulated data response"


@dataclass
class WeatherDataHandler:

    cache_dir: Path
    latitude: float
    longitude: float
    credentials: dict

    def load_cached(self):
        """Load cached data from disk"""
        # convert cache_dir to Path object
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
        if self.cache_dir.is_file() and self.cache_dir.suffix == ".pkl":
            with open(self.cache_dir, "rb") as cache_file:
                raw_PRISM = pickle.load(cache_file)
                return raw_PRISM
        else:
            raise FileNotFoundError(f"Cache file {self.cache_dir} not found.")

    def fetch_remote_data(self) -> xr.Dataset:
        """Fetch data and store it internally."""
        bbox = self.compute_bbox()  # 15 km bounding box half-width

        api_key = self.credentials.get("api_key")

        # Pseudocode for actual data fetch from your API or server:
        response = remote_weather_api.query(
            bbox=bbox,
            api_key=api_key,
            variables=["temp", "precip", "wind"],
            format="netcdf",
        )

        dataset = xr.open_dataset(response)

        # Store inside the instance
        self.raw_data = dataset

        return dataset

    def _compute_bbox(self) -> tuple:
        return None

    def get_recent_observed(self) -> xr.Dataset:
        return self._load_cached("recent")

    def _download_recent_data(self) -> xr.Dataset:
        """Raw API communication (private method)"""

    def _preprocess(self, data: xr.Dataset) -> xr.Dataset:
        """Business logic like unit conversion, quality checks"""
