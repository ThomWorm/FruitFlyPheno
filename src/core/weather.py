import pickle
import xarray as xr
from pathlib import Path
@dataclass
class WeatherServerClass(frozen= True)
    
        credentials: dict
@dataclass
class WeatherDataHandler:
        
    cache_dir: Path
    latitude: float
    longitude: float
    credentials: dict
    
    def load_cached(self):
        """Load cached data from disk"""

        if self.cache_dir.exists():
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
            format="netcdf"
        )

        dataset = xr.open_dataset(response)

        # Store inside the instance
        self.raw_data = dataset

        return dataset
    def _compute_bbox(self) -> tuple:
        """Compute a bounding box around the given latitude and longitude."""
        # Assuming a simple square bounding box for simplicity
          bbox = {
            "min_lat": self.latitude - delta,
            "max_lat": self.latitude + delta,
            "min_lon": self.longitude - delta,
            "max_lon": self.longitude + delta
        }
    
    def get_recent_observed(self) -> xr.Dataset:
        return self._load_cached("recent")

    def _download_recent_data(self) -> xr.Dataset:
        """Raw API communication (private method)"""

    def _preprocess(self, data: xr.Dataset) -> xr.Dataset:
        """Business logic like unit conversion, quality checks"""
