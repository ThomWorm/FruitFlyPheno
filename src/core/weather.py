import pickle
import xarray as xr


class WeatherDataHandler:
    def __init__(self, weather_config: dict, latitude: float, longitude: float):
        # self.api_key = weather_config['api_key']
        self.cache_dir = weather_config["cache_dir"]
        self.latitude = latitude
        self.longitude = longitude

        # Set up data sources in priority order
        self.sources = sorted(weather_config["sources"], key=lambda x: x["priority"])

        # Create cache directory if needed
        self.cache_dir.mkdir(exist_ok=True)

        # Store other parameters
        self.default_vars = weather_config["default_vars"]
        self.timeout = weather_config.get("timeout", 10)

    def load_cached(self):
        """Load cached data from disk"""

        if self.cache_dir.exists():
            with open(self.cache_dir, "rb") as cache_file:
                raw_PRISM = pickle.load(cache_file)
                return raw_PRISM
        else:
            raise FileNotFoundError(f"Cache file {self.cache_dir} not found.")

    def get_recent_observed(self) -> xr.Dataset:
        return self._load_cached("recent")

    def _download_recent_data(self) -> xr.Dataset:
        """Raw API communication (private method)"""

    def _preprocess(self, data: xr.Dataset) -> xr.Dataset:
        """Business logic like unit conversion, quality checks"""
