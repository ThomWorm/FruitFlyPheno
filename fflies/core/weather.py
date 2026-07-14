import pickle
import xarray as xr
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
import datetime

'''
@dataclass
class WeatherServerClass:

    credentials: dict
    urls = [
    "http://thredds.northwestknowledge.net:8080/thredds/dodsC/agg_met_tmmx_1979_CurrentYear_CONUS.nc",# add #fillmismatch to URL if broken
    "http://thredds.northwestknowledge.net:8080/thredds/dodsC/agg_met_tmmn_1979_CurrentYear_CONUS.nc",
]
data = xr.open_mfdataset(gridmet_URL)
    def connect(self):
        """Placeholder method to simulate server connection."""
        if not self.credentials.get("api_key"):
            raise ValueError("API key is missing in credentials.")
        return "Connected to Weather Server"

    def fetch_data(self, bbox: tuple, variables: List[str], format: str) -> str:
        """Placeholder method to simulate data fetching."""
        return "Simulated data response"

'''


@dataclass
class WeatherDataHandler:

    cache_dir: Optional[Path] = None
    gridmet_urls: list[str] = field(
        default_factory=lambda: [
            "http://thredds.northwestknowledge.net:8080/thredds/dodsC/agg_met_tmmx_1979_CurrentYear_CONUS.nc#fillmismatch",  # add #fillmismatch to URL if broken
            "http://thredds.northwestknowledge.net:8080/thredds/dodsC/agg_met_tmmn_1979_CurrentYear_CONUS.nc#fillmismatch",
        ]
    )
    bounding_box: Optional[tuple] = None  # Store bounding box as a class variable

    def __post_init__(self):
        """Initialize the WeatherDataHandler by pinging URLs and retrieving bounding box."""
        self.bounding_box = self._get_dataset_bbox()

    def _get_dataset_bbox(self) -> tuple:
        """
        Ping the GridMET URLs and retrieve the bounding box of the served datasets.
        Returns a tuple (lat_min, lat_max, lon_min, lon_max).
        """
        try:
            # Open the dataset to retrieve metadata
            ds = xr.open_dataset(self.gridmet_urls[0], decode_cf=False)
            lat_min, lat_max = ds["lat"].min().item(), ds["lat"].max().item()
            lon_min, lon_max = ds["lon"].min().item(), ds["lon"].max().item()
            return lat_min, lat_max, lon_min, lon_max
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve bounding box: {e}")

    def is_within_bbox(self, latitude: float, longitude: float) -> bool:
        """
        Check if the given latitude and longitude are within the bounding box.
        Returns True if within bounds, False otherwise.
        """
        if self.bounding_box is None:
            raise ValueError(
                "Bounding box is not defined. Please initialize the handler."
            )

        lat_min, lat_max, lon_min, lon_max = self.bounding_box
        return lat_min <= latitude <= lat_max and lon_min <= longitude <= lon_max

    def _open_gridmet_dataset(self):
        """Open the GridMET tmax and tmin datasets as a combined xarray dataset, removing CRS and disabling chunking along 't'."""
        ds = xr.open_mfdataset(
            self.gridmet_urls,
            combine="by_coords",
            chunks={"lat": 50, "lon": 50, "day": -1},
        )
        # Remove CRS dimension if present

        self.gridmet_data = ds
        # convert kelvin to celsius
        # convert weather data from kelvin to celsius

        return ds

    def _rename_variables(self, ds: xr.Dataset) -> xr.Dataset:
        """Rename variables in the dataset for consistency and remove CRS."""
        ds = ds.rename(
            {
                "daily_maximum_temperature": "tmax",
                "daily_minimum_temperature": "tmin",
                "day": "t",
                "lat": "latitude",
                "lon": "longitude",
            }
        )

        return ds

    def load_cached(self):
        """Load cached data from disk"""
        # convert cache_dir to Path object
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
        path = self.cache_dir
        if path.is_dir():
            # Default to pred_cache.pkl inside the directory
            path = path / "pred_cache.pkl"
        if path.is_file() and path.suffix == ".pkl":
            with open(path, "rb") as cache_file:
                raw_PRISM = pickle.load(cache_file)
                return raw_PRISM
        else:
            raise FileNotFoundError(f"Cache file {path} not found.")

    def _compute_bbox(self) -> tuple:
        return None

    def get_recent_observed(self) -> xr.Dataset:
        return self._load_cached("recent")

    def _download_recent_data(self) -> xr.Dataset:
        """Raw API communication (private method)"""

    def _preprocess(self, data: xr.Dataset) -> xr.Dataset:
        """Business logic like unit conversion, quality checks"""

    def fetch_data_gridmet(
        self,
        latitude,
        longitude,
        time_range=None,
        use_buffer=False,
        buffer_deg=None,
    ):
        """
        Fetch a subset of GridMET data for a given lat/lon point (with optional buffer and time range).
        Returns an xarray.Dataset with tmax and tmin for the region and time.
        buffer_deg: spatial buffer in degrees (default 0.125 ~ 15km)
        time_range: (start_date, end_date) as strings or np.datetime64 (optional)
        """
        # Open the dataset if not already loaded
        if not hasattr(self, "gridmet_data"):
            self._open_gridmet_dataset()
        ds = self.gridmet_data

        # Define bounding box
        if buffer_deg is None:
            if use_buffer:
                buffer_deg = 0.4  # Default buffer in degrees (~44km)
            else:
                buffer_deg = 0.1
        # Ensure slice direction matches coordinate order in dataset
        lat_min, lat_max = latitude - buffer_deg, latitude + buffer_deg
        lon_min, lon_max = longitude - buffer_deg, longitude + buffer_deg
        # Check if lat and lon are ascending or descending in the dataset
        lat_asc = ds.lat[0] < ds.lat[-1]
        lon_asc = ds.lon[0] < ds.lon[-1]
        lat_slice = slice(lat_min, lat_max) if lat_asc else slice(lat_max, lat_min)
        lon_slice = slice(lon_min, lon_max) if lon_asc else slice(lon_max, lon_min)
        subset = ds.sel(
            lat=lat_slice,
            lon=lon_slice,
        )
        if time_range is not None:
            subset = subset.sel(day=slice(time_range[0], time_range[1]))
        else:
            subset = subset.sel(
                day=slice(datetime.datetime(2000, 1, 1), datetime.datetime.now())
            )
        subset = self._rename_variables(subset)

        return subset
