from dataclasses import dataclass
import xarray as xr
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import timedelta
import datetime  # Add this import

# ---
import panel as pn
import holoviews as hv
import geoviews as gv

gv.extension("bokeh")
hv.extension("bokeh")


@dataclass
class FfliesOutput:
    data: xr.Dataset  # Accepts an xarray Dataset or DataArray
    latitude: float  # Latitude for the data
    longitude: float  # Longitude for the data
    detection_date: str
    generations: dict
    species: str
    all_historical: int
    unique_id: str = None  # Optional unique identifier for the output

    def _extract_completion_dates(self):
        """
        Helper method to extract mean and max completion dates for all generations.
        Returns tuple of (mean_completion_dates, max_completion_dates).
        """
        # Convert detection_date string to datetime.date
        detection_date_dt = datetime.datetime.strptime(
            self.detection_date, "%Y-%m-%d"
        ).date()
        
        mean_completion_dates = []
        max_completion_dates = []
        for gen_i in range(
            1, len(self.data["generation"]) + 1
        ):  # Dynamically adjust range based on available generations
            generation_data = self.data["days_to_completion"].sel(
                latitude=self.latitude,
                longitude=self.longitude,
                generation=gen_i,
                method="nearest",
            )
            # Mean across years
            mean_duration = int(generation_data.mean(dim='year').values)
            mean_completion_date = (
                detection_date_dt + timedelta(days=mean_duration)
            ).strftime("%Y-%m-%d")
            mean_completion_dates.append(mean_completion_date)
            # Max across years
            max_duration = int(generation_data.max(dim='year').values)
            max_completion_date = (
                detection_date_dt + timedelta(days=max_duration)
            ).strftime("%Y-%m-%d")
            max_completion_dates.append(max_completion_date)
        
        return mean_completion_dates, max_completion_dates

    def create_csv(self):
        """
        Return the output data as a list of dictionaries formatted for CSV export.
        Each row represents a single simulation result.
        Structure: unique_id first, then metadata, then completion dates, then days to completion.
        """
        mean_completion_dates, max_completion_dates = self._extract_completion_dates()
        
        # Convert detection_date to datetime for calculating days
        detection_date_dt = datetime.datetime.strptime(
            self.detection_date, "%Y-%m-%d"
        ).date()
        
        # Create a single row with unique_id first, then metadata, then generation dates
        row = {
            "unique_id": self.unique_id if self.unique_id else "",
            "detection_date": self.detection_date,
            "species": self.species,
            "latitude": self.latitude,
            "longitude": self.longitude,
        }
        
        # Add generation completion dates
        for i in range(len(mean_completion_dates)):
            row[f"F{i+1}_likely_completion_date"] = mean_completion_dates[i]
            if i == len(mean_completion_dates) - 1:  # Only add max for last generation
                row[f"F{i+1}_latest_likely_completion_date"] = max_completion_dates[i]
        
        # Add days to F3 completion (last generation)
        last_gen_idx = len(mean_completion_dates) - 1
        f3_mean_date = datetime.datetime.strptime(mean_completion_dates[last_gen_idx], "%Y-%m-%d").date()
        f3_max_date = datetime.datetime.strptime(max_completion_dates[last_gen_idx], "%Y-%m-%d").date()
        
        row[f"F{last_gen_idx+1}_days_to_likely_completion"] = (f3_mean_date - detection_date_dt).days
        row[f"F{last_gen_idx+1}_days_to_latest_likely_completion"] = (f3_max_date - detection_date_dt).days
        
        return [row]  # Return as list for consistency with CSV writing

    def create_json(self):
        """
        Return the output JSON structure as a dictionary instead of writing to a file.
        """
        mean_completion_dates, max_completion_dates = self._extract_completion_dates()
        
        # ==============
        # Create the JSON structure
        # ==============
        output_json = {
            "unique_id": self.unique_id if self.unique_id else "",
            "detection_date": self.detection_date,
            "species": self.species,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "generations": {
                f"F{i+1}": (
                    {
                        "likely_completion_date": mean_completion_dates[i],
                        "latest_likely_completion_date": max_completion_dates[i],
                    }
                    if i == len(mean_completion_dates) - 1
                    else {"likely_completion_date": mean_completion_dates[i]}
                )
                for i in range(len(mean_completion_dates))
            },
        }

        return output_json

    def plot(self, var_name: str = "days_to_completion", save_path: str = None):
        """
        This method no longer launches the dashboard. Please use the separate dashboard app to visualize model outputs.
        It returns the requested DataArray for external use.

        Parameters:
        -----------
        var_name : str, optional
            Name of the variable in self.data to return. If None, the first data variable is used.
        save_path : str, optional
            (Unused) Previously used to save a plot as HTML.

        Returns:
        --------
        xr.DataArray
            The requested DataArray for visualization in the dashboard app.
        """
        if var_name:
            return self.data[var_name]
        else:
            return next(iter(self.data.data_vars.values()))

    def _extract_point(self):
        """
        Helper method to extract data from the coordinate.
        Assumes `coordinate` is a neighborhood window and extracts the central point.
        """
        # Example logic to extract the central point from a neighborhood window
        return self.data.sel(lat=self.latitude, lon=self.longitude)

    def _MCMC_latest_completion_date(self):
        """
        Calculate the latest likely completion date based on the MCMC model.
        """
        # Example logic to calculate the latest likely completion date
        # This is a placeholder and should be replaced with actual MCMC logic
        # TODO implement MCMC logic
        return self.data["days_to_completion"].max().values

    def to_netcdf(self, path):
        """
        Write the output data to a NetCDF file, including all relevant metadata and computed layers.

        Parameters
        ----------
        path : str or Path
            Path to the output NetCDF file.
        """
        ds = self.data.copy()
        # Add metadata as global attributes
        ds.attrs["detection_date"] = self.detection_date
        ds.attrs["species"] = self.species
        ds.attrs["latitude"] = self.latitude
        ds.attrs["longitude"] = self.longitude
        ds.attrs["unique_id"] = self.unique_id if self.unique_id is not None else ""
        ds.attrs["generations"] = json.dumps(self.generations)
        # --- Add computed layers ---
        # Most Likely Completion Date (mean over year)
        if "days_to_completion" in ds:
            print(days_to_completion := ds["days_to_completion"])
            ds["most_likely_completion_date"] = ds["days_to_completion"].mean(
                dim="year"
            )
            ds["latest_likely_completion_date"] = ds["days_to_completion"].max(
                dim="year"
            )
            ds["range_of_completion_dates"] = ds["days_to_completion"].max(
                dim="year"
            ) - ds["days_to_completion"].min(dim="year")
        # Write to NetCDF
        print(ds)
        ds.to_netcdf(path)


def dict_to_table(d):
    # Flatten dict for tabulate
    rows = []
    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, dict):
                    for k3, v3 in v2.items():
                        rows.append([f"{k}.{k2}.{k3}", v3])
                else:
                    rows.append([f"{k}.{k2}", v2])
        else:
            rows.append([k, v])
    return rows


def dict_to_table_multi(outputs):
    rows = []
    for entry in outputs:
        prefix = entry["species"]
        for k, v in entry["output"].items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, dict):
                        for k3, v3 in v2.items():
                            rows.append([f"{prefix}.{k}.{k2}.{k3}", v3])
                    else:
                        rows.append([f"{prefix}.{k}.{k2}", v2])
            else:
                rows.append([f"{prefix}.{k}", v])
    return rows


def serve_panel(panel_obj, port=5006, open_browser=True):
    """
    Serve a Panel object using panel.serve.

    Parameters:
    -----------
    panel_obj : pn.Column or pn.panel
        The Panel object to serve.
    port : int
        The port to serve on.
    open_browser : bool
        Whether to open the browser automatically.
    """
    pn.serve(panel_obj, port=port, show=open_browser, threaded=True, blocking=False)
