from dataclasses import dataclass
import xarray as xr
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import timedelta

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

    def create_json(self, filename: str):
        """
        Save the xarray Dataset/DataArray as a JSON file.

        Parameters:
            filename (str): The name of the JSON file to create.
        """
        # ==============
        # extract mean completions for each generation
        # ==============
        mean_completion_dates = []
        for gen_i in range(
            1, len(self.data["generation"]) + 1
        ):  # Dynamically adjust range based on available generations
            generation_data = self.data["days_to_completion"].sel(
                latitude=self.latitude,
                longitude=self.longitude,
                generation=gen_i,
                method="nearest",
            )
            generation_data.values

            mean_duration = generation_data.values.mean()
            mean_completion_date = (
                self.detection_date + timedelta(days=mean_duration)
            ).strftime("%Y-%m-%d")
            mean_completion_dates.append(mean_completion_date)
        # ==============
        # calculate latest likely completion date
        # ==============
        if self.all_historical == 0:
            latest_completion_date = self._MCMC_latest_completion_date()
        else:
            latest_completion_date = self.data["days_to_completion"].max().values
            latest_completion_date = (
                self.detection_date + timedelta(days=latest_completion_date)
            ).strftime("%Y-%m-%d")
        # ==============
        # Create the JSON structure
        # ==============
        output_json = {
            "detection_date": self.detection_date,
            "species": self.species,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "generations": {
                f"F{i+1}": {
                    "mean_completion_days": mean_completion_dates[i],
                    "latest_completion_date": (
                        latest_completion_date
                        if i == len(mean_completion_dates) - 1
                        else None
                    ),
                }
                for i in range(len(mean_completion_dates))
            },
        }
        # write the json to a file
        with open(filename, "w") as json_file:
            json.dump(output_json, json_file, indent=4)

    def plot(self, var_name: str = None):
        """
        Interactive map-based plot of the dataset using OpenStreetMap underlay.

        Parameters:
        -----------
        var_name : str, optional
            Name of the variable in self.data to plot. If None, the first data variable is used.

        Returns:
        --------
        pn.Column
            A Panel layout object which can be shown, served, or saved.
        """
        da = (
            self.data[var_name]
            if var_name
            else next(iter(self.data.data_vars.values()))
        )
        precomputed = {"Likely Completion Date": da.mean(dim="year")}

        # Ensure lat/lon are sorted
        da = da.sortby(["latitude", "longitude"])

        years = da.coords["year"].values
        generations = da.coords["generation"].values
        year_labels = {f"sim{y}": y for y in da.coords["year"].values}
        custom_layers = {"Mean (All Years)": "mean"}

        year_options = {**year_labels, **custom_layers}
        year_select = pn.widgets.Select(name="Year / Layer", options=year_options)
        # Widgets
        # year_select = pn.widgets.Select(name="Year", options=years.tolist())
        gen_select = pn.widgets.Select(name="Generation", options=generations.tolist())
        alpha_slider = pn.widgets.FloatSlider(
            name="Transparency", start=0.0, end=1.0, step=0.05, value=0.8
        )

        # Plotting function
        def make_plot(year_or_stat, generation, alpha):

            if year_or_stat == "mean":
                # Use the mean of all years
                sliced = precomputed["Likely Completion Date"].sel(
                    generation=generation
                )
            else:
                sliced = da.sel(year=year_or_stat, generation=generation)

            img = gv.Image(
                sliced, kdims=["longitude", "latitude"], vdims=[sliced.name]
            ).opts(
                cmap="Viridis",
                alpha=alpha,
                colorbar=True,
                width=1000,
                height=900,
                tools=["hover"],
                clim=(float(da.min()), float(da.max())),
                # projection=gv.plotting.bokeh.CRS.PlateCarree(),
            )
            tiles = gv.tile_sources.OSM.opts(alpha=1.0)
            return tiles * img

        # Bind widgets
        plot_pane = pn.bind(
            make_plot,
            year_or_stat=year_select,
            generation=gen_select,
            alpha=alpha_slider,
        )
        layout = pn.Column(
            f"## {self.species} Detection Visualization",
            pn.Row(year_select, gen_select, alpha_slider),
            plot_pane,
        )

        return layout  # Can call .show(), .save(), or .servable() on this

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
