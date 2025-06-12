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

    def create_json(self):
        """
        Return the output JSON structure as a dictionary instead of writing to a file.
        """
        # Convert detection_date string to datetime.date
        detection_date_dt = datetime.datetime.strptime(
            self.detection_date, "%Y-%m-%d"
        ).date()
        # ==============
        # extract mean and max completions for each generation
        # ==============
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
            # Mean
            mean_duration = generation_data.values.mean().astype(int)
            mean_duration = int(mean_duration)
            mean_completion_date = (
                detection_date_dt + timedelta(days=mean_duration)
            ).strftime("%Y-%m-%d")
            mean_completion_dates.append(mean_completion_date)
            # Max
            max_duration = generation_data.values.max().astype(int)
            max_duration = int(max_duration)
            max_completion_date = (
                detection_date_dt + timedelta(days=max_duration)
            ).strftime("%Y-%m-%d")
            max_completion_dates.append(max_completion_date)
        # ==============
        # calculate latest likely completion date
        # ==============
        """
        if self.all_historical == 0:
            latest_completion_date = self._MCMC_latest_completion_date()
        else:
            latest_completion_date = self.data["days_to_completion"].max().values
            latest_completion_date = int(latest_completion_date)
            latest_completion_date = (
                detection_date_dt + timedelta(days=latest_completion_date)
            ).strftime("%Y-%m-%d")
        """
        # ==============
        # Create the JSON structure
        # ==============
        output_json = {
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
        Interactive map-based plot of the dataset using OpenStreetMap underlay.

        Parameters:
        -----------
        var_name : str, optional
            Name of the variable in self.data to plot. If None, the first data variable is used.
        save_path : str, optional
            If provided, saves the plot as an HTML file to this path.

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
        precomputed = {
            "Likely Completion Date": da.mean(dim="year"),
            "Range of Likely Completion Dates": da.max(dim="year") - da.min(dim="year"),
        }

        # Ensure lat/lon are sorted
        da = da.sortby(["latitude", "longitude"])

        years = da.coords["year"].values
        generations = da.coords["generation"].values
        year_labels = {f"sim{y}": y for y in da.coords["year"].values}
        # Place custom_layers first in the options
        custom_layers = {"Mean (All Years)": "mean", "Range (All Years)": "range"}
        # Combine so that mean/range are at the top, followed by years
        year_options = {**custom_layers, **year_labels}
        # Set "mean" as the default value for the year/layer select
        select_styles = {"color": "#FF3C00"}  # Gold text for good contrast

        year_select = pn.widgets.Select(
            name="Year / Layer",
            options=year_options,
            value="mean",
            styles=select_styles,
        )
        gen_select = pn.widgets.Select(
            name="Generation",
            options=generations.tolist(),
            value=generations[2] if len(generations) > 2 else generations[0],
            styles=select_styles,
        )
        alpha_slider = pn.widgets.FloatSlider(
            name="Transparency",
            start=0.0,
            end=1.0,
            step=0.05,
            value=0.8,
            styles=select_styles,
        )

        # Precompute clim per generation for mean and range
        clim_per_gen = {}
        clim_range_per_gen = {}
        for gen in generations:
            gen_data = da.sel(generation=gen)
            clim_per_gen[gen.item() if hasattr(gen, "item") else gen] = (
                float(gen_data.min()),
                float(gen_data.max()),
            )
            range_data = precomputed["Range of Likely Completion Dates"].sel(
                generation=gen
            )
            clim_range_per_gen[gen.item() if hasattr(gen, "item") else gen] = (
                float(range_data.min()),
                float(range_data.max()),
            )

        # Plotting function
        def make_plot(year_or_stat, generation, alpha):
            # Ensure generation is the correct type for lookup
            gen_key = generation.item() if hasattr(generation, "item") else generation

            if year_or_stat == "mean":
                sliced = precomputed["Likely Completion Date"].sel(
                    generation=generation
                )
                clim = clim_per_gen[gen_key]
                cmap = "Viridis"
            elif year_or_stat == "range":
                sliced = precomputed["Range of Likely Completion Dates"].sel(
                    generation=generation
                )
                clim = clim_range_per_gen[gen_key]
                cmap = "Magma"
            else:
                sliced = da.sel(year=year_or_stat, generation=generation)
                clim = clim_per_gen[gen_key]
                cmap = "Viridis"

            img = gv.Image(
                sliced, kdims=["longitude", "latitude"], vdims=[sliced.name]
            ).opts(
                cmap=cmap,
                alpha=alpha,
                colorbar=True,
                width=1000,
                height=900,
                tools=["hover"],
                clim=clim,
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
            f"## {self.species} Detection on {self.detection_date} at ({self.latitude}, {self.longitude})",
            pn.Row(
                year_select, gen_select, alpha_slider, sizing_mode="stretch_width"
            ),  # Set sizing_mode here
            plot_pane,
            sizing_mode="stretch_width",  # Already set here
        )

        if save_path is not None:
            layout.save(save_path, embed=True)
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
    pn.serve(panel_obj, port=port, show=open_browser)
